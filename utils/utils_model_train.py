import os  
from .utils1 import *
from .pyce1 import *
from copy import deepcopy
import traceback
import json
import warnings
warnings.filterwarnings("ignore")

import pickle
from functools import partial
import lightgbm as lgb 
from lightgbm import early_stopping
from sklearn import  metrics
# from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn2pmml import PMMLPipeline,sklearn2pmml
import optuna
# optuna.logging.set_verbosity(optuna.logging.WARNING) ## 只打印警告和异常，不打印中间结果
### 使用LR模型训练
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
import shutil

# 将评估指标函数移到模块级别，避免pickle序列化问题
# 1) KS评估指标函数
def ks_score(y_true, y_pred_proba):
    """KS评估指标函数"""  
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba, pos_label=1)
    return abs(fpr - tpr).max()

# 2) AUC评估指标函数
def auc_score(y_true, y_pred_proba):
    """AUC评估指标函数"""
    auc_tmp = roc_auc_score(y_true, y_pred_proba)
    return auc_tmp

# 3）基于预测概率做等频 10 箱，输出：
#   - overall_bad_rate
#   - 每箱 bad_rate 列表（从高分到低分）
#   - head/tail 10% lift
def decile_lift_and_bad_rates(y_true, y_score, n_bins=10):
    df = pd.DataFrame({"y": y_true, "score": y_score})
    # 按 score 从高到低排序（高分 [概率]= 风险高）
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    
    # 利用 rank 做等频分箱
    df["rank"] = df["score"].rank(method="first", ascending=False)
    df["decile"] = pd.qcut(df["rank"], q=n_bins, labels=False, duplicates="drop") #duplicates="drop" 在样本量小或重复值多时避免因无法分箱报错。
    
    overall_bad_rate = df["y"].mean()  # 1=坏人的占比
    
    bad_rates = []
    for d in range(df["decile"].max() + 1):
        sub = df[df["decile"] == d]
        if len(sub) == 0:
            bad_rates.append(np.nan)
        else:
            bad_rates.append(sub["y"].mean())
    
    # head 10% (decile=0)、tail 10% (最后一箱)
    head_bad_rate = bad_rates[0]
    tail_bad_rate = bad_rates[-1]
    
    if overall_bad_rate == 0:
        lift_head = np.nan
        lift_tail = np.nan
    else:
        lift_head = head_bad_rate / overall_bad_rate
        lift_tail = tail_bad_rate / overall_bad_rate
    
    return overall_bad_rate, bad_rates, lift_head, lift_tail


# 4）单调性约束（坏人占比随分数从高到低单调递减）
#  bad_rates: 从 decile 0 (最高分) -> decile 9 (最低分) 的坏人占比列表
# 输出：monotonic_violation >= 0
#   - 如果完全单调递减（允许相等），则 monotonic_violation = 0
#   - 如果有违背单调的地方，monotonic_violation > 0，越大表示越严重
def monotonicity_violation(bad_rates):
    bad_rates = np.array(bad_rates, dtype=float)
    # 去掉 NaN
    valid = ~np.isnan(bad_rates)
    br = bad_rates[valid]
    if len(br) <= 1:
        return 0.0
    
    # 理想情况：br[i] >= br[i+1] （非增）
    diffs = br[1:] - br[:-1]  # 如果 >0 就是违背单调递减
    max_violation = np.max(np.maximum(diffs, 0.0))  # 最大“向上跳”的幅度
    return float(max_violation)

# LR模型超参数寻优
def train_lr_with_optuna(X_train, y_train, X_test, y_test, dic_p):
    n_trials = dic_p["n_trials"]
    ks_threshold_min = dic_p['ks_threshold_min']
    ks_gap_threshold = dic_p['ks_gap_threshold']
    config_model_params = dic_p['model_param']  # 默认参数

    def objective(trial):
        C = trial.suggest_loguniform("C", 1e-3, 100)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        solver = "liblinear" if penalty == "l1" else "lbfgs"
        model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000)

        model.fit(X_train, y_train)
        pred_train = model.predict_proba(X_train)[:, 1]
        pred_test = model.predict_proba(X_test)[:, 1]
        ks_train = ks_score(y_train, pred_train)
        ks_test = ks_score(y_test, pred_test)

        return ks_test if (ks_test >= ks_threshold_min and abs(ks_train - ks_test) <= ks_gap_threshold and ks_train > ks_test) else 0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_value = study.best_trial.value
    best_params = study.best_trial.params
    
    print("模型寻优结束！")
    if best_value == 0:
        print("未能找到满足条件的参数，使用默认参数为：", config_model_params)
        best_params = config_model_params
    else:
        best_params = {**config_model_params, **best_params}
        print("寻找的最佳LR模型结果KS值", best_value, "，参数为：", best_params)

    return best_params

# LightGBM + Optuna 模型超参数寻优
def train_lgb_with_optuna(X_train, y_train, X_test, y_test, dic_p):
    IF_RUN = dic_p['RUN_HPO']
    config_model_params = dic_p['model_param']
    if  IF_RUN:
        print("=" * 60)
        print("开始模型超参数寻优 (Optuna)")
        n_trials=dic_p["n_trials"]
        ks_threshold_min = dic_p['ks_threshold_min']
        ks_gap_threshold = dic_p['ks_gap_threshold']
        print(f"   - 寻优次数: {n_trials}")
        print(f"   - KS阈值范围: [{ks_threshold_min:.3f},1.0]")
        print(f"   - KS差值阈值: {ks_gap_threshold:.3f}")
        print(f"   - 训练集样本: {len(X_train)}")
        print(f"   - 测试集样本: {len(X_test)}")

        # 定义约束函数
        def constraints(trial: optuna.Trial):
            ks_gap = trial.user_attrs["ks_gap"]
            mono_violation = trial.user_attrs["mono_violation"]
            
            # 约束1：KS gap <= 0.05
            g1 = ks_gap - 0.05
            
            # 约束2：单调性违背程度 <= 0（完全单调则 mono_violation=0，否则>0）
            g2 = mono_violation  # 已经是 >=0 的量，要求 <=0
            
            return (g1, g2)

        # 定义目标函数
        def objective(trial: optuna.Trial):
            # 获取当前试验的参数
            search_space = dic_p['model_selection_params']
            param_grid = {
                "min_split_gain": trial.suggest_int("min_split_gain", 0,10,step=1),
                "learning_rate": trial.suggest_float("learning_rate", *search_space["learning_rate"]),
                "num_leaves": trial.suggest_int("num_leaves", *search_space["num_leaves"]),
                "max_bin": trial.suggest_int("max_bin", *search_space["max_bin"]),
                "reg_alpha": trial.suggest_int("reg_alpha", *search_space["reg_alpha"]),
                "reg_lambda": trial.suggest_int("reg_lambda", *search_space["reg_lambda"]),
                "subsample": trial.suggest_float("subsample", *search_space["subsample"]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *search_space["colsample_bytree"]),
                "subsample_freq": trial.suggest_int("subsample_freq", *search_space["subsample_freq"]),
                "min_child_weight": trial.suggest_int("min_child_weight", *search_space["min_child_weight"]),
                "n_estimators": trial.suggest_categorical("n_estimators", search_space["n_estimators"]),
                "random_state": 42,
                "max_depth": trial.suggest_int("max_depth", *search_space["max_depth"]),
                "scale_pos_weight": 1.0,
            }
            
            # 训练模型
            model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', n_jobs=-1,verbose=-1, **param_grid)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
                    eval_metric=ks_metric, callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
            
            # 预测概率
            pred_train = model.predict_proba(X_train)[:, 1]
            pred_test = model.predict_proba(X_test)[:, 1]

            # 计算KS指标
            ks_train = ks_score(y_train, pred_train)
            ks_test = ks_score(y_test, pred_test)
            ks_gap = abs(ks_train - ks_test)
            # head/tail 10% lift + decile 坏人占比
            overall_bad_rate, bad_rates, lift_head10, lift_tail10 = decile_lift_and_bad_rates(y_test, pred_test, n_bins=10)
            mono_violation = monotonicity_violation(bad_rates)

            # 设置trial的属性-⑥ 把所有中间结果存到 user_attrs，后面 constraints_func 和 DataFrame 都会用到
            trial.set_user_attr("best_iteration", model.best_iteration_)
            trial.set_user_attr("ks_train", ks_train)
            trial.set_user_attr("ks_test", ks_test)
            trial.set_user_attr("params", param_grid)
            trial.set_user_attr("ks_gap", ks_gap)
            trial.set_user_attr("lift_head10", lift_head10)
            trial.set_user_attr("lift_tail10", lift_tail10)
            trial.set_user_attr("overall_bad_rate", overall_bad_rate)
            trial.set_user_attr("bad_rates", bad_rates)
            trial.set_user_attr("mono_violation", mono_violation)

            # 多目标返回：（多目标优化）
            #   目标1：ks_test（越大越好）
            #   目标2：head 10% lift（越大越好）
            #   目标3：tail 10% lift（越小越好）
            return ks_test, lift_head10, lift_tail10

        
        # 使用NSGA-II多目标优化
        sampler = optuna.samplers.NSGAIISampler(constraints_func=constraints) 
        study = optuna.create_study(
            directions=["maximize", "maximize", "minimize"],
            sampler=sampler
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=1)  # n_jobs 可并行
        
        # 多目标优化：从Pareto front中选择最佳解
        print("\n" + "=" * 60)
        print("多目标优化结果分析")
        print("=" * 60)
        
        # 获取Pareto front中的解（多目标优化返回的是best_trials列表）
        try:
            pareto_trials = study.best_trials  # 多目标优化时，best_trials 包含 Pareto front
        except AttributeError:
            # 如果没有best_trials属性，尝试从所有trials中筛选
            pareto_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        print(f"Pareto front 解数量: {len(pareto_trials)}")
        
        # 初始化状态标志
        is_optuna = "optuna_partial"
        
        if len(pareto_trials) == 0:
            # 如果没有Pareto解，使用所有trials中最好的
            print("警告：未找到Pareto front解，使用所有trials中KS_test最高的解")
            all_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(all_trials) == 0:
                raise ValueError("没有完成的trials，无法选择最佳解")
            best_trial = max(all_trials, key=lambda t: t.user_attrs.get("ks_test", -1))
            is_optuna = "optuna_not_found"
        else:
            # 从Pareto front中选择最佳解
            # 策略：优先考虑ks_test，同时满足约束条件
            valid_trials = []
            for trial in pareto_trials:
                ks_test_val = trial.user_attrs.get("ks_test", 0)
                ks_train_val = trial.user_attrs.get("ks_train", 0)
                ks_gap_val = trial.user_attrs.get("ks_gap", 1.0)
                
                # 检查是否满足约束条件
                meets_ks_threshold = ks_test_val >= ks_threshold_min
                meets_ks_gap = ks_gap_val <= ks_gap_threshold
                meets_ks_order = ks_train_val > ks_test_val
                
                if meets_ks_threshold and meets_ks_gap and meets_ks_order:
                    valid_trials.append(trial)
            
            if len(valid_trials) > 0:
                # 从满足约束条件的解中选择ks_test最高的
                best_trial = max(valid_trials, key=lambda t: t.user_attrs.get("ks_test", -1))
                print(f"从 {len(valid_trials)} 个满足约束条件的Pareto解中选择最佳解")
                is_optuna = "optuna_found"
            else:
                # 如果没有满足所有约束条件的解，选择ks_test最高的Pareto解
                best_trial = max(pareto_trials, key=lambda t: t.user_attrs.get("ks_test", -1))
                print(f"未找到满足所有约束条件的解，选择KS_test最高的Pareto解")
                is_optuna = "optuna_partial"
        
        # 提取最佳解的参数和指标
        best_params = best_trial.user_attrs["params"].copy()
        best_params["n_estimators"] = best_trial.user_attrs["best_iteration"]
        best_params = {**config_model_params, **best_params}
        
        # 获取所有指标
        ks_train = best_trial.user_attrs.get("ks_train", 0)
        ks_test = best_trial.user_attrs.get("ks_test", 0)
        ks_gap = best_trial.user_attrs.get("ks_gap", 0)
        lift_head10 = best_trial.user_attrs.get("lift_head10", 0)
        lift_tail10 = best_trial.user_attrs.get("lift_tail10", 0)
        mono_violation = best_trial.user_attrs.get("mono_violation", 0)
        overall_bad_rate = best_trial.user_attrs.get("overall_bad_rate", 0)
        
        # 输出详细结果
        print("\n" + "-" * 60)
        print("选择的最终模型参数")
        print("-" * 60)
        print(f"  KS测试集: {ks_test:.4f}")
        print(f"  KS训练集: {ks_train:.4f}")
        print(f"  KS差值: {ks_gap:.4f}")
        print(f"  Head 10% Lift: {lift_head10:.4f}")
        print(f"  Tail 10% Lift: {lift_tail10:.4f}")
        print(f"  单调性违背: {mono_violation:.4f}")
        print(f"  整体坏样本率: {overall_bad_rate:.4f}")
        print(f"  最佳迭代次数: {best_trial.user_attrs['best_iteration']}")
        
        # 检查约束条件
        print("\n" + "-" * 60)
        print("约束条件检查")
        print("-" * 60)
        constraint1_ok = ks_gap <= 0.05
        constraint2_ok = mono_violation <= 0
        condition1_ok = ks_test >= ks_threshold_min
        condition2_ok = abs(ks_train - ks_test) <= ks_gap_threshold
        condition3_ok = ks_train > ks_test
        
        print(f"  约束1 (KS gap <= 0.05): {'✓' if constraint1_ok else '✗'} ({ks_gap:.4f})")
        print(f"  约束2 (单调性违背 <= 0): {'✓' if constraint2_ok else '✗'} ({mono_violation:.4f})")
        print(f"  条件1 (KS_test >= {ks_threshold_min}): {'✓' if condition1_ok else '✗'} ({ks_test:.4f})")
        print(f"  条件2 (KS gap <= {ks_gap_threshold}): {'✓' if condition2_ok else '✗'} ({ks_gap:.4f})")
        print(f"  条件3 (KS_train > KS_test): {'✓' if condition3_ok else '✗'} ({ks_train:.4f} vs {ks_test:.4f})")
        
        all_conditions_met = constraint1_ok and constraint2_ok and condition1_ok and condition2_ok and condition3_ok
        
        # 根据约束条件满足情况更新状态
        if all_conditions_met:
            print(f"\n✓ 找到满足所有约束和条件的最优参数")
            is_optuna = "optuna_found"
        else:
            print(f"\n⚠ 部分约束或条件未满足，但已选择最优可用解")
            # 如果之前标记为found，但现在条件不满足，改为partial
            if is_optuna == "optuna_found":
                is_optuna = "optuna_partial"
            # 如果满足部分关键条件，可以标记为partial
            elif condition1_ok and condition2_ok:
                is_optuna = "optuna_partial"
        
        # 输出Pareto front统计信息
        if len(pareto_trials) > 1:
            print("\n" + "-" * 60)
            print("Pareto Front 统计信息")
            print("-" * 60)
            ks_test_values = [t.user_attrs.get("ks_test", 0) for t in pareto_trials]
            lift_head_values = [t.user_attrs.get("lift_head10", 0) for t in pareto_trials]
            lift_tail_values = [t.user_attrs.get("lift_tail10", 0) for t in pareto_trials]
            print(f"  KS_test 范围: [{min(ks_test_values):.4f}, {max(ks_test_values):.4f}]")
            print(f"  Lift_head10 范围: [{min(lift_head_values):.4f}, {max(lift_head_values):.4f}]")
            print(f"  Lift_tail10 范围: [{min(lift_tail_values):.4f}, {max(lift_tail_values):.4f}]")
        
        # 输出最终选择的模型参数（关键参数）
        print("\n" + "-" * 60)
        print("最终选择的模型参数")
        print("best_params:",best_params)
        
        print("\n" + "=" * 60)
        print(f"模型optuna多目标优化完成! (状态: {is_optuna})")
        print("=" * 60)       
    else:
        print("=" * 60)
        print("跳过超参数寻优，使用默认参数")
        print(f"默认参数为: {config_model_params}")
        best_params = config_model_params  # 返回默认参数
        is_optuna = "default"
        print("=" * 60)

    return best_params, is_optuna

### Step 4: 保存 PMML 模型
def save_model_pmml(model_lgb, X_train, y_train, X_test, y_test, use_fea, model_save_path):
    """
    训练 LightGBM 模型、评估指标、保存PMML模型。
    
    参数：
    - X_train, y_train: 训练集特征和标签
    - X_test, y_test: 测试集特征和标签
    - use_fea: 使用的特征列表
    - model_save_path: PMML模型保存路径
    
    返回：
    - model_metrics: 包含训练集和测试集评估指标的字典
    """
    X_train = X_train[use_fea]
    X_test = X_test[use_fea] 
    mapper = DataFrameMapper([(f, None) for f in use_fea])
    pipeline = PMMLPipeline([("mapper", mapper), ("classifier", deepcopy(model_lgb))])
    pipeline.fit(X_train, y_train)

    # 预测 + 计算指标
    pmml_pred_train = pipeline.predict_proba(X_train)[:, 1]
    pmml_auc_train = roc_auc_score(y_train, pmml_pred_train)
    pmml_ks_train = ks_score(y_train, pmml_pred_train)

    pmml_pred_test = pipeline.predict_proba(X_test)[:, 1]
    pmml_auc_test = roc_auc_score(y_test, pmml_pred_test)
    pmml_ks_test = ks_score(y_test, pmml_pred_test)
    
    # 保存模型评估指标
    model_metrics = {
        'ks_train': pmml_ks_train,
        'ks_test': pmml_ks_test,
        'auc_train': pmml_auc_train,
        'auc_test': pmml_auc_test
    }
    
    # 尝试保存PMML模型
    try:
        sklearn2pmml(pipeline, model_save_path, with_repr=True)
        print("   PMML模型训练完成并保存!")
    except Exception as e:
        print("PMML模型无法保存，缺少java环境！！！-模型预测结果：\n")
    
    print("   PMML模型性能指标:")
    print("   ┌─────────────────────────────────────┐")
    print(f"   │ 训练集: AUC={pmml_auc_train:.6f}, KS={pmml_ks_train:.6f} │")
    print(f"   │ 测试集: AUC={pmml_auc_test:.6f}, KS={pmml_ks_test:.6f} │") 
    print("   └─────────────────────────────────────┘")
    return model_metrics


def save_model_pkl(model_lgb, X_train, y_train, X_test, y_test, use_fea, model_save_path):
    """
    训练 LightGBM 模型、评估指标、保存模型（如果KS一致）。
    
    参数：
    - train, y_train: 训练集特征和标签
    - test, y_test: 测试集特征和标签
    - best_value: optuna 训练时的最佳 ks_test 值
    - best_params: optuna 输出的最优参数（包含 n_estimators）
    - model_name: 要保存的模型文件名，默认是 modelname_ex.pkl
    """
    X_train = X_train[use_fea]
    X_test = X_test[use_fea]
    model_lgb = deepcopy(model_lgb) 
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric=ks_metric)
    
    # 预测 + 计算指标
    lgb_pred_train = model_lgb.predict_proba(X_train)[:, 1]
    lgb_auc_train = roc_auc_score(y_train, lgb_pred_train)
    lgb_ks_train = ks_score(y_train, lgb_pred_train)

    pmml_test = model_lgb.predict_proba(X_test)[:, 1]
    pmml_auc_test = roc_auc_score(y_test, pmml_test)
    pmml_ks_test = ks_score(y_test, pmml_test)
    
    # 保存模型评估指标
    model_metrics = {
        'ks_train': lgb_ks_train,
        'ks_test': pmml_ks_test,
        'auc_train': lgb_auc_train,
        'auc_test': pmml_auc_test
    }
    
    # 判断并保存模型
    pickle.dump(model_lgb, open(model_save_path, 'wb'))
    print("   PKL模型训练完成并保存!")
    print("   模型性能指标:")
    print("   ┌─────────────────────────────────────┐")
    print(f"   │ 训练集: AUC={lgb_auc_train:.6f}, KS={lgb_ks_train:.6f} │")
    print(f"   │ 测试集: AUC={pmml_auc_test:.6f}, KS={pmml_ks_test:.6f} │")
    print("   └─────────────────────────────────────┘")

    
    return model_metrics

def generate_model_train_json(model_metrics, best_params, feature_imp, output_path, is_optuna):
    """
    生成模型训练的JSON报告
    
    参数:
        model_metrics: dict, 包含模型评估指标
        best_params: dict, 模型最优参数
        feature_imp: DataFrame, 特征重要性数据
        output_path: str, 输出路径
        is_optuna: bool, 是否使用Optuna进行参数优化
    """
    try:
        # 1. 模型训练结果
        if is_optuna == "optuna_found":
            model_search_desc = "Optuna自动寻优(满足条件)"
        elif is_optuna == "optuna_not_found":
            model_search_desc = "Optuna自动寻优(未满足条件，取KS_test最高)"
        else:
            model_search_desc = "使用默认参数"
        model_result = {
            "模型训练结果": {
                "模型训练方式": "LightGBM",
                "训练集KS值": f"{model_metrics['ks_train']:.3f}",
                "测试集KS值": f"{model_metrics['ks_test']:.3f}",
                "训练集AUC值": f"{model_metrics['auc_train']:.3f}",
                "测试集AUC值": f"{model_metrics['auc_test']:.3f}",
                "模型寻优方式": model_search_desc,
                "最优参数": best_params
            }
        }
        
        # 2. 特征重要性分析
        # 计算特征数量统计
        total_features = len(feature_imp)
        non_zero_features = len(feature_imp[feature_imp['importance_score_split'] > 0])
        zero_features = total_features - non_zero_features
        
        # 获取TOP15特征
        top_features = feature_imp.head(15).copy()
        top_features['importance_ratio'] = top_features['importance_score_split'] / top_features['importance_score_split'].sum()
        top_features['importance_ratio'] = top_features['importance_ratio'].apply(lambda x: f"{x:.1%}")
        # 填充NaN为None，保证json合法
        top_features = top_features.where(~top_features.isna(), None)  
        feature_importance = {
            "特征数量统计": {
                "总特征数": total_features,
                "非零重要性特征数": non_zero_features,
                "零重要性特征数": zero_features
            },
            "特征重要性TOP15": []
        }
        
        # 添加TOP15特征详情
        for _, row in top_features.iterrows():
            feature_importance["特征重要性TOP15"].append({
                "特征名称": row['Feature'],
                "中文名称": row['feature_cn'],
                "数据源": row.get('data_source', '未知'),
                "分裂重要性": int(row['importance_score_split']),
                "增益重要性": f"{row['importance_score_gain']:.3f}",
                "重要性占比": row['importance_ratio']
            })
        
        # 整合所有结果
        report_data = {
            "report_title": "模型训练报告",
            "model_result": model_result,
            "feature_importance": feature_importance
        }
        
        # 转换为JSON
        json_str = json.dumps(report_data, ensure_ascii=False, indent=2)
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        return output_path
            
    except Exception as e:
        print(f"Error generating model training JSON report: {str(e)}")
        return None

def generate_autogluon_train_json(leaderboard, output_path, autogluon_supported):
    """
    生成AutoGluon多模型对比的JSON报告
    参数:
        leaderboard: DataFrame, AutoGluon输出的leaderboard
        output_path: str, 输出路径
        autogluon_supported: list, 对比的模型类型列表
    """
    try:
        # 保留所有float列4位小数
        float_cols = leaderboard.select_dtypes(include=['float', 'float64']).columns
        leaderboard[float_cols] = leaderboard[float_cols].round(4)
        # 只展示前15行
        if len(leaderboard) > 15:
            leaderboard_show = leaderboard.head(15)
        else:
            leaderboard_show = leaderboard
        report_data = {
            "模型训练方式": "AutoGluon多模型对比",
            "对比模型列表": autogluon_supported,
            "leaderboard": leaderboard_show.to_dict(orient='records')
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        return output_path
    except Exception as e:
        print(f"Error generating AutoGluon training JSON report: {str(e)}")
        return None
        
#### 特征重要性计算
def feature_importance_analysis(model_lgb, use_fea, feature_dict_path, imp_path):
    """
    计算并保存特征重要性（split + gain），并输出筛选后的有效特征列表。

    参数：
    - model_lgb: 训练好的 LightGBM 模型
    - use_fea: 使用的特征列表
    - feature_dict_path: 特征英文-中文映射表路径（CSV）
    - output_dir: 输出路径

    返回：
    - final_features: 最终保留的特征列表（按 importance_score_split > 0）
    """


     # 加载特征字典
    if feature_dict_path and os.path.exists(feature_dict_path):
        feature_map = pd.read_csv(feature_dict_path)
        if 'Feature' not in feature_map.columns:
            print(f"特征字典文件 {feature_dict_path} 缺少主键列 'Feature'，请检查表头命名。将使用空表。")
            feature_map = pd.DataFrame(columns=['Feature', 'feature_cn', 'data_source'])
    else:
        print(f"特征字典文件 {feature_dict_path} 不存在，将使用空表。")
        feature_map = pd.DataFrame(columns=['Feature', 'feature_cn', 'data_source'])
        
    # 获取特征重要性
    feature_imp_split = pd.DataFrame({
        'Feature': use_fea,
        'importance_score_split': model_lgb.booster_.feature_importance(importance_type='split')
    })
    feature_imp_gain = pd.DataFrame({
        'Feature': use_fea,
        'importance_score_gain': model_lgb.booster_.feature_importance(importance_type='gain')
    })
    feature_imp = pd.merge(feature_imp_split, feature_imp_gain, on='Feature', how='inner')

    # 合并中文名称
    feature_imp = pd.merge(feature_map, feature_imp, on='Feature', how='right')
    feature_imp.sort_values(by='importance_score_split', ascending=False, inplace=True)
    feature_imp['imp_split_ratio'] = feature_imp['importance_score_split'] / feature_imp['importance_score_split'].sum()
    feature_imp['imp_gain_ratio'] = feature_imp['importance_score_gain'] / feature_imp['importance_score_gain'].sum()

    # 保存特征重要性文件
    feature_imp.to_csv(imp_path, index=False)

    # 输出 importance_score_split > 0 的有效特征[不做筛选]
    final_feature_df = feature_imp[feature_imp['importance_score_split'] > 0][['Feature']]
    print("=" * 60)
    print(f"特征重要性大于0的特征数量: {len(final_feature_df)}")
    print("特征重要性大于0的特征：",list(final_feature_df['Feature']) )
    return feature_imp

def autogluon_model_comparison(
    train_df,
    test_df,
    target_col,
    include_model_types=None,  # 只用 config['settings']['model_type_list'] 传递的模型类型
    time_limit=300,
    eval_metric=None,
    output_path=None,
    predictor_path=None,
    log_path=None,
    save_predictor=True  # 新增参数，是否保存模型文件
):
    """
    用AutoGluon对比多模型效果，输出leaderboard。
    关于 include_model_types：
      - 支持的 AutoGluon 模型类型及含义：
          "CAT"      : CatBoost（高效的梯度提升决策树）
          "GBM"      : LightGBM（微软的高效梯度提升树）
          "XGB"      : XGBoost（流行的梯度提升树）
          "NN_TORCH" : 神经网络 (PyTorch 实现)
          "RF"       : 随机森林 (Random Forest)
          "XT"       : Extra Trees（极端随机树）
          "KNN"      : K-Nearest Neighbors（K近邻）
          "FASTAI"   : FastAI 神经网络

    参数：
        train_df: 训练集DataFrame
        test_df: 验证集DataFrame
        target_col: 目标列名
        include_model_types: 要对比的模型类型list（如None/[]/空则用AutoGluon默认）
        time_limit: 训练时间限制（秒）
        eval_metric: 评估指标
            - None: 使用AutoGluon默认指标
            - 'ks'/'KS': 使用自定义KS指标
            - 其他字符串: 使用AutoGluon内置指标（如'roc_auc', 'f1', 'precision'等）
        output_path: leaderboard保存路径（如有）
        predictor_path: 模型保存路径（如有）
        log_path: 日志文件路径（如有）
        save_predictor: bool, 是否保存模型文件（默认True）。为False时训练后会删除predictor_path目录。
    返回：
        leaderboard DataFrame
    """
    import warnings
    warnings.filterwarnings('ignore')
    from autogluon.tabular import TabularPredictor
    from autogluon.core.metrics import make_scorer
    
    # eval_metric处理逻辑
    if eval_metric is None:
        # 使用AutoGluon默认指标
        metric = None
        print("使用AutoGluon默认评估指标")
    elif isinstance(eval_metric, str) and eval_metric.lower() in ['ks', 'k-s']:
        # 使用自定义KS指标
        ks_scorer = make_scorer(
            name='ks',
            score_func=ks_score,  # 使用模块级别的函数
            optimum=1,
            greater_is_better=True,
            needs_proba=True
        )
        metric = ks_scorer
        print("使用自定义KS评估指标")
    else:
        # 使用AutoGluon内置指标
        metric = eval_metric
        print(f"使用AutoGluon内置评估指标: {eval_metric}")

    train_data = train_df.copy()
    test_data = test_df.copy()
    
    # 构建TabularPredictor参数
    predictor_kwargs = {
        'label': target_col,
        'eval_metric': metric,
        'problem_type': 'binary',
        'verbosity': 2  # 标准日志等级
    }
    
    # 添加可选参数
    if predictor_path:
        predictor_kwargs['path'] = predictor_path
    if log_path:
        predictor_kwargs['log_to_file'] = True
        predictor_kwargs['log_file_path'] = log_path
    
    predictor = TabularPredictor(**predictor_kwargs)

    fit_kwargs = dict(
        train_data=train_data,
        tuning_data=test_data,
        presets='best_quality',
        time_limit=time_limit,
        auto_stack=False,
        included_model_types=include_model_types
    )

    predictor.fit(**fit_kwargs)

    leaderboard = predictor.leaderboard(silent=True)
    if output_path:
        leaderboard.to_csv(output_path, index=False)
    print(leaderboard)
     # 新增：如不保存模型，删除目录
    if predictor_path and not save_predictor:
        try:
            shutil.rmtree(predictor_path)
            print("不保存多模型对比文件")
        except Exception as e:
            print(f"Warning: 删除AutoGluon模型目录失败: {predictor_path}, {e}")
    return leaderboard

##模块6：模型训练
# output_dst/
# ├── model.pkl                  # 训练好的模型文件（pkl格式）
# ├── model.pmml                 # PMML格式模型（若有需求）
# ├── use_features.json         # 入模特征 list
# ├── best_params.json          # 最优参数
# ├── model_fea/                # 特征重要性分析输出目录（已有）
####---------------------主程序---------------------------------    
def model_train_func(dev_data, oot_data, config):
    try:
        ##读取参数
        feature_dic_path = config['data_path']['feature_dic_path']
        with open(config['data_path']['use_fea_path'],"r", encoding="utf-8") as f:
            use_fea = json.load(f)
        label_col = config['settings']['label']
        id_col = config['settings']['id_col']
        dst_model = config['output_path']['dst_model']
        dst_outs = config['output_path']['dst_outs']
        os.makedirs(dst_model, exist_ok=True)
        os.makedirs(dst_outs, exist_ok=True)
        pkl_model_path = config['output_path']['pkl_model_path']
        pmml_model_path = config['output_path']['pmml_model_path']
        feature_importance_path = config['output_path']['feature_importance_path']
        model_params_path = config['output_path']['model_params_path']
        model_fea_path = config['output_path']['model_fea_path']
        model_json_path = config['output_path']['model_json_path']

        # ====== 执行模式控制（只用 model_select 控制） ======
        model_select = config['settings'].get('model_select', 'LightGBM').lower()

        print("=" * 60)
        print(f"  - 当前模型选择: {model_select}")
        print("=" * 60)

        if model_select == 'lightgbm':
            # ====== LightGBM模型训练 ======
            print("\n>>> LightGBM模型训练开始 ...")
            X_train = dev_data[use_fea]
            y_train = dev_data[label_col]
            X_test = oot_data[use_fea]
            y_test = oot_data[label_col]
            
            ## 模型寻优：【参数配置：optuna寻优|使用默认参数】
            dic_p = config['settings']
            best_params, is_optuna = train_lgb_with_optuna(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, dic_p=dic_p)
            # 保存模型最优参数
            with open(model_params_path, 'w', encoding='utf8') as f:
                json.dump(best_params, f, indent=4, ensure_ascii=False)
                
            ## 使用调参构造模型
            model_lgb = lgb.LGBMClassifier(**best_params)
            print(f"入模特征数量: {len(use_fea)}")
            print("入模特征为：", use_fea)
            
            # === Step 5.1. 评估模型 + 导出PKL ===
            model_metrics = save_model_pkl(model_lgb, X_train, y_train, X_test, y_test, use_fea, pkl_model_path)
            
            # === Step 5.2. 评估模型 + 导出PMML ===
            pmml_metrics = save_model_pmml(model_lgb, X_train, y_train, X_test, y_test, use_fea, pmml_model_path)
    
            ## 特征重要性评估筛选
            model_lgb = deepcopy(model_lgb)
            model_lgb.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric=ks_metric)
            feature_imp = feature_importance_analysis(model_lgb, use_fea, feature_dic_path, feature_importance_path)
            
            # 保存最终的入模特征列表
            with open(model_fea_path, 'w', encoding='utf8') as f:
                json.dump(use_fea, f, indent=4, ensure_ascii=False)
                
            # 生成模型训练JSON报告
            generate_model_train_json(
                model_metrics=model_metrics,
                best_params=best_params,
                feature_imp=feature_imp,
                output_path=model_json_path,
                is_optuna=is_optuna
            )
            print(">>> LightGBM模型训练完成！\n")
        elif model_select == 'autogluon':
            # ====== AutoGluon多模型对比 ======
            print("\n>>> AutoGluon多模型对比开始 ...")
            autogluon_supported = ['CAT', 'GBM','XGB','NN_TORCH','RF','XT','KNN','FASTAI']
            save_predictor = config['settings'].get('save_autogluon_predictor', False)# 是否保存模型由config控制，默认False
            time_limit = config['settings'].get('autogluon_time_limit', 60*5)
            output_path = os.path.join(dst_outs, 'autogluon_leaderboard.csv')
            predictor_path = os.path.join(dst_model, 'autogluon_predictor')
            os.makedirs(predictor_path, exist_ok=True)
            autogluon_log_path = os.path.join(dst_outs, 'autogluon_log.txt')
            leaderboard = autogluon_model_comparison(
                train_df=dev_data[use_fea+[label_col]],
                test_df=oot_data[use_fea+[label_col]],
                target_col=label_col,
                include_model_types=autogluon_supported,
                time_limit=time_limit,
                eval_metric='Ks',
                output_path=output_path,
                predictor_path=predictor_path,
                log_path=autogluon_log_path,
                save_predictor=save_predictor
            )
             # 保存为json用于页面展示，包含对比模型列表
            generate_autogluon_train_json(leaderboard, model_json_path, autogluon_supported)
            print(">>> AutoGluon多模型对比完成！\n")
        else:
            print(f"错误: model_select 参数只能为 'LightGBM' 或 'AutoGluon'，当前为: {model_select}")
            return False

        print("=" * 60)
        print("所有任务执行完成!")
        print("=" * 60)
        return True   
    except Exception as e:
        msg = traceback.format_exc()
        print(msg)
        return False
