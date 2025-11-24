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
import pandas as pd
# optuna.logging.set_verbosity(optuna.logging.WARNING) ## 只打印警告和异常，不打印中间结果
### 使用LR模型训练
from sklearn.linear_model import LogisticRegression
import shutil
def decile_analysis(y_true, y_pred, n_bins=10):
    # 排序后bins
    df_tmp = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).sort_values('y_pred', ascending=False)
    bins_df = sta_groups(df_tmp['y_true'], df_tmp['y_pred'], bins_num=n_bins, labels=list(range(n_bins, 0, -1)))
    # 计算head/tail lift
    base_bad_rate = df_tmp['y_true'].mean()
    bad_rates = []
    for idx in range(len(bins_df)):
        bad_rates.append(bins_df.iloc[idx]['bad'] / bins_df.iloc[idx]['num'] if bins_df.iloc[idx]['num'] > 0 else np.nan)
    lift_head = bad_rates[0] / base_bad_rate if base_bad_rate else np.nan
    lift_tail = bad_rates[-1] / base_bad_rate if base_bad_rate else np.nan
    return bins_df, bad_rates, lift_head, lift_tail, base_bad_rate


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

def compute_psi(train_scores, test_scores):
    psi_calculator = Psi()
    psi_calculator.fit(train_scores)
    return float(psi_calculator.transform(test_scores)[0])

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
    optuna_trials_csv_path = dic_p.get('optuna_trials_csv_path')
    # 1. 如果RUN_HPO为True，则进行模型超参数寻优
    if  IF_RUN:
        print("=" * 60)
        print("开始模型超参数寻优 (Optuna)")
        n_trials=dic_p["n_trials"]
        ks_threshold_min = dic_p['ks_threshold_min']
        ks_gap_threshold = dic_p['ks_gap_threshold']
        psi_threshold_max = 0.2 
        print(f"   - 寻优次数: {n_trials}")
        print(f"   - KS阈值范围: [{ks_threshold_min:.3f},1.0]")
        print(f"   - KS差值阈值: {ks_gap_threshold:.3f}")
        print(f"   - PSI最大阈值: {psi_threshold_max:.3f}")
        print(f"   - 训练集样本: {len(X_train)}")
        print(f"   - 测试集样本: {len(X_test)}")

        # 2. 定义约束函数
        def constraints(trial: optuna.Trial):
            ks_gap = trial.user_attrs["ks_gap"]
            mono_violation = trial.user_attrs["mono_violation"]          
            # 约束1：KS gap <= ks_gap_threshold
            g1 = ks_gap - ks_gap_threshold         
            # 约束2：单调性违背程度 <= 0（完全单调则 mono_violation=0，否则>0）
            g2 = mono_violation  # 已经是 >=0 的量，要求 <=0           
            return (g1, g2)

        # 3. 定义目标函数
        def objective(trial):
            ## 获取当前试验的参数
            search_space = dic_p['model_selection_params']
            param_grid = {
                ####  补充参数：min_split_gain，可寻优不可修改，大模型不识别
                "min_split_gain": trial.suggest_int("min_split_gain", 0,10,step=1), # 最小分割增益
                #### 大模型可调参数如下
                "learning_rate": trial.suggest_float("learning_rate", *search_space["learning_rate"]), # 学习率
                "num_leaves": trial.suggest_int("num_leaves", *search_space["num_leaves"]), # 叶子节点数
                "max_bin": trial.suggest_int("max_bin", *search_space["max_bin"]), # 最大bin数
                "reg_alpha": trial.suggest_int("reg_alpha", *search_space["reg_alpha"]), # 正则化系数
                "reg_lambda": trial.suggest_int("reg_lambda", *search_space["reg_lambda"]), # 正则化系数
                "subsample": trial.suggest_float("subsample", *search_space["subsample"]), # 子采样比例
                "colsample_bytree": trial.suggest_float("colsample_bytree", *search_space["colsample_bytree"]), # 列采样比例
                "subsample_freq": trial.suggest_int("subsample_freq", *search_space["subsample_freq"]), # 子采样频率
                "min_child_weight": trial.suggest_int("min_child_weight", *search_space["min_child_weight"]), # 最小子权重
                "n_estimators": trial.suggest_categorical("n_estimators", search_space["n_estimators"]), # 迭代次数
                "random_state": 42, # 随机种子
                "max_depth": trial.suggest_int("max_depth", *search_space["max_depth"]), # 最大深度
                "scale_pos_weight": 1.0, # 正样本权重
            }
           ## 训练模型
            print(f"============== Trial #{trial.number} ==============")
            model_params = config_model_params.copy()
            model_params.update(param_grid)
            model = lgb.LGBMClassifier(**model_params)
            # 使用ks早停和ks评估，来增加ks的优化权重
            eval_result = {}
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
                    eval_metric=ks_metric, callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0),lgb.record_evaluation(eval_result)])          
            
            ## 通过日志寻找最佳的ks所在的真实迭代轮次
            best_score = model.best_score_['valid_1']['ks']   #最佳迭代的ks
            # best_iteration = model.best_iteration_
            # print(f"模型记录的的轮次: {best_iteration}")
            # print(f"最佳迭代轮次的KS: {best_score}")
            
            # 通过日志寻找最佳的ks所在的真实迭代轮次（全量）
            # 获取所有指标列表
            train_loss_list = eval_result['training']['binary_logloss']   # 训练集的 loss
            train_ks_list = eval_result['training']['ks']   # 训练集的 ks
            valid_loss_list = eval_result['valid_1']['binary_logloss']   # 验证集的 loss
            valid_ks_list = eval_result['valid_1']['ks']   # 验证集的 ks

            # 去掉连续重复，按照 training's binary_logloss 进行去重
            # 保留"真正变化的一步一步的历史
            kept = []
            removed = []
            prev_train_loss = None
            new_iter = 0         # 去重后的新迭代轮次，从 1 开始计

            for i in range(len(train_loss_list)):
                log_iter = i + 1   # 日志里轮次从 1 开始
                train_loss = train_loss_list[i]
                train_ks = train_ks_list[i]
                valid_loss = valid_loss_list[i]
                valid_ks = valid_ks_list[i]
                
                # first round or different training loss
                if prev_train_loss is None or train_loss != prev_train_loss:
                    new_iter += 1
                    kept.append((log_iter, new_iter, train_loss, train_ks, valid_loss, valid_ks))
                else:
                    # 重复（training loss 相同）
                    removed.append((log_iter, train_loss, train_ks, valid_loss, valid_ks))
                    print(f"重复删除: {log_iter}, train_loss={train_loss}, train_ks={train_ks}, valid_loss={valid_loss}, valid_ks={valid_ks}")
                # 更新 prev
                prev_train_loss = train_loss
                # 到最佳 ks 就停止
                if valid_ks == best_score:
                    break

            best_iteration = kept[-1][1]
            # print(f"去重后的真实轮次: {best_iteration}") 
            
            # 使用 booster 的 predict 方法，指定最佳轮次
            pred_train = model.booster_.predict(X_train, num_iteration=best_iteration)
            pred_test = model.booster_.predict(X_test, num_iteration=best_iteration)
            ## 计算各种指标，包括KS指标、head/tail 10% lift + decile 坏人占比 以及单调性违背程度 
            ks_train = ks_score(y_train, pred_train)
            ks_test = ks_score(y_test, pred_test) 
            auc_train = auc_score(y_train, pred_train)
            auc_test = auc_score(y_test, pred_test)
            ks_gap = abs(ks_train - ks_test)
            df_lift, bad_rates, lift_head10, lift_tail10 , overall_bad_rate= decile_analysis(y_test, pred_test, n_bins=10)

            mono_violation = monotonicity_violation(bad_rates)
            psi_value = compute_psi(pred_train, pred_test)

            # ⑤ 打印最关键的调试信息
            # print(f"模型(去重)预测的KS: {ks_test}")
            if best_score == ks_test:
                print(f"KS是否一致？一致")
            else:
                # 红色高亮警告
                print(f"重大警告：去重后的 KS（{ks_test2}）与 best_score（{best_score}）仍旧不一致！")

            ## 设置trial的属性-⑥ 把所有中间结果存到 user_attrs，后面 constraints_func 和 DataFrame 都会用到
            # 将 param_grid 中的 n_estimators 替换为实际的 best_iteration
            param_grid_final = param_grid.copy()
            param_grid_final["n_estimators"] = model.best_iteration_
            trial.set_user_attr("ks_train", ks_train)
            trial.set_user_attr("ks_test", ks_test)
            trial.set_user_attr("auc_train", auc_train)
            trial.set_user_attr("auc_test", auc_test)
            trial.set_user_attr("params", param_grid_final)
            trial.set_user_attr("ks_gap", ks_gap)
            trial.set_user_attr("lift_head10", lift_head10)
            trial.set_user_attr("lift_tail10", lift_tail10)
            trial.set_user_attr("overall_bad_rate", overall_bad_rate)
            trial.set_user_attr("bad_rates", bad_rates)
            trial.set_user_attr("mono_violation", mono_violation)
            trial.set_user_attr("psi", psi_value)

            # # 优先满足条件的trial（单目标优化）
            # if (ks_test >= ks_threshold_min and abs(ks_train - ks_test) <= ks_gap_threshold and ks_train > ks_test):
            #     return ks_test
            # else:
            #     return -1e6 + ks_test  # 不满足条件的trial，值很小但还能区分ks_test大小

            # 多目标返回：（多目标优化）
            #   目标1：ks_test（越大越好）
            #   目标2：head 10% lift（越大越好）
            #   目标3：tail 10% lift（越小越好）
            return ks_test, lift_head10, lift_tail10

        # 4.总结trial
        def summarize_trial(trial):
            summary = {
                "ks_train": trial.user_attrs.get("ks_train", 0.0),
                "ks_test": trial.user_attrs.get("ks_test", 0.0),
                "ks_gap": trial.user_attrs.get("ks_gap", 0.0),
                "auc_train": trial.user_attrs.get("auc_train", 0.0),
                "auc_test": trial.user_attrs.get("auc_test", 0.0),
                "psi": trial.user_attrs.get("psi", None),
                "lift_head10": trial.user_attrs.get("lift_head10", 0.0),
                "lift_tail10": trial.user_attrs.get("lift_tail10", 0.0),
                "mono_violation": trial.user_attrs.get("mono_violation", 0.0),
                "overall_bad_rate": trial.user_attrs.get("overall_bad_rate", 0.0),
                "params": trial.user_attrs.get("params", {}).copy(),
            }
            ## 检查约束条件（硬约束2个）
            summary["constraint_gap_ok"] = summary["ks_gap"] <= ks_gap_threshold # KS gap <= KS_gap_threshold
            summary["constraint_mono_ok"] = summary["mono_violation"] <= 0 # 单调性违背程度 <= 0
            summary["constraints_ok"] = summary["constraint_gap_ok"] and summary["constraint_mono_ok"]
            ## 检查业务条件（3个可选）
            summary["business_ks_threshold_ok"] = summary["ks_test"] >= ks_threshold_min # KS_test >= KS_threshold_min
            summary["business_order_ok"] = summary["ks_train"] > summary["ks_test"] # KS_train > KS_test
            summary["business_psi_ok"] = summary["psi"] <= psi_threshold_max # psi < psi_threshold_max
            summary["business_ok"] = summary["business_ks_threshold_ok"] and summary["business_order_ok"] and summary["business_psi_ok"]
            ## 检查是否满足全部约束和业务条件
            summary["all_conditions_ok"] = summary["constraints_ok"] and summary["business_ok"] 
            return summary
        
        # 5. 收集trial总结
        def collect_trial_summaries(trials, pareto_numbers=None):
            pareto_numbers = pareto_numbers or set()
            summaries = []
            for trial in trials:
                summary = summarize_trial(trial)
                summary["is_pareto"] = int(trial.number in pareto_numbers)
                summaries.append((trial, summary))
            return summaries

        def export_trial_summaries_to_csv(trial_summaries, csv_path):
            if not csv_path or len(trial_summaries) == 0:
                return
            records = []
            for trial, summary in trial_summaries:
                record = {
                    "trial_number": trial.number,
                    "state": getattr(trial.state, "name", str(trial.state)),
                    "objective_values": json.dumps(list(trial.values) if trial.values is not None else None, ensure_ascii=False),
                    "ks_train": summary["ks_train"],
                    "ks_test": summary["ks_test"],
                    "ks_gap": summary["ks_gap"],
                    "auc_train": summary["auc_train"],
                    "auc_test": summary["auc_test"],
                    "psi": summary["psi"],
                    "lift_head10": summary["lift_head10"],
                    "lift_tail10": summary["lift_tail10"],
                    "mono_violation": summary["mono_violation"],
                    "overall_bad_rate": summary["overall_bad_rate"],
                    "params_json": json.dumps(summary["params"], ensure_ascii=False),
                    "constraint_gap_ok": int(summary["constraint_gap_ok"]),
                    "constraint_mono_ok": int(summary["constraint_mono_ok"]),
                    "business_psi_ok": int(summary["business_psi_ok"]),
                    "business_ks_threshold_ok": int(summary["business_ks_threshold_ok"]),
                    "business_order_ok": int(summary["business_order_ok"]),
                    "all_conditions_ok": int(summary["all_conditions_ok"]),
                    "is_pareto_front": summary.get("is_pareto", 0),
                }
                records.append(record)
            if not records:
                return
            output_dir = os.path.dirname(csv_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            df = pd.DataFrame(records)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"已保存 {len(records)} 条Optuna可行解到 {csv_path}")
        

        # # 单目标
        # objective_with_params = partial(objective)
        # study = optuna.create_study(direction="maximize")
        # study.optimize(objective_with_params, n_trials=n_trials)

        # 6. 使用NSGA-II多目标优化
        sampler = optuna.samplers.NSGAIISampler(constraints_func=constraints) 
        study = optuna.create_study(
            directions=["maximize", "maximize", "minimize"],
            sampler=sampler
        )
        study.optimize(objective, n_trials=n_trials)  # n_jobs 可并行
        
        ## 多目标优化：从Pareto front中选择最佳解
        print("\n" + "=" * 60)
        print("多目标优化结果分析")
        print("=" * 60)
        
        # 获取Pareto front中的解（多目标优化返回的是best_trials列表）
        pareto_trials = study.best_trials
        pareto_trial_numbers = {trial.number for trial in pareto_trials}        
        pareto_summaries = collect_trial_summaries(pareto_trials, pareto_trial_numbers)
        print(f"Pareto front 解数量: {len(pareto_summaries)}")
        # 导出所有完成的trial总结到csv文件
        all_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        all_summaries = collect_trial_summaries(all_trials, pareto_trial_numbers)
        export_trial_summaries_to_csv(all_summaries, optuna_trials_csv_path)
        
        # 7. 选择最佳解
        # 如果找到满足全部约束的解，则标记为optuna_found
        # 如果找到满足部分约束的解，则标记为optuna_partial
        # 如果未找到满足约束的解，则标记为optuna_not_found

        ## 先确定候选池：优先使用 Pareto front，否则退回所有完成的 trial
        if len(pareto_summaries) > 0:
            trial_selection_pool = pareto_summaries
        else:
            print("警告：未找到满足全部约束的Pareto front解，使用所有trials中KS_test最高的解")
            if len(all_summaries) == 0:
                raise ValueError("没有完成的trials，无法选择满足全部约束的最佳解")
            trial_selection_pool = all_summaries

        ## 1）优先选：全部约束 + 业务条件都满足的解
        fully_valid = [(trial, summary) for trial, summary in trial_selection_pool if summary["all_conditions_ok"]]
        ## 2）其次选：至少满足“硬约束”（constraints_ok），但没满足全部业务条件的解
        partially_valid = [(trial, summary) for trial, summary in trial_selection_pool if (not summary["all_conditions_ok"]) and summary["constraints_ok"]]
        
        if fully_valid:
            # 情况1：有完全满足全部约束+业务条件的解，标记为optuna_found
            best_trial, best_summary = max(fully_valid, key=lambda ts: ts[1]["ks_test"])
            print(f"从 {len(fully_valid)} 个满足全部约束的解中选择最佳解（KS_test最高）")
            is_optuna = "optuna_found"
        elif partially_valid:
            # 情况2：没有完全满足的解，但有部分满足约束的解，标记为optuna_partial
            best_trial, best_summary = max(partially_valid, key=lambda ts: ts[1]["ks_test"])
            print(f"未找到满足全部约束的解，从 {len(partially_valid)} 个部分满足约束的解中选择KS_test最高的解")
            is_optuna = "optuna_partial"
        else:
            # 情况3：连部分满足约束的解都没有，只能退化为纯KS最大，标记为optuna_not_found
            best_trial, best_summary = max(trial_selection_pool, key=lambda ts: ts[1]["ks_test"])
            print("未找到满足约束的解，退化为选择候选池中KS_test最高的解，所有的trail都已记录，可自行挑选")
            is_optuna = "optuna_not_found"
        
        # 8. 提取最佳解的参数和指标
        best_params = best_summary["params"].copy()
        best_params = {**config_model_params, **best_params}
        
        # 获取所有指标
        ks_train = best_summary["ks_train"]
        ks_test = best_summary["ks_test"]
        ks_gap = best_summary["ks_gap"]
        psi = best_summary['psi']
        lift_head10 = best_summary["lift_head10"]
        lift_tail10 = best_summary["lift_tail10"]
        mono_violation = best_summary["mono_violation"]
        overall_bad_rate = best_summary["overall_bad_rate"]
        
        # 9. 输出详细结果
        print(f"模型optuna多目标优化完成! (状态: {is_optuna})")
        print("\n"+"-" * 60)       
        print("选择的最终模型指标和参数")
        print(f"  KS训练集 / AUC训练集: {ks_train:.6f} / {best_summary['auc_train']:.6f}")
        print(f"  KS测试集 / AUC测试集: {ks_test:.6f} / {best_summary['auc_test']:.6f}")
        print(f"  PSI (训练 vs 测试): {psi:.6f}")
        print(f"  Head 10% / Tail 10% Lift: {lift_head10:.4f} / {lift_tail10:.4f}")     
        print("最终选择的模型参数",best_params)
        
        # 检查约束条件
        print("\n"+"-" * 60)
        print("约束条件检查：")
        print(f"  硬约束1 (KS gap <= {ks_gap_threshold}): {'[OK]' if best_summary['constraint_gap_ok'] else '[FAIL]'} ({ks_gap:.6f})")
        print(f"  硬约束2 (单调性违背 <= 0): {'[OK]' if best_summary['constraint_mono_ok'] else '[FAIL]'} ({mono_violation:.4f})")
        print(f"  业务条件1 (KS_test >= {ks_threshold_min}): {'[OK]' if best_summary['business_ks_threshold_ok'] else '[FAIL]'} ({ks_test:.6f})")
        print(f"  业务条件2 (KS_train > KS_test): {'[OK]' if best_summary['business_order_ok'] else '[FAIL]'} ({ks_train:.6f} vs {ks_test:.6f})")
        print(f"  业务条件3 (PSI <= {psi_threshold_max}): {'[OK]' if best_summary['business_psi_ok'] else '[FAIL]'} ({psi})")
        print("-" * 60)
    else:
        print("=" * 60)
        print("跳过超参数寻优，使用默认参数")  
        best_params = config_model_params  # 返回默认参数
        print(f"默认参数为: {best_params}")
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
    pmml_auc_train = auc_score(y_train, pmml_pred_train)
    pmml_ks_train = ks_score(y_train, pmml_pred_train)

    pmml_pred_test = pipeline.predict_proba(X_test)[:, 1]
    pmml_auc_test = auc_score(y_test, pmml_pred_test)
    pmml_ks_test = ks_score(y_test, pmml_pred_test)
            
    # 计算PSI
    psi_value = compute_psi(pmml_pred_train, pmml_pred_test)      
    # 计算KS差值
    ks_gap = abs(pmml_ks_train- pmml_ks_test)
     # 计算lift
    df_lift, bad_rates, lift_head10, lift_tail10, overall_bad_rate= decile_analysis(y_test, pmml_pred_test, n_bins=10)
    # 计算分箱单调性
    mono_violation = monotonicity_violation(bad_rates)
   
            
    # 保存模型评估指标
    model_metrics = {
        'ks_train': pmml_ks_train,
        'ks_test': pmml_ks_test,
        'auc_train': pmml_auc_train,
        'auc_test': pmml_auc_test,
        'psi_value':psi_value,
        'ks_gap':ks_gap,
        'lift_head10': lift_head10,
        'lift_tail10': lift_tail10,
        'mono_violation':mono_violation,
        'overall_bad_rate':overall_bad_rate,
        'df_lift': df_lift.to_dict(orient="records"), # DataFrame 转成字典存入
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
    print(f"   模型稳定性: PSI = {psi_value:.6f}")
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
    lgb_auc_train = auc_score(y_train, lgb_pred_train)
    lgb_ks_train = ks_score(y_train, lgb_pred_train)

    pmml_test = model_lgb.predict_proba(X_test)[:, 1]
    pmml_auc_test = auc_score(y_test, pmml_test)
    pmml_ks_test = ks_score(y_test, pmml_test)

     # 计算PSI
    psi_value = compute_psi(lgb_pred_train, pmml_test)      
    # 计算KS差值
    ks_gap = abs(lgb_ks_train- pmml_ks_test)
    # 计算lift
    df_lift, bad_rates, lift_head10, lift_tail10, overall_bad_rate= decile_analysis(y_test, pmml_test, n_bins=10)
    # 计算分箱单调性
    mono_violation = monotonicity_violation(bad_rates)

    preds_arr = np.array(pmml_test)
    mins, maxs, meanp = preds_arr.min(), preds_arr.max(), preds_arr.mean()
    # 保存模型评估指标
    model_metrics = {
        'ks_train': lgb_ks_train,
        'ks_test': pmml_ks_test,
        'auc_train': lgb_auc_train,
        'auc_test': pmml_auc_test,
        'psi': psi_value,
        'ks_gap':ks_gap,
        'lift_head10': lift_head10,
        'lift_tail10': lift_tail10,
        'mono_violation':mono_violation,
        'overall_bad_rate':overall_bad_rate,
        'df_lift': df_lift.to_dict(orient="records"),  # DataFrame 放进字典
    }
    
    # 判断并保存模型
    pickle.dump(model_lgb, open(model_save_path, 'wb'))
    print("   PKL模型训练完成并保存!")
    print("   模型性能指标:")
    print("   ┌─────────────────────────────────────┐")
    print(f"   │ 训练集: AUC={lgb_auc_train:.6f}, KS={lgb_ks_train:.6f} │")
    print(f"   │ 测试集: AUC={pmml_auc_test:.6f}, KS={pmml_ks_test:.6f} │")
    print("   └─────────────────────────────────────┘")
    print(f"   模型稳定性: PSI = {psi_value:.6f}")

    
    return model_metrics

def generate_model_train_json(model_metrics, best_params, feature_imp, output_path, is_optuna,use_fea):
    """
    生成模型训练的JSON报告
    
    参数:
        model_metrics: dict, 包含模型评估指标
        best_params: dict, 模型最优参数
        feature_imp: DataFrame, 特征重要性数据
        output_path: str, 输出路径
        is_optuna: str, Optuna优化状态
        use_fea: list, 入模特征列表
    """
    try:
        # 1. 模型训练结果
        if is_optuna == "optuna_found":
            model_search_desc = "Optuna自动寻优(满足条件)"
        elif is_optuna == "optuna_partial":
            model_search_desc = "Optuna自动寻优(部分满足条件)"
        elif is_optuna == "optuna_not_found":
            model_search_desc = "Optuna自动寻优(未满足条件，取KS_test最高)"
        else:
            model_search_desc = "使用默认参数"
        model_result = {
            "模型训练结果": {
                "模型训练方式": "LightGBM",
                "训练集KS值": f"{model_metrics['ks_train']:.4f}",
                "测试集KS值": f"{model_metrics['ks_test']:.4f}",
                "训练集AUC值": f"{model_metrics['auc_train']:.4f}",
                "测试集AUC值": f"{model_metrics['auc_test']:.4f}",
                "PSI": float(model_metrics.get("psi", None)),
                "KS差值(KS_train - KS_test)": float(model_metrics.get("ks_gap", None)),
                "模型寻优方式": model_search_desc,
                "最优参数": best_params
            }
        }
         # ---------------------------------------------------------------------
        # 2. 新增模块：Lift & 分箱结果
        raw_lift = model_metrics.get("df_lift", None)
        lift_df = pd.DataFrame(raw_lift) if raw_lift is not None else pd.DataFrame()
        total_bins = len(lift_df)
        # 根据实际分箱数量计算百分比
        # 注意：由于duplicates='drop'，实际分箱数量可能少于预期
        if len(lift_df) > 0:
            # 计算第一箱的实际样本比例
            first_bin_ratio = lift_df.iloc[0]['acc_num_ration']  # 累积样本比例
            first_bin_pct = round(first_bin_ratio * 100, 1)
        else:
            first_bin_pct = 0         
        # 确保有足够的分箱数据
        first_lift = lift_df.iloc[0]['lift'] if len(lift_df) > 0 else 0
        mono_violation = float(model_metrics.get("mono_violation", None))
        lift_analysis = {
            "Lift分析结果": {
                "Lift分析概览": {
                    f"前{first_bin_pct}%样本的Lift值": f"{first_lift:.1f}"
                },
                 "单调性检测": {
                    "坏样本率是否单调": "是" if mono_violation == 0 else "否",
                    "最大单调性违背幅度": mono_violation
                },
                "等频十分箱明细": []
            }
        }
        # 获取等频十分箱的Lift分箱结果
        overall_bad_rate = float(model_metrics['overall_bad_rate'])
        for _, row in lift_df.iterrows():
            lift_info = {
                "分箱": int(row['level']),  # 使用level列
                "分数区间": row['range'],  # 使用range列
                "样本数": int(row['num']),
                "好样本": int(row['good']),
                "坏样本": int(row['bad']),
                "坏率": f"{row['bad_ration']:.1%}",  # 使用bad_ration列
                "基准坏率": f"{overall_bad_rate:.1%}",
                "Lift": f"{row['lift']:.2f}"
            }
            lift_analysis["Lift分析结果"]["等频十分箱明细"].append(lift_info)
        
        # 3. 特征重要性分析
        # 计算特征数量统计
        total_features = len(feature_imp)
        non_zero_features = len(feature_imp[feature_imp['importance_score_split'] > 0])
        zero_features = total_features - non_zero_features
        
        feature_importance = {
            "入模特征数量": len(use_fea) if use_fea is not None else total_features,
            "入模特征列表": use_fea if use_fea is not None else [],
            "特征数量统计": {
                "总特征数": total_features,
                "非零重要性特征数": non_zero_features,
                "零重要性特征数": zero_features
            },
            "特征重要性TOP15": []
        }
        
        # 获取TOP15特征
        top_features = feature_imp.head(15).copy()
        if len(top_features) > 0 and top_features['importance_score_split'].sum() > 0:
            top_features['importance_ratio'] = top_features['importance_score_split'] / top_features['importance_score_split'].sum()
            top_features['importance_ratio'] = top_features['importance_ratio'].apply(lambda x: f"{x:.1%}")
            # 填充NaN为None，保证json合法
            top_features = top_features.where(~top_features.isna(), None)
            
            # 添加TOP15特征详情
            for _, row in top_features.iterrows():
                feature_importance["特征重要性TOP15"].append({
                    "特征名称": row['Feature'],
                    "中文名称": row.get('feature_cn', '未知'),
                    "数据源": row.get('data_source', '未知'),
                    "分裂重要性": int(row['importance_score_split']),
                    "增益重要性": f"{row['importance_score_gain']:.3f}" if 'importance_score_gain' in row else "N/A",
                    "重要性占比": row['importance_ratio']
                })
        
        # 整合所有结果
        report_data = {
            "report_title": "模型训练报告",
            "model_result": model_result,
            "lift_analysis":lift_analysis,
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
    print("特征重要性大于0的特征(按重要性排序)：",list(final_feature_df['Feature']) )
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
        optuna_trials_csv_path = os.path.join(dst_outs, 'optuna_trials.csv')

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
            dic_p = config['settings'].copy()
            dic_p['optuna_trials_csv_path'] = optuna_trials_csv_path
            best_params, is_optuna = train_lgb_with_optuna(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                dic_p=dic_p
            )
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
                is_optuna=is_optuna,
                use_fea = use_fea
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
        print("所有任务执行完成!")
        return True   
    except Exception as e:
        msg = traceback.format_exc()
        print(msg)
        return False
