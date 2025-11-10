import gc
import pandas as pd
import numpy as np
from multiprocessing.pool import Pool
from .utils_data_quality import identify_types
import os
import config as pipe_conf
import warnings
import json
from concurrent.futures import ProcessPoolExecutor
import logging
import math
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import time
import traceback

warnings.filterwarnings("ignore")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量定义
STATISTIC_BOUND = -9990
MISSING_VALUES = [-99999, -9999, -9998, -9997, -9996]

def get_process_num():
    """获取CPU核心数"""
    import multiprocessing
    return max(1, multiprocessing.cpu_count() - 1)

def bining4iv(s: pd.Series, flag: int, binning_number: int):
    """IV分箱"""
    max_n = s.max()
    min_n = s[s > max(MISSING_VALUES)].min()
    init_bins = [-10000, -9999]
    binning_number = min(binning_number, len(set(s)))
    return _bining(s, max_n, min_n, init_bins, flag, binning_number)

def bining4psi(s1: pd.Series, s2: pd.Series, flag: int, binning_number: int):
    """PSI分箱"""
    max_n = max(s1.max(), s2.max()) + 0.1
    min_n = min(s1[s1 > -1].min(), s2[s2 > -1].min())
    init_bins = [-10000, min_n]
    return _bining(s1, max_n, min_n, init_bins, flag, binning_number)

def _bining(s: pd.Series, max_n: float, min_n: float, init_bins: list, flag: int, binning_number: int):
    """分箱实现"""
    bins_split = init_bins.copy()
    bin_step = 1.0 / binning_number
    
    # 等频分箱
    if flag == 1:
        perc = np.arange(0, 1, bin_step)
        node = list(s[s >= min_n].quantile(perc[1:]))
        bins_split.extend(node)
        bins_split.append(max_n)
    
    # 等距分箱
    if flag == 2:
        step_n = (max_n - min_n) / binning_number
        for i in range(binning_number):
            bins_split.append(min_n + i * step_n)
        tmp_max = max(min_n + binning_number * step_n, max_n)
        bins_split.append(tmp_max)
    
    return [x for x in np.unique(bins_split) if not math.isnan(x)]

def calculate_iv(data: pd.DataFrame, feature: str, target: str, flag: int, binning_number: int) -> tuple:
    """计算IV值"""
    df = data[[feature, target]].copy()
    df = df[df[target].notna()]  # 过滤掉目标变量为空的行
    
    # 分箱
    bin_split = bining4iv(df[feature], flag, binning_number)
    
    # 替换缺失值为-9999
    s = df[feature].replace(MISSING_VALUES + [np.nan], -9999)
    
    # 按照给定边界进行分箱
    binned_fc = pd.cut(s, bins=bin_split)
    result = df.groupby(binned_fc)[target].agg(["count", "sum"]).rename(
        columns={"count": 'total_count', "sum": "bad_count"})
    
    result['good_count'] = result['total_count'] - result['bad_count']
    
    all_total = result.total_count.sum()
    good_total = result.good_count.sum()
    bad_total = result.bad_count.sum()
    
    # 计算分布
    result['total_pcnt'] = result.apply(lambda x: "{:.2f}%".format(x.total_count / all_total * 100) \
        if all_total > 0 else "0.00%", axis=1)
    result['good_gtotal_pcnt'] = result.apply(lambda x: x.good_count / good_total if good_total > 0 else 0.0, axis=1)
    result['bad_btotal_pcnt'] = result.apply(lambda x: x.bad_count / bad_total if bad_total > 0 else 0.0, axis=1)
    
    # 计算单调性
    result['bad_rate'] = result.apply(lambda x: "{:.2f}%".format(x.bad_count / x.total_count * 100) \
        if x.total_count > 0 else "0.00%", axis=1)
    
    # 计算WOE和IV
    result['woe'] = result.apply(lambda x: np.log(x.good_gtotal_pcnt / x.bad_btotal_pcnt) \
        if x.good_gtotal_pcnt > 0 and x.bad_btotal_pcnt > 0 else 0.0, axis=1)
    result['iv'] = result.apply(lambda x: (x.good_gtotal_pcnt - x.bad_btotal_pcnt) * x.woe \
        if x.good_gtotal_pcnt > 0 and x.bad_btotal_pcnt > 0 else 0.0, axis=1)
    
    iv_value = result['iv'].sum()
    
    # 生成分箱点信息
    dist_mono_points = result[["total_count", "bad_count", "total_pcnt", "bad_rate"]].reset_index().rename(
        columns={feature: "binned"})
    dist_mono_points.insert(0, "Feature", feature)
    
    return iv_value, dist_mono_points

def calculate_psi(dev_data: pd.Series, psi_data: pd.Series, flag: int, binning_number: int) -> tuple:
    """计算PSI值"""
    # 分箱
    bins_split = bining4psi(dev_data, psi_data, flag, binning_number)
    
    # 计算PSI
    psi_value = _calculate_psi_value(dev_data, psi_data, bins_split)
    psi_value_without_missing = _calculate_psi_value(
        dev_data.replace(MISSING_VALUES, np.nan),
        psi_data.replace(MISSING_VALUES, np.nan),
        bins_split
    )
    
    return psi_value, psi_value_without_missing

def _calculate_psi_value(s1: pd.Series, s2: pd.Series, bins_split: list) -> float:
    """计算PSI值的具体实现"""
    train_num = s1.size
    test_num = s2.size
    
    tr_count_ = s1.groupby(pd.cut(s1, bins_split, right=False)).count().rename("train")
    te_count_ = s2.groupby(pd.cut(s2, bins_split, right=False)).count().rename("test")
    
    _psi_train_prec = tr_count_.apply(lambda x: np.round((x if x > 0 else x + 1) / train_num, 8))
    _psi_test_prec = te_count_.apply(lambda x: np.round((x if x > 0 else x + 1) / test_num, 8))
    
    psi_result = (_psi_train_prec - _psi_test_prec) * (_psi_train_prec / _psi_test_prec).apply(
        lambda x: math.log(x, math.e)).round(4)
    
    psi_result.replace(to_replace=np.inf, value=0.0, inplace=True)
    psi_result.fillna(0.0, inplace=True)
    
    return psi_result.sum()

def process_feature(args):
    """处理单个特征的统计信息"""
    feature, data, psi_data, target, iv_binning_number, iv_binning_type, psi_binning_number, psi_binning_type,idx,total = args
    try:
        print(f"正在处理特征: {feature} ({idx+1}/{total})")
        # 计算IV值
        iv_value, mono_stats = calculate_iv(data, feature, target, iv_binning_type, iv_binning_number)
        
        # 计算PSI值
        psi_value, psi_value_without_missing = calculate_psi(
            data[feature], psi_data[feature], psi_binning_type, psi_binning_number
        )
        
        # 计算缺失率
        # missing_rate = (data[feature] <= STATISTIC_BOUND).mean()
        missing_rate = data[feature].replace(MISSING_VALUES, np.nan).isna().mean()
        
        # 计算其他统计量
        valid_data = data[feature][data[feature] > STATISTIC_BOUND]
        mean_value = valid_data.mean()
        std_value = valid_data.std(ddof=1)
        min_value = valid_data.min()
        max_value = valid_data.max()
        unique_rate = valid_data.value_counts(dropna=True).max() / len(data)
        neg_rate = (data[feature] < 0).mean()
        
        return {
            'Feature': feature,
            'IV': iv_value,
            'PSI': psi_value,
            'PSI(剔特殊值)': psi_value_without_missing,
            'missing': f"{missing_rate:.2%}",
            'mean': mean_value,
            'std': std_value,
            'min': min_value,
            'max': max_value,
            '单一值占比': f"{unique_rate:.2%}",
            '<0': f"{neg_rate:.2%}",
            'mono_stats': mono_stats
        }
    except Exception as e:
        logger.error(f"Error processing feature {feature}: {str(e)}")
        return None

# def calculate_correlation(data: pd.DataFrame, sorted_features: list, threshold: float) -> list:
#     """
#     计算特征相关性并返回需要删除的特征，sorted_features需已按IV降序、特征名升序排序。
#     相关性计算前将-9999, -9998, -9997, -9996替换为np.nan，
#     只有当pearson和spearman相关性都大于阈值时才剔除后面的特征。
#     """
#     # 替换特殊缺失值为 NaN
#     data_corr = data[sorted_features].replace(MISSING_VALUES, np.nan)
#     # 计算两种相关性
#     pearson_corr = data_corr.corr(method='pearson').abs()
#     spearman_corr = data_corr.corr(method='spearman').abs()
#     to_drop = set()
#     for i in range(len(sorted_features)):
#         for j in range(i + 1, len(sorted_features)):
#             f1, f2 = sorted_features[i], sorted_features[j]
#             if f1 in to_drop or f2 in to_drop:
#                 continue
#             if (pearson_corr.loc[f1, f2] > threshold) or (spearman_corr.loc[f1, f2] > threshold):
#                 to_drop.add(f2)
#     return list(to_drop)

def calculate_correlation(data: pd.DataFrame, sorted_features: list, threshold: float) -> list:
    """
    计算特征相关性并返回需要删除的特征，sorted_features需已按IV降序、特征名升序排序。
    相关性计算前将-9999, -9998, -9997, -9996替换为np.nan，
    只有当pearson和spearman相关性都大于阈值时才剔除后面的特征。
    优化：用numpy下标访问加速。
    """
    # 替换特殊缺失值为 NaN
    data_corr = data[sorted_features].replace(MISSING_VALUES, np.nan)
    pearson_corr = data_corr.corr(method='pearson').abs().values
    spearman_corr = data_corr.corr(method='spearman').abs().values
    n = len(sorted_features)
    to_drop = np.zeros(n, dtype=bool)
    for i in range(n):
        if to_drop[i]:
            continue
        for j in range(i + 1, n):
            if to_drop[j]:
                continue
            if (pearson_corr[i, j] > threshold) or (spearman_corr[i, j] > threshold):
                to_drop[j] = True
    return [sorted_features[i] for i in range(n) if to_drop[i]]
    
def get_feature_importances(data: pd.DataFrame, target: str, features: list, shuffle: bool, seed: int = None) -> pd.DataFrame:
    """计算特征重要性"""
    np.random.seed(817)
    # 只使用目标变量为0或1的数据
    data = data[data[target].isin([0,1])]
    
    # 如果需要打乱目标变量
    y = data[target].copy()
    if shuffle:
        y = data[target].copy().sample(frac=1.0)
    
    # 使用LightGBM的随机森林模式
    dtrain = lgb.Dataset(data[features], y, free_raw_data=False)
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'subsample': 0.6,
        'colsample_bytree': 0.8,
        'num_leaves': 31,
        'max_depth': 4,
        'seed': seed,
        'bagging_freq': 1,
        'verbose':-1,
        'n_jobs': -1
    }
    
    # 训练模型
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=60)

    # 获取特征重要性
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')  
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(y, clf.predict(data[features]))
    
    return imp_df

def get_null_importance(data: pd.DataFrame, target: str, features: list, nb_runs: int = 80) -> pd.DataFrame:
    """计算空重要性分布"""
    null_imp_df = pd.DataFrame()
    start = time.time()
    dsp = ''
    
    for i in range(nb_runs):
        # 获取当前运行的重要性
        imp_df = get_feature_importances(data, target, features, shuffle=True)
        imp_df['run'] = i + 1 
        # 合并结果
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
        # 显示进度
        spent = (time.time() - start) / 60
        dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
        print(dsp, end='', flush=True)
        print('\r', end='', flush=True)

    return null_imp_df

def null_importance_select(data: pd.DataFrame, target: str, features: list, null_result_path: str, top_n: int = 500) -> list:
    """使用Null Importance方法选择特征"""
    # 计算实际重要性
    actual_imp_df = get_feature_importances(data, target, features, shuffle=False)
    actual_imp_df.to_csv(os.path.join(null_result_path, 'actual_imp_df.csv'), index=None, header=True)

    # 计算空重要性
    null_imp_df = get_null_importance(data, target, features)
    null_imp_df.to_csv(os.path.join(null_result_path, 'null_imp_df.csv'), index=None, header=True)

    # 计算特征得分
    feature_scores = []
    for _f in actual_imp_df['feature'].unique():
        f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
        gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))
        
        f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
        split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))
        
        feature_scores.append((_f, split_score, gain_score))

    # 保存得分结果
    scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])
    scores_df.to_csv(os.path.join(null_result_path, 'scores_df.csv'), index=None)
    
    # 选择top N特征
    scores_df = scores_df.sort_values(by='gain_score', ascending=False)
    feature_list = scores_df.feature.to_list()      
    select_cnt = min(top_n, len(feature_list))
    feature_list_selected = feature_list[0:select_cnt]
    
    return feature_list_selected

def generate_feature_select_json(dev_data, use_fea, single_var, output_path, base_fea=None):
    """
    生成特征筛选报告的JSON文件
    
    参数:
        dev_data: DataFrame, 原始数据
        use_fea: list, 筛选后的特征列表
        single_var: DataFrame, 特征统计信息
        output_path: str, 输出路径
        base_fea: list, 基础特征列表
    """
    try:
        # 如果没有传入base_fea，使用默认值[]
        if base_fea is None:
            base_fea = []
            
        # 1. 特征筛选概览
        total_features = len(dev_data.columns) - len(base_fea)
        selected_features = len(use_fea)
        removed_features = total_features - selected_features
        
        overview = {
            "总特征数": total_features,
            "筛选后特征数": selected_features,
            "移除特征数": removed_features
        }
        
        # 2. 特征详情（IV值最高的前10个筛选后特征）
        # 从single_var中筛选出use_fea中的特征，并按IV值排序
        top_features = single_var[single_var['Feature'].isin(use_fea)].sort_values('IV', ascending=False).head(10)
        
        # 格式化特征详情
        feature_details = []
        for _, row in top_features.iterrows():
            feature_details.append({
                "特征名称": row['Feature'],
                "IV值": f"{float(row['IV']):.3f}",
                "PSI值": f"{float(row['PSI']):.3f}",
                "缺失率": row['missing'],
                "均值": f"{float(row['mean']):.1f}" if not pd.isna(row['mean']) else "-",
                "最大值": f"{float(row['max']):.1f}" if not pd.isna(row['max']) else "-",
                "最小值": f"{float(row['min']):.1f}" if not pd.isna(row['min']) else "-",
                "单一值占比": row['单一值占比'],
                "负值占比": row['<0']
            })
        
        # 整合所有结果
        report_data = {
            "report_title": "特征筛选报告",
            "overview": overview,
            "feature_details": feature_details
        }
        
        # 转换为JSON
        json_str = json.dumps(report_data, ensure_ascii=False, indent=2)
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        return output_path
            
    except Exception as e:
        logger.error(f"Error generating feature selection JSON report: {str(e)}")
        return None

def select_features_func(dev_data, psi_data, config):
    """特征选择主函数"""
    try:
        # 从config中获取参数
        settings = config['settings']
        output_path = config['output_path']
        
        id_col = settings['id_col']
        date_col = settings['date_col']
        label_col = settings['label']
        base_fea = settings['base_fea']
        iv_thresh = settings['iv_thresh']
        psi_thresh = settings['psi_thresh']
        miss_thresh = settings['miss_thresh']
        corr_thresh = settings['corr_thresh']
        use_null_importance = settings['use_null_importance']
        top_n = settings['top_n']
        upload_fea_list = settings['upload_fea_list']
        
        output_dst = output_path['dst_outs']
        use_fea_path = output_path['use_feature_path']
        feature_json_path = output_path['feature_json_path']
        single_var_path = output_path['single_var_path']
        mono_var_path = output_path['mono_var_path']
        
        # 创建输出目录
        os.makedirs(output_dst, exist_ok=True)
        
        # 可选：人工上传覆盖特征
        if upload_fea_list:
            logger.info("已选择自定义的入模特征")
            
            # 定义可用的特征列表（排除基础特征）
            available_features = [feat for feat in dev_data.columns if feat not in base_fea]
            
            # 检查特征是否存在于数据中（排除基础特征）
            missing_features = [feat for feat in upload_fea_list if feat not in available_features]
            if missing_features:
                error_msg = f"以下特征在数据中不存在：{missing_features}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 检查人工上传的特征是否都是数值型
            non_numeric_features = []
            for feat in upload_fea_list:
                # 检查数据类型是否为数值型
                if dev_data[feat].dtype in ['object', 'string', 'category']:
                    non_numeric_features.append(feat)
            
            if non_numeric_features:
                error_msg = f"当前建模不适用非数值型特征，请检查以下特征：{non_numeric_features}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 所有检查通过后，使用上传的特征列表
            use_fea = upload_fea_list

            # 检查是否需要重新计算统计信息
            need_recalc = True
            if os.path.exists(single_var_path):
                try:
                    single_var_df = pd.read_csv(single_var_path)
                    # 检查Feature列是否包含所有upload_fea_list
                    if set(upload_fea_list).issubset(set(single_var_df['Feature'])):
                        single_var = single_var_df
                        need_recalc = False
                except Exception as e:
                    logger.warning(f"读取单变量统计分析文件single_var.csv失败，将重新计算: {e}")

            if need_recalc:
                # 计算自定义特征的统计信息
                n_cpus = min(get_process_num(), 4)
                args_list = [(feat, dev_data, psi_data, label_col, 10, 1, 10, 1, idx, len(upload_fea_list)) 
                            for idx, feat in enumerate(upload_fea_list)]
                
                if len(args_list) < 10:
                    results = [process_feature(args) for args in args_list]
                else:
                    with ProcessPoolExecutor(max_workers=n_cpus) as executor:
                        results = list(executor.map(process_feature, args_list))
                
                single_var = pd.DataFrame([{k: v for k, v in r.items() if k != 'mono_stats'} for r in results if r is not None])
                mono_var = pd.concat([r['mono_stats'] for r in results if r is not None])
                
                # 保存特征统计分析结果
                single_var.to_csv(single_var_path, index=False)
                mono_var.to_csv(mono_var_path, index=False)
        else:
            # 数据识别
            num_fea, _ = identify_types(df=dev_data, remove_fea=base_fea)
            
            # 使用多进程计算特征统计量
            n_cpus = min(get_process_num(), 4)  # 限制最大进程数，避免Windows多进程问题
            args_list = [(feat, dev_data, psi_data, label_col, 10, 1, 10, 1, idx, len(num_fea)) for idx, feat in enumerate(num_fea)]
            
            # 如果特征数量较少，使用单进程
            if len(args_list) < 10:
                results = [process_feature(args) for args in args_list]
            else:
                with ProcessPoolExecutor(max_workers=n_cpus) as executor:
                    results = list(executor.map(process_feature, args_list))
            
            # 整理结果
            single_var = pd.DataFrame([{k: v for k, v in r.items() if k != 'mono_stats'} for r in results if r is not None])
            mono_var = pd.concat([r['mono_stats'] for r in results if r is not None])
            
            # 保存特征统计分析结果
            single_var.to_csv(single_var_path, index=False)
            mono_var.to_csv(mono_var_path, index=False)
            
            # 特征筛选
            single_var[["IV", "PSI"]] = single_var[["IV", "PSI"]].astype(float)
            single_var["missing"] = single_var["missing"].str.rstrip('%').astype(float) / 100
            
            filter_fea = single_var.loc[
                (single_var.IV >= iv_thresh) & 
                (single_var.missing < miss_thresh) & 
                (single_var.PSI < psi_thresh),
                'Feature'
            ].values
            
            logger.info(f"After IV>{iv_thresh}, PSI<{psi_thresh}, Missingrate<{miss_thresh}, use_fea counts: {len(filter_fea)}")
            
            # 如果启用Null Importance特征选择
            if use_null_importance:
                logger.info("Using Null Importance for feature selection")
                filter_fea = null_importance_select(dev_data, label_col, filter_fea, output_dst, top_n)
                logger.info(f"After Null Importance selection, use_fea counts: {len(filter_fea)}")
            
            # 相关性分析
            iv_dict = {f: single_var.loc[single_var['Feature'] == f, 'IV'].values[0] for f in filter_fea}
            sorted_features = sorted(iv_dict.keys(), key=lambda x: (-iv_dict[x], x))
            corr_drop = calculate_correlation(dev_data, sorted_features, corr_thresh)
            use_fea = [i for i in sorted_features if i not in corr_drop]
            
            logger.info(f"Corr<{corr_thresh}, use_fea counts: {len(use_fea)}")
        
        # 保存结果
        with open(use_fea_path, "w", encoding="utf-8") as f:
            json.dump(use_fea, f, ensure_ascii=False, indent=4)
        
        # 生成特征筛选报告
        generate_feature_select_json(dev_data, use_fea, single_var, feature_json_path, base_fea)
        
        print(f"筛选后特征数量: {len(use_fea)}")
        print(f"入模特征为：", use_fea)
        return True
        
    except Exception as e:
        msg = traceback.format_exc()
        print(msg)
        return False


