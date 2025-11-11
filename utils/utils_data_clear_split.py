import pandas as pd
import warnings
from .utils_data_quality import identify_types
from sklearn.model_selection import train_test_split
import os
import json
import traceback
import copy

import pandas as pd
import os

def compute_stats(df, label_col):
    return {
        'n_samples': len(df),
        'good': (df[label_col] == 0).sum(),
        'bad': (df[label_col] == 1).sum(),
        'grey': (df[label_col] == 2).sum(),
        'bad_rate': df[label_col].astype(int).replace({2: 0}).mean()
    }

def compute_monthly_stats(df, label_col, time_col):
    df['month'] = pd.to_datetime(df[time_col], format='%Y%m%d').dt.to_period('M').astype(str)
    grouped = df.groupby('month')
    results = []
    for month, group in grouped:
        stats = compute_stats(group, label_col)
        stats['month'] = month
        results.append(stats)
    return pd.DataFrame(results)

def spliter_report(X_train, X_test, time_col, label_col, save_path):
    result = []

    for df, name in [(X_train, 'dev'), (X_test, 'oot')]:
        # Overall stats
        result.append(pd.DataFrame([compute_stats(df.copy(), label_col)]).assign(dataset=name, month='Overall'))
        # Monthly stats
        result.append(compute_monthly_stats(df.copy(), label_col, time_col).assign(dataset=name))

    df_all = pd.concat(result, ignore_index=True)
    df_all = df_all[['dataset', 'month', 'n_samples', 'good', 'bad', 'grey', 'bad_rate']]

    print('数据集分布统计')
    print(df_all[df_all['month'] == 'Overall'])
    
    df_all.to_csv(save_path, index=False)
    return df_all

#数据清洗-样本选择
def clean_input_data(data,label,channel_col,num_fea, product_list_keep=None, label_list_keep=None,fill_val=-9999,min_sample_size=500):
    """
    模型训练前的数据清洗函数

    参数说明：
    - data: 原始DataFrame
    - num_fea: 数值特征-转换为数值
    - product_list_keep: 保留的产品渠道列表（为空则不筛选）
    - label_list_keep: 保留的标签列表（为空则默认保留[0, 1]）
    - fill_val: 缺失值填充值
    - start_month/end_month: 月份筛选（字符串格式 "202301"）
    - min_sample_size: 筛选后最小样本数，默认500

    返回：
    - 清洗后的数据
    """
    df_clear = copy.deepcopy(data)
    # 渠道筛选
    if len(product_list_keep) !=0:
        df_clear = df_clear[df_clear[channel_col].isin(product_list_keep)]

    # 标签筛选
    if len(label_list_keep)==0:
        label_list_keep = [0, 1]
    df_clear = df_clear[df_clear[label].isin(label_list_keep)]

    # # 时间筛选
    # if start_date and end_date:
    #     df_clear[date_col] = df_clear[date_col].astype(str)
    #     df_clear = df_clear[(df_clear[date_col] >= start_date) & (df_clear[date_col] <= end_date)]

    # 缺失值填充
    df_clear.fillna(fill_val, inplace=True)
    df_clear.replace('', fill_val, inplace=True)
    df_clear.reset_index(drop=True, inplace=True)

    # # 数值特征转换为数值
    # data.fillna(fill_val, inplace=True)

    # 样本量校验
    if len(df_clear) < min_sample_size:
        warnings.warn(f"样本量仅为 {len(df_clear)}，小于设定阈值 {min_sample_size}，可能不适合建模！")

    df_clear = df_clear.reset_index(drop=True)
    return df_clear

# 时间范围校验
def check_range_valid(name, date_range):
    if len(date_range) != 2 or date_range[0] > date_range[1]:
        raise ValueError(f"{name} 的时间范围设置不合理: {date_range}")
    
###  按照指定的时间范围划分
def split_by_date_range(data,date_col,train_range,oot_range,psi_range):
    """
    按照指定的时间范围划分数据集，要求所有范围都不为空，否则抛出异常。
    """
    if not train_range or not oot_range or not psi_range:
        raise ValueError(f"[split_by_date_range] train_range/oot_range/psi_range 不能有空值，当前值：train_range={train_range}, oot_range={oot_range}, psi_range={psi_range}")
    check_range_valid('TRAIN_SPLIT', train_range)
    check_range_valid('OOT_SPLIT', oot_range)
    check_range_valid('PSI_SPLIT', psi_range)
    # 切分数据 -类型转为int，保证比较正确
    data[date_col] = data[date_col].astype(int)
    train_df = data[(data[date_col] >= train_range[0]) & (data[date_col] <= train_range[1])].copy()
    test_df = data[(data[date_col] >= oot_range[0]) & (data[date_col] <= oot_range[1])].copy()
    psi_df = data[(data[date_col] >= psi_range[0]) & (data[date_col] <= psi_range[1])].copy()
    return train_df, test_df, psi_df
    
# ... 按照时间排序-比例划分 ...
def split_by_date_proportion(data, date_col, test_size=0.2):
    """
    按时间排序后用比例切分，前面为训练集，后面为测试集
    返回train_df, test_df, psi_df，并输出切分方式和时间范围
    """
    data_sorted = data.sort_values(by=date_col)
    n = len(data_sorted)
    split_idx = int(n * (1 - test_size))
    train_df = data_sorted.iloc[:split_idx].copy()
    test_df = data_sorted.iloc[split_idx:].copy()
    psi_df = test_df.copy()
    return train_df, test_df, psi_df

def split_randomly(data,test_size = 0.2,random_state = 42):
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    psi_df = test_df.copy()
    return train_df, test_df,psi_df

    
#============================数据清洗主程序==========================================    
def data_clear_func(df,config):
    try:
        save_dir=config['output_path']['dst_data']
        data_path = config['data_path']
        base_fea = config['settings']['base_fea']
        date_col = config['settings']['date_col']
        channel_col = config['settings']['channel_col']
        label = config['settings']['label']
        product_list_keep = config['settings']['product_list_keep']
        label_list_keep = config['settings']['label_list_keep']
        fill_val = config['settings']['fill_val']
        
        os.makedirs(save_dir, exist_ok=True)
        # 数据识别
        num_fea, _ = identify_types(df=df,remove_fea=base_fea)
        # 样本筛选和缺失值填充
        data = clean_input_data(data=df, channel_col =channel_col, label =label,num_fea =num_fea, 
                                product_list_keep =product_list_keep,label_list_keep =label_list_keep,fill_val =fill_val,min_sample_size=500)
        
        # 保存清洗后的所有数据到指定路径，不保存索引
        cleaned_data_path = config['output_path']['cleaned_data_path']
        data.to_csv(cleaned_data_path, index=False) 
        print(f"数据清洗完成，清洗后大小为 {data.shape[0]} 行 {data.shape[1]} 列，清洗数据已保存")    
        return data
    except Exception as e:
        msg=traceback.format_exc()
        print(msg)


  
def generate_clear_split_json(data, data_cleared, method, train_df, test_df, df_all, output_path, channel_col, date_col):
    """
    生成数据清洗和切分的JSON报告
    
    参数:
        data: DataFrame, 原始数据
        data_cleared: DataFrame, 清洗后的数据
        method: str, 切分方法
        train_df: DataFrame, 训练集数据
        test_df: DataFrame, 测试集数据
        df_all: DataFrame, 数据集分布情况
        output_path: str, 输出路径
        channel_col: str, 渠道列名
        date_col: str, 日期列名
    """
    try:
        # 1. 数据清洗概览
        cleaning_overview = {
            "原始样本数量": len(data),
            "清洗后样本数量": len(data_cleared),
            "适用渠道": sorted(data_cleared[channel_col].unique().tolist())
        }
        
        # 2. 数据切分方式
        if method == "date_range":
            split_method = {
                "切分方法": "按指定时间范围切分",
                "训练集(DEV)时间范围": f"{train_df[date_col].min()}-{train_df[date_col].max()}",
                "测试集(OOT)时间范围": f"{test_df[date_col].min()}-{test_df[date_col].max()}"
            }
        elif method == "date_proportion":
            split_method = {
                "切分方法": "按时间排序比例切分",
                "训练集(DEV)时间范围": f"{train_df[date_col].min()}-{train_df[date_col].max()}",
                "测试集(OOT)时间范围": f"{test_df[date_col].min()}-{test_df[date_col].max()}"
            }
        else:  # random
            split_method = {
                "切分方法": "随机切分",
                "训练集(DEV)时间范围": "随机划分",
                "测试集(OOT)时间范围": "随机划分"
            }
        
        # 3. 数据集分布情况
        # 整体分布
        overall_stats = df_all[df_all['month'] == 'Overall']
        overall_distribution = []
        for _, row in overall_stats.iterrows():
            if row['dataset'] == 'dev':
                time_range = f"{train_df[date_col].min()}-{train_df[date_col].max()}"
            else:
                time_range = f"{test_df[date_col].min()}-{test_df[date_col].max()}"    
            overall_distribution.append({
                "数据集": row['dataset'],
                "时间范围": time_range,
                "样本量": int(row['n_samples']),
                "好样本": int(row['good']),
                "坏样本": int(row['bad']),
                "灰样本": int(row['grey']),
                "违约率": f"{float(row['bad_rate']):.1%}"
            })
        
        # 月度分布
        monthly_stats = df_all[df_all['month'] != 'Overall']
        monthly_distribution = {
            "训练集(DEV)": [],
            "测试集(OOT)": []
        }
        
        for _, row in monthly_stats.iterrows():
            month_stats = {
                "月份": row['month'],
                "样本量": int(row['n_samples']),
                "好样本": int(row['good']),
                "坏样本": int(row['bad']),
                "灰样本": int(row['grey']),
                "违约率": f"{float(row['bad_rate']):.1%}"
            }
            if row['dataset'] == 'dev':
                monthly_distribution["训练集(DEV)"].append(month_stats)
            else:
                monthly_distribution["测试集(OOT)"].append(month_stats)
        
        # 整合所有结果
        report_data = {
            "report_title": "数据清洗与切分报告",
            "cleaning_overview": cleaning_overview,
            "split_method": split_method,
            "distribution": {
                "整体分布": overall_distribution,
                "按月份分布": monthly_distribution
            }
        }
        
        # 转换为JSON
        json_str = json.dumps(report_data, ensure_ascii=False, indent=2)
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        return output_path
            
    except Exception as e:
        print(f"Error generating JSON report: {str(e)}")
        return None

#============================数据切分主程序==========================================  
def data_split_func(orl_data, config):
    try:
        ## 在这里加载数据清洗流程
        data = data_clear_func(orl_data, config)

        save_dir = config['output_path']['dst_data']
        dst_rp = config['output_path']['dst_rp']
        dst_rp_path = config['output_path']['dst_rp_path']
        clean_json_path = config['output_path']['clean_json_path']
        # 解包配置
        id_col = config['settings']['id_col']
        date_col = config['settings']['date_col']
        channel_col = config['settings']['channel_col']
        label = config['settings']['label']
        method = config['settings']['method'].lower()
        random_state = config['settings']['random_state']
        test_size = config['settings']['test_size']
        train_range = config['settings'].get('train_split_range', [])
        oot_range = config['settings'].get('oot_split_range', [])
        psi_range = config['settings'].get('psi_split_range', [])
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(dst_rp, exist_ok=True)
    
        # 划分数据集
        if method == 'date_range':
            train_df, test_df, psi_df = split_by_date_range(data, date_col, train_range, oot_range, psi_range) 
        elif method == 'date_proportion':
            train_df, test_df, psi_df = split_by_date_proportion(data, date_col, test_size)
        else:  # method == 'random'
            train_df, test_df, psi_df = split_randomly(data, test_size, random_state)
        
        train_time = f"{train_df[date_col].min()} - {train_df[date_col].max()}"
        test_time = f"{test_df[date_col].min()} - {test_df[date_col].max()}"
        print(f"[切分方式] {method} | 训练集时间范围: {train_time} | 测试集时间范围: {test_time}")

        # 保存清洗后的数据为 csv 文件，不保存索引  
        train_df.to_csv(config['output_path']['dev_data_path'], index=False)
        test_df.to_csv(config['output_path']['oot_data_path'], index=False)
        psi_df.to_csv(config['output_path']['psi_data_path'], index=False)
 
        # 生成数据集分布报告
        df_all = spliter_report(train_df, test_df, date_col, label, dst_rp_path)
        
        # 生成JSON报告-用于页面展示
        generate_clear_split_json(
            orl_data,
            data_cleared=data,  # 这里假设data已经是清洗后的数据
            method=method,
            train_df=train_df,
            test_df=test_df,
            df_all=df_all,
            output_path=clean_json_path,
            channel_col=channel_col,
            date_col=date_col
        )     
        return True
    except Exception as e:
        msg=traceback.format_exc()
        print(msg)
        return False
