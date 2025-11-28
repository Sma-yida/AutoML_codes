import pandas as pd
import numpy as np
import xlwt
import utils.utils_report as rpt
import os
import json
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
from dateutil.relativedelta import relativedelta

def identify_types(df,remove_fea=None):
    """
    identify_types: 识别数值型和类别型变量
    参数:
        df: DataFrame，输入数据
        remove_fea: list，需要排除的字段（如ID列、日期列）
    返回:
        num_fea: list，数值型变量名列表（保持原始列顺序）
        cat_fea: list，类别型变量名列表（保持原始列顺序）
    """
    # 将 remove_fea 转换为 set 以提高查找效率
    remove_fea_set = set(remove_fea) if remove_fea else set()
    
    cat_fea = []
    num_fea = []
    
    # 遍历所有列，保持原始顺序，只遍历一次以提高性能
    for col in df.columns:
        if col not in remove_fea_set:
            if df[col].dtype == 'object':
                cat_fea.append(col)
            else:
                num_fea.append(col)
    
    return num_fea, cat_fea	

def describe_missing(df,features,missing_list):
	"""
    describe_missing: 全面统计变量缺失情况（包括NaN和特定缺失值）
    """
	miss_name_all = []
	miss_df_tmp = {} 
	if len(missing_list) > 0 :
		for i in missing_list :
			if type(i)==str:
				miss_name = 'missing_'+str(i)+'_str'
			else:
				miss_name = 'missing_'+str(i)+'_num'
			miss_name_all.append(miss_name)
			miss_cnt = []
			for ii in features :
				miss_cnt.append(sum(np.where(df[ii]==i ,1 ,0 ))/len(df))
				miss_df_tmp[miss_name] = miss_cnt 
		miss_df_tmp = pd.DataFrame.from_dict(miss_df_tmp)
		miss_df_tmp['feature_name'] = features
	#缺失为空
	nan_total = df[features].isnull().sum()/len(df)
	nan_total = nan_total.reset_index()
	nan_total.columns=['feature_name','missing_nan']
	#变量类型
	fea_type = df[features].dtypes.reset_index()
	fea_type.columns=['feature_name','type']
	fea_type['total_count'] = len(df)
	#缺失率整体
	missing_data = pd.merge(fea_type,nan_total,on='feature_name',how='left')
	missing_data = pd.merge(missing_data,miss_df_tmp,on='feature_name',how='left')
	missing_data['missing_tot'] =missing_data[miss_name_all+['missing_nan']].sum(axis=1)
	return missing_data	

def describe_missing_simplified(df,features,missing_list):
    """
    describe_missing_simplified: 简化版缺失统计（统一将missing_list视作缺失）
    """
    missing_data_simplifed = pd.DataFrame(columns=['feature_name','missing_tot'])
    tot_cnt = df.shape[0]
    for val in features:
        tmp = df[val].copy().replace(missing_list,np.nan)
        missing_data_simplifed=missing_data_simplifed.append([{
            'feature_name': val,
            'missing_tot':tmp.isnull().sum()/tot_cnt
            }],ignore_index=True)
    return missing_data_simplifed

def describe_num_sta(df,features,missing_list):
    """
    describe_num_sta: 数值型变量的描述性统计（均值、方差、分位数等）
    """
    col_desc_all = {}
    for i in features :
        tmp = df[df[i].notnull()][~df[i].isin(missing_list)]	
        X1 = len(tmp)
        X2 = tmp[i].mean()
        X3 = tmp[i].std()
        X4 = tmp[i].min()
        X5 = tmp[i].quantile(0.1)
        X6 = tmp[i].quantile(0.2)
        X7 = tmp[i].quantile(0.3)
        X8 = tmp[i].quantile(0.4)
        X9 = tmp[i].quantile(0.5)
        X10= tmp[i].quantile(0.6)
        X11= tmp[i].quantile(0.7)
        X12= tmp[i].quantile(0.8)
        X13= tmp[i].quantile(0.9)
        X14= tmp[i].max()
        X15= tmp[i].quantile(0.01)
        X16= tmp[i].quantile(0.05)
        X17= tmp[i].quantile(0.95)
        X18= tmp[i].quantile(0.99)
        col_desc = [X1,X2,X3,X4,X15,X16,X5,X6,X7,X8,X9,X10,X11,X12,X13,X17,X18,X14]
        col_desc_all[i] =col_desc
    col_desc_all = pd.DataFrame.from_dict(col_desc_all,orient='index').reset_index()
    col_desc_all.columns=['feature_name','count', 'mean', 'std', 'min', 'p01','p05','p10', 'p20','p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90', 'p95','p99','max']
    col_desc_all['cv'] = col_desc_all.apply(lambda x : '-' if x['mean'] == 0 else x['std']/x['mean'],axis=1  )	
    #变量类型
    fea_type = df[features].dtypes.reset_index()
    fea_type.columns=['feature_name','type']
    #
    con_desc_df =pd.merge(fea_type,col_desc_all,left_on='feature_name',right_on='feature_name',how='left')
    con_desc_df = con_desc_df[['feature_name','count', 'mean', 'std', 'min', 'p01','p05','p10', 'p20','p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90', 'p95','p99','max', 'cv']]
    return con_desc_df		

#2.03.03 类别型统计量【！！！有问题需要优化！！！】

def describe_cat_sta(df,features,missing_list):
    """
    describe_cat_sta: 类别变量的频数统计（TOP5及其他计数）
    参数:
        df: DataFrame, 输入数据
        features: list, 类别型变量列表
        missing_list: list, 缺失值列表
    返回:
        DataFrame, 包含每个类别变量的统计信息：
        - feature_name: 变量名
        - count: 非空值数量
        - unique_cnt: 唯一值数量
        - value1-5: 频率最高的5个值
        - cnt1-5: 对应的频数
        - ratio1-5: 对应的频率
        - cnt_other: 其他值的频数
        - ratio_other: 其他值的频率
    """
    col_desc_all = {} 
    for i in features :
        tmp = df[df[i].notnull()][~df[i].isin(missing_list)]	
        count = len(tmp)
        unique_cnt = df[i].nunique()
        #统计量详情
        value_counts = tmp[i].value_counts(ascending = False).head(5)
        var_tmp = pd.DataFrame({
            'value': value_counts.index,
            'cnt': value_counts.values
        })
        var_tmp_df = pd.DataFrame({'num':[1,2,3,4,5]})
        var_tmp['count'] = count
        var_tmp = pd.merge(var_tmp_df,var_tmp,left_index=True,right_index=True,how='left')
        var_tmp['cnt_ratio'] = var_tmp['cnt']/var_tmp['count']
        var_detail = [count,unique_cnt] + list(var_tmp['value'])+list(var_tmp['cnt'])+list(var_tmp['cnt_ratio'])
        col_desc_all[i] = var_detail
    col_desc_all = pd.DataFrame.from_dict(col_desc_all,orient='index').reset_index()
    value_name=['value1','value2','value3','value4','value5']
    cnt_name =['cnt1','cnt2','cnt3','cnt4','cnt5']
    ratio_name=['ratio1','ratio2','ratio3','ratio4','ratio5']
    col_desc_all.columns = ['feature_name','count','unique_cnt','value1','value2','value3','value4','value5','cnt1','cnt2','cnt3','cnt4','cnt5','ratio1','ratio2','ratio3','ratio4','ratio5']
    col_desc_all['cnt_other'] =col_desc_all['count']-col_desc_all[cnt_name].sum(axis=1)
    col_desc_all['ratio_other'] =1-col_desc_all[ratio_name].sum(axis=1)
    col_desc_all= col_desc_all[['feature_name','count','unique_cnt','value1','value2','value3','value4','value5','cnt1','cnt2','cnt3','cnt4','cnt5','cnt_other','ratio1','ratio2','ratio3','ratio4','ratio5','ratio_other']]
    #变量类型
    fea_type = df[features].dtypes.reset_index()
    fea_type.columns=['feature_name','type']
    #
    cat_desc_df= pd.merge(fea_type,col_desc_all,on='feature_name',how='left')
    return cat_desc_df

def describe_dup_sta(df,id_col):
    """
    describe_dup_sta: 样本ID重复性检查
    """
    dup_sta = pd.DataFrame({
        'count': [df.shape[0]],
        'unique_cnt':[len(df[id_col].value_counts())],
        'max_dup': [df[id_col].value_counts().max()]
        },columns=['count','unique_cnt','max_dup'])
    return dup_sta
    
def parse_flexible_date(date_str):
    """
    支持解析8位(年月日)和6位(年月)两种格式的日期字符串
    """
    if pd.isnull(date_str):
        return pd.NaT
    s = str(date_str)
    try:
        if len(s) == 8:
            return pd.to_datetime(s, format='%Y%m%d', errors='coerce')
        elif len(s) == 6:
            return pd.to_datetime(s, format='%Y%m', errors='coerce')
        else:
            return pd.NaT
    except Exception:
        return pd.NaT

def build_dup_sta_with_overview(df, id_col, channel_col=None, date_col=None):
    # 基本重复ID检查
    dup_sta = pd.DataFrame({
        'count': [df.shape[0]],
        'unique_cnt': [len(df[id_col].value_counts())],
        'max_dup': [df[id_col].value_counts().max()]
    })
    # 渠道统计
    if channel_col and channel_col in df.columns:
        channels = df[channel_col].dropna().unique().tolist()
        dup_sta['channel_count'] = len(channels)
        dup_sta['channel_list'] = [channels[:10] if len(channels) > 10 else channels]
    else:
        dup_sta['channel_count'] = None
        dup_sta['channel_list'] = None
    # 时间范围
    if date_col and date_col in df.columns:
        try:
            df['temp_date'] = df[date_col].apply(parse_flexible_date)
            valid_dates = df.dropna(subset=['temp_date'])
            if len(valid_dates) > 0:
                min_date = valid_dates['temp_date'].min()
                max_date = valid_dates['temp_date'].max()
                delta = relativedelta(max_date, min_date)
                months = delta.years * 12 + delta.months
                days = delta.days
                dup_sta['date_range'] = [{
                    "开始日期": min_date.strftime('%Y-%m-%d'),
                    "结束日期": max_date.strftime('%Y-%m-%d'),
                    "时间跨度": f"{months}个月{days}天",
                    "有效样本数量": len(valid_dates)
                }]
            else:
                dup_sta['date_range'] = [None]
        except Exception as e:
            dup_sta['date_range'] = [f"date parse error: {str(e)}"]
    else:
        dup_sta['date_range'] = [None]
    return dup_sta


def describe_con_sta(df,features,missing_list):
    """
    describe_con_sta: 单一值统计
    """
    con_sta = pd.DataFrame(columns=['feature_name','type','count','unique_cnt','top','freq_ratio'])
    tot_cnt = df.shape[0]
    dfTypeDict = dict(df.dtypes)
    for val in features:
        tmp = df[val].copy().replace(missing_list,np.nan)
        value_cnt = tmp.value_counts()
        con_sta = pd.concat([con_sta, pd.DataFrame([{
        'feature_name': val,
        'type': str(dfTypeDict[val]),
        'count': tmp.count(),
        'unique_cnt': len(value_cnt),
        'top': value_cnt.index[0],
        'freq': value_cnt.iloc[0],
        'freq_ratio': value_cnt.iloc[0]/tot_cnt
    }])], ignore_index=True)
    return con_sta

def compute_default_stats(df, label_col, group_col=None):
    """
    计算违约率统计
    参数:
        df: DataFrame, 输入数据
        label_col: str, 标签列名
        group_col: str, 分组列名(可选)
    返回:
        DataFrame, 包含违约率统计结果
    """
    if group_col is None:
        # 处理缺失值
        valid_data = df.dropna(subset=[label_col])
        stats = {
            'group': 'all',  # 添加group列，表示总体统计
            'total_samples': len(valid_data),
            'good': (valid_data[label_col] == 0).sum(),
            'bad': (valid_data[label_col] == 1).sum(),
            'grey': (valid_data[label_col] == 2).sum(),
            'bad_rate': round(valid_data[label_col].astype(int).replace({2: 0}).mean(), 4)
        }
        return pd.DataFrame([stats])
    else:
        # 处理缺失值
        valid_data = df.dropna(subset=[label_col, group_col])
        grouped = valid_data.groupby(group_col)
        results = []
        for group_name, group in grouped:
            stats = {
                'group': group_name,
                'total_samples': len(group),
                'good': (group[label_col] == 0).sum(),
                'bad': (group[label_col] == 1).sum(),
                'grey': (group[label_col] == 2).sum(),
                'bad_rate': round(group[label_col].astype(int).replace({2: 0}).mean(), 4)
            }
            results.append(stats)
        return pd.DataFrame(results)

def compute_cross_default_stats(df, label_col, group_col1, group_col2):
    """
    计算交叉违约率统计
    参数:
        df: DataFrame, 输入数据
        label_col: str, 标签列名
        group_col1: str, 第一个分组列名
        group_col2: str, 第二个分组列名
    返回:
        DataFrame, 包含交叉违约率统计结果
    """
    grouped = df.groupby([group_col1, group_col2])
    results = []
    for (group1, group2), group in grouped:
        stats = {
            'group1': group1,
            'group2': group2,
            'total_samples': len(group),
            'good': (group[label_col] == 0).sum(),
            'bad': (group[label_col] == 1).sum(),
            'grey': (group[label_col] == 2).sum(),
            'bad_rate': group[label_col].astype(int).replace({2: 0}).mean()
        }
        results.append(stats)
    return pd.DataFrame(results)

def plot_missing_heatmap(missing_sta, figsize=(12, 8)):
    """
    绘制缺失值热力图
    参数:
        missing_sta: DataFrame, 包含feature_name和missing_tot列的缺失值统计数据
        figsize: tuple, 图表大小，默认为(12, 8)
    返回:
        matplotlib.figure.Figure, 热力图对象
    """
    # 准备数据
    missing_data = missing_sta[['feature_name', 'missing_tot']].copy()
    missing_data.columns = ['feature', 'missing_rate']
    missing_data = missing_data.sort_values('missing_rate', ascending=False)
    
    # 绘制热力图
    plt.figure(figsize=figsize)
    sns.heatmap(missing_data.set_index('feature').T, 
                cmap='YlOrRd', 
                cbar_kws={'label': 'Missing Rate'})
    plt.title('Feature Missing Rate Heatmap')
    plt.tight_layout()
    return plt.gcf()

def plot_default_rate_trend(default_sta, figsize=(12, 6)):
    """
    绘制违约率趋势图
    """
    # 筛选月度数据
    if 'group' not in default_sta.columns:
        return None
        
    # 确保group列不包含NA/NaN值
    monthly_data = default_sta.dropna(subset=['group']).copy()
    monthly_data = monthly_data[monthly_data['group'].str.match(r'\d{4}-\d{2}', na=False)]
    
    if len(monthly_data) == 0:
        return None
        
    monthly_data['month'] = pd.to_datetime(monthly_data['group'])
    monthly_data = monthly_data.sort_values('month')
    
    plt.figure(figsize=figsize)
    plt.plot(monthly_data['month'], monthly_data['bad_rate'], marker='o')
    plt.title('Default Rate Trend')
    plt.xlabel('Month')
    plt.ylabel('Default Rate')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

def plot_channel_default_rate(default_sta, figsize=(10, 6)):
    """
    绘制渠道违约率对比图
    """
    if default_sta is None or len(default_sta) == 0:
        return None
        
    # 确保group列存在且不包含NA/NaN值
    if 'group' not in default_sta.columns:
        return None
        
    # 处理NA/NaN值并筛选非月度数据
    channel_data = default_sta.dropna(subset=['group']).copy()
    channel_data = channel_data[~channel_data['group'].str.match(r'\d{4}-\d{2}', na=False)]
    
    if len(channel_data) == 0:
        return None
        
    plt.figure(figsize=figsize)
    plt.bar(channel_data['group'], channel_data['bad_rate'])
    plt.title('Default Rate by Channel')
    plt.xlabel('Channel')
    plt.ylabel('Default Rate')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    return plt.gcf()

def plot_cross_default_heatmap(cross_default_sta, figsize=(12, 8)):
    """
    绘制交叉违约率热力图
    """
    if cross_default_sta is None or len(cross_default_sta) == 0:
        return None
        
    # 重塑数据为热力图格式
    heatmap_data = cross_default_sta.pivot(
        index='group1', 
        columns='group2', 
        values='bad_rate'
    )
    
    plt.figure(figsize=figsize)
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.2%', 
                cmap='YlOrRd',
                cbar_kws={'label': 'Default Rate'})
    plt.title('Cross Default Rate Heatmap')
    plt.tight_layout()
    return plt.gcf()

def data_quality_rpt(ws, missing_sta, num_sta, cat_sta, dup_sta, con_sta, default_sta=None, cross_default_sta=None):
    """
    data_quality_rpt: 数据质量报告
    """
    row_mark = 0
    ws.write(row_mark, 0, '1.样本重复ID检查')
    row_mark += 2
    
    ws = rpt.df_writer(df=dup_sta, start_row=row_mark, start_col=1, ws=ws)
    
    row_mark += dup_sta.shape[0] + 1 + 2
    
    ws.write(row_mark, 0, '2.缺失值及单一值')
    row_mark += 2
    
    df_pt2 = pd.merge(missing_sta, con_sta[['feature_name','unique_cnt','top','freq','freq_ratio']], how='inner', on=['feature_name'])
    ws = rpt.df_writer(df=df_pt2, start_row=row_mark, start_col=1, ws=ws)
    
    row_mark += df_pt2.shape[0] + 1 + 2
    ws.write(row_mark, 0, '3.特征分布统计')
    row_mark += 2
    ws.write(row_mark, 0, '3.1 数值型')
    row_mark += 1
    ws = rpt.df_writer(df=num_sta, start_row=row_mark, start_col=1, ws=ws)
    row_mark += num_sta.shape[0] + 1 + 2
    ws.write(row_mark, 0, '3.2 类别型')
    row_mark += 1
    ws = rpt.df_writer(df=cat_sta, start_row=row_mark, start_col=1, ws=ws)
    
    if default_sta is not None:
        row_mark += cat_sta.shape[0] + 1 + 2
        ws.write(row_mark, 0, '4.违约率分析')
        row_mark += 2
        ws = rpt.df_writer(df=default_sta, start_row=row_mark, start_col=1, ws=ws)
        
        if cross_default_sta is not None:
            row_mark += default_sta.shape[0] + 1 + 2
            ws.write(row_mark, 0, '5.交叉违约率分析')
            row_mark += 2
            ws = rpt.df_writer(df=cross_default_sta, start_row=row_mark, start_col=1, ws=ws)
    
    return ws
def generate_data_quality_json(missing_sta, num_sta, dup_sta, con_sta, default_sta=None, output_path=None):
    """
    生成数据质量分析报告的JSON文件【用于界面展示】
    
    参数:
        missing_sta: DataFrame, 缺失值统计结果
        num_sta: DataFrame, 数值型变量统计结果
        cat_sta: DataFrame, 类别型变量统计结果
        dup_sta: DataFrame, 样本重复ID检查结果【整体统计】
        con_sta: DataFrame, 集中度分析结果
        default_sta: DataFrame, 违约率统计结果
        output_path: str, JSON文件保存路径，如果为None则返回JSON字符串
    
    返回:
        str: 如果output_path为None，返回JSON字符串；否则返回保存路径
    """
    try:
        # 1. 样本整体概述（包含重复ID、渠道、时间范围等）
        sample_check = {
            "数据总量": int(dup_sta['count'].iloc[0]),
            "唯一ID数量": int(dup_sta['unique_cnt'].iloc[0]),
            "最大重复次数": int(dup_sta['max_dup'].iloc[0])
        }
        # 加入渠道统计
        if 'channel_count' in dup_sta.columns and pd.notna(dup_sta['channel_count'].iloc[0]):
            sample_check["样本渠道数量"] = int(dup_sta['channel_count'].iloc[0])
            sample_check["渠道列表"] = dup_sta['channel_list'].iloc[0]
        # 加入时间范围统计
        if 'date_range' in dup_sta.columns and dup_sta['date_range'].iloc[0]:
            sample_check["样本时间范围"] = dup_sta['date_range'].iloc[0]

            
        # 2. 特征分析
        # 合并所有特征信息
        feature_stats = pd.merge(
            missing_sta[['feature_name', 'missing_nan']],
            con_sta[['feature_name', 'unique_cnt', 'top', 'freq', 'freq_ratio']],
            how='inner',
            on=['feature_name']
        )
        
        # 添加数值型特征的统计信息
        numeric_features = num_sta[['feature_name', 'mean', 'std', 'max', 'min', 'p50', 'cv']]
        feature_stats = pd.merge(
            feature_stats,
            numeric_features,
            how='left',
            on=['feature_name']
        )
        
        # 格式化特征统计表
        feature_table = []
        for _, row in feature_stats.iterrows():
            feature_row = {
                "特征名称": row['feature_name'],
                "数据类型": "float64" if pd.notna(row['mean']) else "object",
                "总缺失率": f"{float(row['missing_nan']):.1%}",
                "非空值数量": int(row['freq']),
                "唯一值数量": int(row['unique_cnt']),
                "最常见值": str(row['top']),
                "频率": f"{float(row['freq_ratio']):.1%}"
            }
            
            # 为数值型特征添加统计信息
            if pd.notna(row['mean']):
                feature_row.update({
                    "平均值": f"{float(row['mean']):.1f}",
                    "标准差": f"{float(row['std']):.1f}",
                    "最大值": f"{float(row['max']):.1f}",
                    "最小值": f"{float(row['min']):.1f}",
                    "中位数": f"{float(row['p50']):.1f}",
                    "变异系数": f"{float(row['cv']):.3f}"
                })
            else:
                feature_row.update({
                    "平均值": "-",
                    "标准差": "-",
                    "最大值": "-",
                    "最小值": "-",
                    "中位数": "-",
                    "变异系数": "-"
                })
            
            feature_table.append(feature_row)
        
        feature_analysis = {
            "特征总数": len(feature_stats),
            "数值型特征数": len(feature_stats[pd.notna(feature_stats['mean'])]),
            "类别型特征数": len(feature_stats[pd.isna(feature_stats['mean'])]),
            "特征明细": feature_table
        }
        
        # 3. 违约率分析
        default_analysis = {}
        if default_sta is not None:
            # 总体情况
            overall_stats = default_sta[default_sta['group'] == 'all'].iloc[0]
            default_analysis["总体情况"] = {
                "样本总量": int(overall_stats['total_samples']),
                "好样本数量": int(overall_stats['good']),
                "坏样本数量": int(overall_stats['bad']),
                "灰样本数量": int(overall_stats['grey']),
                "平均违约率": f"{float(overall_stats['bad_rate']):.1%}"
            }
            
            # 按渠道的违约率
            channel_stats = default_sta[~default_sta['group'].str.match(r'\d{4}-\d{2}', na=False)]
            if len(channel_stats) > 0:
                default_analysis["按渠道"] = []
                for _, row in channel_stats.iterrows():
                    channel_dict = {
                        "渠道": str(row['group']),
                        "样本总量": int(row['total_samples']),
                        "好样本数量": int(row['good']),
                        "坏样本数量": int(row['bad']),
                        "灰样本数量": int(row['grey']),
                        "违约率": f"{float(row['bad_rate']):.1%}"
                    }
                    default_analysis["按渠道"].append(channel_dict)
            
            # 按月份的违约率
            monthly_stats = default_sta[default_sta['group'].str.match(r'\d{4}-\d{2}', na=False)]
            if len(monthly_stats) > 0:
                default_analysis["按月份"] = []
                for _, row in monthly_stats.iterrows():
                    month_dict = {
                        "月份": str(row['group']),
                        "样本总量": int(row['total_samples']),
                        "好样本数量": int(row['good']),
                        "坏样本数量": int(row['bad']),
                        "灰样本数量": int(row['grey']),
                        "违约率": f"{float(row['bad_rate']):.1%}"
                    }
                    default_analysis["按月份"].append(month_dict)
        
        # 整合所有结果
        report_data = {
            "报告标题": "数据质量分析报告",
            "样本整体概述": sample_check,
            "特征分析": feature_analysis,
            "违约率分析": default_analysis
        }
        
        # 转换为JSON
        json_str = json.dumps(report_data, ensure_ascii=False, indent=2)
        
        # 保存或返回
        # output_path = os.path.join(output_path, 'qa_show.json')
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            return output_path
        else:
            return json_str
            
    except Exception as e:
        print(f"Error generating JSON report: {str(e)}")
        return None
#============================主程序==========================================    
def data_quality(df, config):
    """
    参数
    ----------
    df : DataFrame
        待分析的数据框
    config : dict
        配置参数字典，包含以下关键信息:
        - output_path: 输出目录配置
            - dst_rp: 报告输出目录
            - dst_outs: 中间结果输出目录
            - dq_report_path: 数据质量报告文件路径
            - dq_json_path:   数据质量分析结果展示文件
        - settings: 基础配置
            - id_col: ID列名
            - date_col: 日期列名
            - channel_col: 渠道列名
            - label: 标签列名
            - base_fea: 基础特征列表
            - missing_list: 缺失值列表
            - OUT_REPORT: 是否输出报告
            - SAVE_OUTPUTS: 是否保存中间结果
            - IF_RUN: 是否运行分析
            
    返回
    -------
    bool
        如果分析成功完成返回True，否则返回False
    """
    try:
        dst_rp = config['output_path']['dst_rp']
        dst_outs = config['output_path']['dst_outs']
        dq_report_path = config['output_path']['dq_report_path']
        dq_json_path = config['output_path']['dq_json_path']
        id_col = config['settings']['id_col']
        date_col = config['settings']['date_col']
        channel_col = config['settings']['channel_col']
        label = config['settings']['label']
        base_fea = config['settings']['base_fea']
        missing_list = config['settings']['missing_list']
        #  run_data_quality = config['settings']['run_data_quality']
        gen_report = config['settings']['gen_report']
        save_output = config['settings']['save_output']

        if not os.path.exists(dst_rp):
            os.makedirs(dst_rp)
        if not os.path.exists(dst_outs):
            os.makedirs(dst_outs)
        # if run_data_quality:
        #2.03.01 数据识别
        num_fea, cat_fea = identify_types(df=df,remove_fea=base_fea)
        #2.02 缺失率分析
        missing_sta = describe_missing(df=df,features=num_fea+cat_fea,missing_list=missing_list)
        #2.03.02 数值型统计量
        num_sta = describe_num_sta(df=df,features=num_fea,missing_list=missing_list)
        #2.03.03 类别型统计量
        try:
            cat_sta = describe_cat_sta(df=df,features=cat_fea,missing_list=missing_list)
        except:
            cat_sta = pd.DataFrame(columns=['feature_name','count','unique_cnt','value1','value2','value3','value4','value5','value6','value7','value8','value9','value10','cnt1','cnt2','cnt3','cnt4','cnt5','cnt6','cnt7','cnt8','cnt9','cnt10','cnt_other','ratio1','ratio2','ratio3','ratio4','ratio5','ratio6','ratio7','ratio8','ratio9','ratio10','ratio_other'])
        #2.04 集中度分析
        con_sta = describe_con_sta(df=df,features=num_fea+cat_fea,missing_list=missing_list)
        #2.05 重复值分析
        # dup_sta = describe_dup_sta(df=df,id_col=id_col)
        #2.05 样本整体分析【唯一值等】
        dup_sta = build_dup_sta_with_overview(df=df, id_col=id_col, channel_col=channel_col, date_col=date_col)

        #2.06 违约率分析
        default_sta = None
        cross_default_sta = None
        if label in df.columns:
            # 总体违约率
            default_sta = compute_default_stats(df=df, label_col=label)
            
            # 按渠道的违约率
            if channel_col in df.columns:
                channel_default = compute_default_stats(df=df, label_col=label, group_col=channel_col)
                default_sta = pd.concat([default_sta, channel_default], ignore_index=True)
            
            # 按月份的违约率
            if date_col in df.columns:
                try:
                    # 先尝试转换日期，处理无效日期
                    df['month'] = pd.to_datetime(df[date_col], format='%Y%m%d', errors='coerce')
                    # 只保留有效的日期数据
                    valid_dates = df.dropna(subset=['month'])
                    valid_dates['month'] = valid_dates['month'].dt.to_period('M').astype(str)
                    monthly_default = compute_default_stats(df=valid_dates, label_col=label, group_col='month')
                    default_sta = pd.concat([default_sta, monthly_default], ignore_index=True)
                except Exception as e:
                    print(f"Warning: Error processing dates: {str(e)}")
                    # 如果日期处理失败，继续处理其他统计
            
            # 交叉违约率分析
            if channel_col in df.columns and date_col in df.columns:
                try:
                    # 使用相同的日期处理逻辑
                    df['month'] = pd.to_datetime(df[date_col], format='%Y%m%d', errors='coerce')
                    valid_dates = df.dropna(subset=['month'])
                    valid_dates['month'] = valid_dates['month'].dt.to_period('M').astype(str)
                    
                    cross_default_sta = compute_cross_default_stats(
                        df=valid_dates, 
                        label_col=label,
                        group_col1=channel_col,
                        group_col2='month'
                    )
                except Exception as e:
                    print(f"Warning: Error in cross default rate analysis: {str(e)}")
                    cross_default_sta = None
       
        # 生成并保存JSON报告用于界面展示
        generate_data_quality_json(
            missing_sta=missing_sta,
            num_sta=num_sta,
            dup_sta=dup_sta,
            con_sta=con_sta,
            default_sta=default_sta,
            output_path=dq_json_path
        )
        if save_output:
            ## 单项结果保存
            missing_sta.to_csv(os.path.join(dst_outs, 'qa_missing_sta.csv'),index=False)
            num_sta.to_csv(os.path.join(dst_outs, 'qa_num_sta.csv'),index=False)
            cat_sta.to_csv(os.path.join(dst_outs, 'qa_cat_sta.csv'),index=False)
            con_sta.to_csv(os.path.join(dst_outs, 'qa_con_sta.csv'),index=False)
            dup_sta.to_csv(os.path.join(dst_outs, 'qa_dup_sta.csv'),index=False)
            
            # 保存图表
            # # 保存缺失值热力图
            # missing_plot = plot_missing_heatmap(missing_sta)
            # missing_plot.savefig(os.path.join(dst_outs, 'missing_heatmap_'+str(run_iter)+'.png'), format='png', dpi=100, bbox_inches='tight')
            # plt.close(missing_plot)
            
            if default_sta is not None:
                default_sta.to_csv(os.path.join(dst_outs, 'qa_default_sta.csv'),index=False)
                
                # # 保存违约率趋势图
                # trend_plot = plot_default_rate_trend(default_sta)
                # if trend_plot is not None:
                #     trend_plot.savefig(os.path.join(dst_outs, 'default_rate_trend.png'), format='png', dpi=100, bbox_inches='tight')
                #     plt.close(trend_plot)
                
                # # 保存渠道违约率对比图
                # channel_plot = plot_channel_default_rate(default_sta)
                # if channel_plot is not None:
                #     channel_plot.savefig(os.path.join(dst_outs, 'channel_default_rate.png'), format='png', dpi=100, bbox_inches='tight')
                #     plt.close(channel_plot)
                
                if cross_default_sta is not None:
                    cross_default_sta.to_csv(os.path.join(dst_outs, 'qa_cross_default_sta.csv'),index=False)
                    
                    # # 保存交叉违约率热力图
                    # cross_plot = plot_cross_default_heatmap(cross_default_sta)
                    # if cross_plot is not None:
                    #     cross_plot.savefig(os.path.join(dst_outs, 'cross_default_heatmap.png'), format='png', dpi=100, bbox_inches='tight')
                    #     plt.close(cross_plot)

        if gen_report:
            #统计结果汇总
            wb = xlwt.Workbook()
            sheet_name = 'dq_rpt'
            ws = wb.add_sheet(sheet_name,cell_overwrite_ok=False)
            ws = data_quality_rpt(
                ws=ws,
                missing_sta=missing_sta,
                num_sta=num_sta,
                cat_sta=cat_sta,
                dup_sta=dup_sta,
                con_sta=con_sta,
                default_sta=default_sta,
                cross_default_sta=cross_default_sta
            )
            wb.save(dq_report_path)
        return True
        # else:
        #     return False
    except Exception as e:
        msg=traceback.format_exc()
        print(msg)

def format_data_quality_output(dup_sta, missing_sta, num_sta, cat_sta, default_sta=None, cross_default_sta=None):
    """
    格式化数据质量分析结果，生成易读的输出格式
    
    参数:
        dup_sta: DataFrame, 样本重复ID检查结果
        missing_sta: DataFrame, 缺失值统计结果
        num_sta: DataFrame, 数值型变量统计结果
        cat_sta: DataFrame, 类别型变量统计结果
        default_sta: DataFrame, 违约率统计结果
        cross_default_sta: DataFrame, 交叉违约率统计结果
        
    返回:
        str: 格式化后的输出文本
    """
    output = []
    
    # 1. 样本重复ID检查结果
    output.append("1. 样本重复ID检查结果")
    output.append(f"当前上传数据的样本总量为：{dup_sta['count'].iloc[0]:,}条")
    output.append(f"唯一ID数量为：{dup_sta['unique_cnt'].iloc[0]:,}个")
    output.append(f"最大重复次数为：{dup_sta['max_dup'].iloc[0]:,}次")
    output.append("")
    
    # 2. 特征分析
    total_features = len(missing_sta)
    output.append(f"2. 特征分析")
    output.append(f"数据集共包含{total_features}个特征，前5个特征详情如下：")
    
    # 合并数值型和类别型特征的统计信息
    feature_stats = pd.concat([
        num_sta[['feature_name', 'type', 'count', 'mean', 'std', 'max', 'min', 'p50', 'cv', 'unique_cnt']],
        cat_sta[['feature_name', 'type', 'count', 'unique_cnt']]
    ])
    
    # 添加缺失率信息
    feature_stats = feature_stats.merge(
        missing_sta[['feature_name', 'missing_tot']], 
        on='feature_name', 
        how='left'
    )
    
    # 添加最常见值信息
    feature_stats = feature_stats.merge(
        cat_sta[['feature_name', 'value1', 'ratio1']], 
        on='feature_name', 
        how='left'
    )
    
    # 格式化前5个特征的输出
    top_5_features = feature_stats.head(5)
    for _, row in top_5_features.iterrows():
        output.append(f"\n特征名称：{row['feature_name']}")
        output.append(f"数据类型：{row['type']}")
        output.append(f"总缺失率：{row['missing_tot']:.2%}")
        if row['type'] != 'object':
            output.append(f"平均值：{row['mean']:.4f}")
            output.append(f"标准差：{row['std']:.4f}")
            output.append(f"最大值：{row['max']:.4f}")
            output.append(f"最小值：{row['min']:.4f}")
            output.append(f"中位数：{row['p50']:.4f}")
            output.append(f"变异系数：{row['cv']:.4f}")
        output.append(f"非空值数量：{row['count']:,}")
        output.append(f"唯一值数量：{row['unique_cnt']:,}")
        if pd.notna(row['value1']):
            output.append(f"最常见值：{row['value1']}")
            output.append(f"频率：{row['ratio1']:.2%}")
    output.append("")
    
    # 3. 违约率分析
    if default_sta is not None:
        output.append("3. 违约率分析")
        # 总体违约率
        overall_stats = default_sta[default_sta['group'] == 'all'].iloc[0]
        output.append(f"样本总量：{overall_stats['total_samples']:,}条")
        output.append(f"好样本数量：{overall_stats['good']:,}条")
        output.append(f"坏样本数量：{overall_stats['bad']:,}条")
        output.append(f"灰样本数量：{overall_stats['grey']:,}条")
        output.append(f"平均违约率：{overall_stats['bad_rate']:.2%}")
        output.append("")
        
        # 按渠道的违约率
        channel_stats = default_sta[~default_sta['group'].str.match(r'\d{4}-\d{2}', na=False)]
        if len(channel_stats) > 0:
            output.append("按渠道的违约率分布：")
            for _, row in channel_stats.iterrows():
                output.append(f"\n渠道：{row['group']}")
                output.append(f"样本量：{row['total_samples']:,}条")
                output.append(f"违约率：{row['bad_rate']:.2%}")
            output.append("")
        
        # 按月份的违约率
        monthly_stats = default_sta[default_sta['group'].str.match(r'\d{4}-\d{2}', na=False)]
        if len(monthly_stats) > 0:
            output.append("按月份的违约率趋势：")
            for _, row in monthly_stats.iterrows():
                output.append(f"\n月份：{row['group']}")
                output.append(f"样本量：{row['total_samples']:,}条")
                output.append(f"违约率：{row['bad_rate']:.2%}")
    
    return "\n".join(output)
