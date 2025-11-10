import sys
import os
from .utils1 import *
from .pyce1 import *
import traceback
from copy import deepcopy
import pandas as pd
import json
from multiprocessing.pool import Pool
import config as pipe_conf
import numpy as np
from typing import List, Dict, Any
import warnings
import gc
warnings.filterwarnings("ignore")

# ---------- data common utils----------------

### 数据量大小判断
def need_batch_read(file_path, size_threshold_mb=500):
    """
    判断是否需要分批读取CSV文件
    - 文件大于 size_threshold_mb MB ，则认为需要分批读取
    """
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > size_threshold_mb:
        return True
    return False


def read_csv_by_columns(file_path, required_columns, batch_size=50):
    """
    按列分批读取CSV，避免一次加载过多列导致内存问题
    - required_columns: 主键列或业务必要字段
    - batch_size: 每批加载的非必要列数
    """
    try:
        from utils_data_clear_split import clean_input_data
        from utils_feature_select import select_features_func
        # Step 1: 读取所有列名（不读内容）
        all_columns = pd.read_csv(file_path, nrows=0).columns.tolist()

        # Step 2: 确认必须列存在
        missing_cols = [col for col in required_columns if col not in all_columns]
        if missing_cols:
            raise ValueError(f"文件缺少必要的字段：{missing_cols}")
        
        # Step 3: 拆分其他特征列为多个批次
        feature_columns = [col for col in all_columns if col not in required_columns]
        column_batches = [feature_columns[i:i + batch_size] for i in range(0, len(feature_columns), batch_size)]

        processed_chunks = []
        base_df = None

        for i, batch_cols in enumerate(column_batches):
            print(f"正在处理第 {i+1} 批特征列：{batch_cols}")

            use_cols = required_columns + batch_cols
            chunk = pd.read_csv(file_path, usecols=use_cols)

            # 自定义处理逻辑--数据清洗和-特征筛选【待完善】
            # 1. 数据清洗和特征筛选
            chunk = data_cleaning(chunk)
            
            # 2. 特征筛选
            selected_features = select_features_func(chunk, chunk)
            if required_columns:
                selected_features = list(set(selected_features + required_columns))
            chunk = chunk[selected_features]

            if base_df is None:
                base_df = chunk
            else:
                base_df = base_df.merge(chunk, on=required_columns, how='outer')  # outer 保证最大覆盖

        print(f"按列读取合并完成，总维度 = {base_df.shape}")
        return base_df

    except Exception as e:
        print(f"按列读取失败：{file_path}", exc_info=e)
        return None

def read_mul_csvs(folder_path, required_columns):
    """
    批量读取文件夹中的CSV文件，校验字段
    """
    dfs = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path)
                if required_columns:
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    if missing_cols:
                        raise ValueError(f"文件缺少必要的字段：{missing_cols}")
                dfs.append((file, df))
                print(f"读取成功：{file_path}，shape = {df.shape}")
            except Exception as e:
                print(f"读取失败：{file_path},error: {e}")
    return dfs

def read_one_csv(file_path, required_columns):
    """
    读取单个CSV文件并校验字段
    """
    try:
        df = pd.read_csv(file_path)
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"文件缺少必要的字段：{missing_cols}")
        print(f"读取成功：shape = {df.shape}")
    except Exception as e:
        print(f"读取失败：{file_path},error: {e}")
    return df

def merge_alldata(df_list, feature_list, base_fea):
    """
    合并多个DataFrame（df_list）并保留选中特征列（feature_list）
    """
    all_fea_names = []
    processed_dfs = []
    for df, fea in zip(df_list, feature_list):
        all_fea_names += fea
        processed_dfs.append(df[base_fea + fea])
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=base_fea, how='inner'), processed_dfs)
    return merged_df, all_fea_names

#### -------------### 主调函数：自动获取数据【旧函数】-------------------------
def get_data_auto(json_str):
    """
    根据配置JSON读取数据：
    - 自动判断是否需要分批读取
    - 支持按列分批逻辑
    """
    try:
        dic=json.loads(json_str)
        dst=dic['out_floder']
        data_path = dic['data_path']
        required_columns  = dic['st']['base_fea']

        if need_batch_read(data_path):
            print(f"文件较大，开始分批读取：{data_path}")
            df= read_csv_by_columns(data_path, required_columns)
            # #获取原文件名（不含路径和扩展名）
            file_name = os.path.splitext(os.path.basename(data_path))[0]  # 得到 'data'
            save_path = os.path.join(dst, f'{file_name}_bybatch.csv')
            df.to_csv(save_path, index=False)
        else:
            print(f"文件较小，可以直接读取：{data_path}")
            df = read_one_csv(data_path, required_columns)
        return df
    except Exception as e:
        msg=traceback.format_exc()
        print(msg)


def read_csv_by_rows(file_path: str, required_columns: List[str] = None, chunk_size: int = 100000) -> pd.DataFrame:
    """
    智能读取CSV文件，根据文件大小自动选择读取策略
    
    参数:
        file_path: CSV文件路径
        required_columns: 必需的列名列表
        chunk_size: 每批处理的行数，默认10万行
        
    返回:
        pd.DataFrame: 合并后的数据框
    """
    try:
        chunks = []
        total_rows = 0
        
        # 使用pandas的chunksize参数进行分批读取
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # # 如果指定了必需列，只保留这些列
            # if required_columns:
            #     chunk = chunk[required_columns]
            
            chunks.append(chunk)
            total_rows += len(chunk)
            print(f"已处理 {total_rows} 行")
            
            # 如果内存压力大，可以在这里添加内存清理
            if len(chunks) % 10 == 0:  # 每10个chunk清理一次内存
                gc.collect()
        
        # 合并所有chunks
        print("开始合并数据...")
        df = pd.concat(chunks, ignore_index=True)
        print(f"读取完成，总行数: {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"分批读取文件时出错: {str(e)}")
        raise

#### -------------### 主调函数：自动获取数据【旧函数】-------------------------
def get_data(file_path: str, required_columns: List[str] = None, chunk_size: int = 100000) -> pd.DataFrame:
    """
    智能读取CSV文件，根据文件大小自动选择读取策略
    
    参数:
        file_path: CSV文件路径（必需）
        required_columns: 必需的列名列表（可选，默认None表示读取所有列）
        chunk_size: 每批处理的行数（可选，默认100000行）
        
    返回:
        pd.DataFrame: 合并后的数据框
        
    示例:
        # 只传入文件路径，读取所有列
        df = get_data("data.csv")
        
        # 指定必需列
        df = get_data("data.csv", required_columns=["id", "date"])
        
        # 指定必需列和批次大小
        df = get_data("data.csv", required_columns=["id", "date"], chunk_size=50000)
    """
    try:
        # 1. 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 2. 检查必需列
        if required_columns:
            columns = pd.read_csv(file_path, nrows=0).columns.tolist()
            missing_cols = [col for col in required_columns if col not in columns]
            if missing_cols:
                raise ValueError(f"文件缺少必需列: {missing_cols}")
        
        # 3. 使用need_batch_read函数判断是否需要分批读取
        if need_batch_read(file_path):
            print(f"文件较大，开始分批读取，每批{chunk_size}行")
            return read_csv_by_rows(file_path, required_columns, chunk_size)
        else:
            df = pd.read_csv(file_path)
            file_name = os.path.basename(file_path)
            print(f"文件 '{file_name}' 读取成功：总行数 = {df.shape[0]}，总列数 = {df.shape[1]}")
            return df
            
        
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        raise
