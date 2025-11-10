import argparse
import json
import os
import pandas as pd
from utils.utils_data_quality import data_quality
from utils.data_utils import get_data
import datetime

def run(config_path, session_id=None):
    """
    运行数据质量分析任务
    
    参数:
        config_path (str): 配置文件JSON的路径
        session_id (str, optional): 用于跟踪运行的会话ID
    """
    try:
        # 1.加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        # 2. 写入启动指示器
        startup_file = config['output_path']['dq_startup_path']
        with open(startup_file, 'w') as f:
            f.write('1')
            
        # 2.5 打印启动信息和当前时间
        log_message = f"[START] 【数据质量分析】启动于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        print(log_message)
        
        # 3.读取数据
        data_path = config['data_path']
        required_columns = config['settings']['base_fea']
        df = get_data(data_path, required_columns)
        
        # 4.运行数据质量分析
        results = data_quality(df, config)
        
        if results:
            print("数据质量分析完成")
            return True
        else:
            print("数据质量分析未执行 (IF_RUN=False)")
            return False
            
    except Exception as e:
        print(f"运行数据质量分析时出错: {str(e)}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='运行数据质量分析')
    parser.add_argument('-conf', '--config', required=True, help='配置文件JSON的路径')
    parser.add_argument('--session-id', help='用于跟踪运行的会话ID')
    
    args = parser.parse_args()
    
    run(args.config, args.session_id)
