import argparse
import json
import os
import pandas as pd
import sys
import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.utils_model_evaluation import evaluation_func
from utils.data_utils import get_data

def run(config_path, session_id=None):
    """
    运行模型评估任务
    
    参数:
        config_path (str): 配置文件JSON的路径
        session_id (str, optional): 用于跟踪运行的会话ID
    """
    try:
        # 1.加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # 2. 写入启动指示器
        startup_file = config['output_path']['eval_startup_path']
        with open(startup_file, 'w') as f:
            f.write('1')
         
        # 2.5 打印启动信息和当前时间
        log_message = f"[START] 【模型评估】启动于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        print(log_message)
        
        # 3.读取数据
        dev_data_path = config['data_path']['dev_data_path']
        oot_data_path = config['data_path']['oot_data_path']
        psi_data_path = config['data_path']['psi_data_path']
        # 读取训练集和测试集数据
        dev_df = get_data(dev_data_path)
        oot_df = get_data(oot_data_path)
        psi_df = get_data(psi_data_path)
      
        # 4.运行模型评估
        result = evaluation_func(dev_df,oot_df,psi_df,config)
        
        if result is not None:
            log_message = f"[END] 【模型评估完成】于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            print(log_message)
            return True
        else:
            log_message = f"[FAIL] 【模型评估执行失败或未执行】于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            print(log_message)
            return False
            
    except Exception as e:
        print(f"运行模型评估时出错: {str(e)}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='运行模型评估')
    parser.add_argument('-conf', '--config', required=True, help='配置文件JSON的路径')
    parser.add_argument('--session-id', help='用于跟踪运行的会话ID')
    
    args = parser.parse_args()
    
    run(args.config, args.session_id) 