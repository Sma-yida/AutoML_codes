import json
import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings("ignore")

## ================== 1.必填参数 ==================
data_file_name = 'data.csv'  #数据集名称,不用带路径，需要把数据放在【当前文件夹】的【data】下面
feature_dic_name = 'feature_dict.csv' ##特征对应的中英文映射以及数据源【如无只含有表头即可】

TRAIN_SPLIT = [20230401, 20240231] #训练集的时间范围,
OOT_SPLIT= [20240301, 20240331] #oot集合的时间范围
PSI_SPLIT=[20240301, 20240331] #psi集合的时间范围

## ================== 2. 默认参数配置，可按需修改，未指定时采用默认值
## 默认控制运行
RUN_DQ        = True   # 是否运行数据质量分析模块
RUN_HPO       = True   # 是否启用超参寻优，如不启用采用默认参数model_param
RUN_EVAL      = True   # 是否运行模型评估模块
GEN_DQ_REPORT    = True   # 是否生成数据质量报告
SAVE_DQ_OUTPUT   = True   # 是否存储数据质量报告的中间结果
GEN_EVAL_REPORT  = True   # 是否生成模型评估报告
SAVE_EVAL_OUTPUT = True   # 是否将评估的输出结果保存为CSV

## 全局config：
id_col = 'id'              # 数据集主键列字段名
date_col = 'date'          # 数据集时间字段名
label = 'label'            # 目标变量（标签）字段名
channel_col = 'channel'    # 渠道字段名
fill_val = -9999           # 缺失值默认填充值
missing_list = [-9999, -9998, -9997, -9996, ' ', '-9999', '-9998', '-9997', '-9996']  
# 缺失值的识别列表，可包括数字和字符型缺失标记
product_list_keep = []     # 渠道过滤白名单，默认为空表示全渠道建模
label_list_keep = [0, 1]   # 标签过滤白名单，当前仅支持二分类建模
if_null_importance = False # 是否启用 Null Importance 筛选特征（True：启用，False：关闭）
split_method= 'date'     # 数据切分的方式 ，随机 (rondam) or 按时间划分（date）
random_state = 42          # 全局随机参数配置
run_iter   = '1'            # 结果输出文件夹的后缀标识，用于多次运行区分

# -------- 特征筛选参数：------------- 
iv_thresh      = 0.02   # IV阈值：低于该值特征将被剔除
psi_thresh     = 0.2    # PSI阈值：高于该值说明特征稳定性差
miss_thresh    = 0.8    # 缺失率阈值：超过该比例将被剔除
corr_thresh    = 0.2    # 相关性阈值：高度相关特征剔除
use_null_importance = False # 是否启用 Null Importance 筛选特征（True：启用，False：关闭）
top_n = 500             # Null Importance 筛选特征的Top N
upload_fea_list = []    # 人工指定的特征列表，若为空则使用自动筛选结果

# -------- 模型寻优参数：-------------
model_select       = 'LightGBM'  # 当前仅支持 LightGBM，可扩展为 LR、XGB 等
n_trials           = 10           # 超参数优化迭代次数（Optuna）
ks_threshold_min   = 0.01        # KS值下限
ks_threshold_max   = 1.0         # KS值上限
ks_gap_threshold   = 1.0         # 训练集与测试集KS差值最大阈值
filter_zero_importance = True    # 是否筛除重要性为0的特征！去除后pkl模型结果将不一致。

# -------- 模型评估的相关参数：-------------
bin_num_lift    = 4            # 分箱数量（如用于Lift图等）
pdo        = 50           # PDO（Points to Double Odds）
theta      = 20           # 每一倍赔率对应的分数差
p0         = 600          # 基准分数
cut_points = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 999] #分数转换的切分点
prob_col   = 'prob_'      # 模型预测概率列名
score_col  = 'score_'     # 模型打分列名

# -------- 模型默认参数：-------------
model_param = { 
    'boosting_type':'gbdt',
    'objective':'binary',
    'random_state':random_state,
    'max_depth':-1,
    'n_jobs':-1,
    'verbose':-1,
    'class_weight':None,
    'scale_pos_weight':1,
    'subsample_for_bin':200000,
    'importance_type':'split',
    'min_child_samples':20,
    'min_split_gain':0.0,
    'learning_rate':0.033,
    'num_leaves':11,
    'max_bin':108,
    'reg_alpha':5,
    'reg_lambda':125,
    'subsample':0.539,
    'colsample_bytree':0.229,
    'subsample_freq':2,
    'min_child_weight':17,
    'n_estimators':5
    }

# ------------模型网格搜索默认参数：-------------
model_selection_params = {
            "learning_rate": [0.03, 0.2],
            "num_leaves": [8, 60],
            "max_bin": [20, 350],
            "reg_alpha": [0, 20],
            "reg_lambda": [0, 200, 5],
            "subsample": [0.2, 1.0],
            "colsample_bytree": [0.2, 1.0],
            "subsample_freq": [1, 10],
            "min_child_weight": [1, 20],
            "n_estimators": [500],
            "max_depth": [-1, -1] 
        }

## ----------路径配置 ----------------------------------------------
##获取当前文件夹目录d:\\PycharmProjects\\autoML-pipline'
file_curr_dst = os.getcwd()
sys.path.append(file_curr_dst)

data_dir = os.path.join(file_curr_dst, 'data')  # 原始/中间数据存放目录
### 任务相关目录
results_dir = os.path.join(file_curr_dst, 'results')      # 结果展示文件目录
outputs_dir = os.path.join(file_curr_dst, 'outputs')      # 中间文件主目录
startup_dir = os.path.join(file_curr_dst, 'startup')      # 启动检查文件目录

### 任务1-数据质量报告相关路径
data_quality_dir = os.path.join(outputs_dir, 'data_analysis')   # 数据质量分析输出目录
dq_report_dir    = os.path.join(data_quality_dir, 'report')     # 质量报告文件目录
dq_outputs_dir   = os.path.join(data_quality_dir, 'outputs')    # 其他相关输出目录
dq_json_path = os.path.join(results_dir, 'data_analysis.json')  # 数据质量分析结果展示文件
dq_startup_path = os.path.join(startup_dir, 'data_analysis.txt')  # 数据质量分析启动检查文件

dq_report_path = os.path.join(dq_report_dir, 'data_quality_report.xls')  # 数据质量报告路径

### 任务2-数据清洗与切分相关路径
data_clean_dir = os.path.join(outputs_dir, 'data_clean_split')  # 数据清洗与切分输出目录
cleaned_split_data_dir = os.path.join(data_clean_dir, 'data')  # 清洗和切分后的数据目录
clean_report_dir = os.path.join(data_clean_dir, 'report')       # 清洗报告目录
clean_outputs_dir = os.path.join(data_clean_dir, 'outputs')     # 其他输出目录
clean_json_path = os.path.join(results_dir, 'data_clean_split.json')  # 数据清洗与切分结果展示文件
clean_startup_path = os.path.join(startup_dir, 'data_clean_split.txt')  # 数据清洗与切分启动检查文件

cleaned_data_path =os.path.join(cleaned_split_data_dir, 'cleaned_data.csv') #清洗后的数据
dev_data_path = os.path.join(cleaned_split_data_dir, 'train.csv')     # 训练集路径
oot_data_path = os.path.join(cleaned_split_data_dir, 'test.csv')      # 测试集路径
psi_data_path = os.path.join(cleaned_split_data_dir, 'psi.csv')       # 用于计算PSI的数据路径
target_stats_report_path   = os.path.join(clean_report_dir, 'target_stats_report.csv')      # 违约率报告

### 任务3-特征筛选相关路径
feature_analysis_dir = os.path.join(outputs_dir, 'feature_select')  # 特征筛选输出目录
feature_outputs_dir = os.path.join(feature_analysis_dir, 'outputs')  # 特征分析输出目录
use_feature_path = os.path.join(feature_outputs_dir, 'use_fea.json')  # 筛选后特征列表
feature_json_path = os.path.join(results_dir, 'feature_select.json')  # 特征筛选结果展示文件
feature_startup_path = os.path.join(startup_dir, 'feature_select.txt')  # 特征筛选启动检查文件

### 任务4-模型训练相关路径
model_dir = os.path.join(outputs_dir, 'model_train')  # 模型训练输出目录
model_file_dir = os.path.join(model_dir, 'model')     # 模型文件目录
model_output_dir = os.path.join(model_dir, 'outputs')  # 模型训练其他输出目录
pkl_model_path = os.path.join(model_file_dir, 'model.pkl')  # 模型pkl文件
pmml_model_path = os.path.join(model_file_dir, 'model.pmml')  # 模型pmml文件
feature_importance_path = os.path.join(model_output_dir, 'feature_imp.csv')  # 特征重要性
model_params_path = os.path.join(model_output_dir, 'model_params.json')  # 模型参数记录
model_fea_path = os.path.join(model_output_dir, 'final_model_fea.json')  # 最终模型特征
model_json_path = os.path.join(results_dir, 'model_train.json')  # 模型训练结果展示文件
model_startup_path = os.path.join(startup_dir, 'model_train.txt')  # 模型训练启动检查文件

### 任务5-模型评估相关路径
evaluation_dir = os.path.join(outputs_dir, 'model_eval')  # 模型评估输出目录
eval_data_dir = os.path.join(evaluation_dir, 'data')      # 评估数据目录[分数转换后的数据]
eval_report_dir = os.path.join(evaluation_dir, 'report')  # 评估报告目录
eval_outputs_dir = os.path.join(evaluation_dir, 'outputs')  # 评估其他输出目录
eval_json_path = os.path.join(results_dir, 'model_eval.json')  # 模型评估结果展示文件
eval_startup_path = os.path.join(startup_dir, 'model_eval.txt')  # 模型评估启动检查文件

model_evaluation_report_path=os.path.join(eval_report_dir, 'model_evaluation_report.xlsx')

### ==================== 3.组合参数[最终的配置文件】，不可更改 ================== ================== 
base_fea = [id_col,date_col,label,channel_col] #非入模特征
data_src_path = os.path.join(data_dir, data_file_name)      ## 组合数据集路径
feature_dic_path = os.path.join(data_dir, feature_dic_name) ## 组合特征字典路径
##主流程的输入json
# ------------- 模块0：数据获取的参数配置【暂时不使用了】-----------
getdata_json_str={         
    "data_path": data_src_path,   
    "out_floder": data_dir,
    "st":{
         "id_col":id_col,
         "date_col":date_col,
         "base_fea":base_fea
        }
      }
getdata_json_str=json.dumps(getdata_json_str)

# ------------- 模块1：数据质量报告模块的参数配置-----------
dq_json_str={         
    "data_path": data_src_path,         
    "output_path": {
        'dst_rp':dq_report_dir,
        'dst_outs':dq_outputs_dir,
        'dq_report_path':dq_report_path,
        'dq_json_path':dq_json_path,
        'dq_startup_path':dq_startup_path
        },
    "settings":{
         "id_col":id_col,
         "date_col":date_col,
         'channel_col':channel_col,
         'label':label,
         "base_fea":base_fea,
         'missing_list': missing_list,
         'IF_RUN': RUN_DQ, 
         'OUT_REPORT': GEN_DQ_REPORT,
         'SAVE_OUTPUTS':SAVE_DQ_OUTPUT
        }
      }
dq_json_str=json.dumps(dq_json_str)

# ----------- 模块2：数据清洗与切分模块的参数配置-----------------
clean_split_json_str={
    "data_path": data_src_path,  
    "output_path": {
        'dst':data_clean_dir,
        'dst_data':cleaned_split_data_dir,
        'dst_rp':clean_report_dir,
        'dst_outs':clean_outputs_dir,
        'dst_rp_path':target_stats_report_path,
        'clean_json_path':clean_json_path,
        'clean_startup_path':clean_startup_path,
        'dev_data_path':dev_data_path,
        'oot_data_path':oot_data_path,
        'psi_data_path':psi_data_path,
        'cleaned_data_path':cleaned_data_path
                   },   
    "settings":{
         "id_col":id_col,
         "date_col":date_col,
         "channel_col":channel_col,
         "label":label,
         'base_fea':base_fea,
         "product_list_keep":product_list_keep,
         "label_list_keep":label_list_keep,
         "fill_val":fill_val,
         "method":split_method,     # 数据切分的方式splitby_random or 按照splitby_range
         'TRAIN_SPLIT':TRAIN_SPLIT, # 训练集的时间范围,
         'OOT_SPLIT':OOT_SPLIT,     # oot集合的时间范围
         'PSI_SPLIT':PSI_SPLIT,     # psi集合的时间范围
         "random_state":random_state
      }}  
clean_split_json_str=json.dumps(clean_split_json_str)

# ----------- 模块3：特征筛选模块的参数配置-----------------
feaure_select_json = {
    "data_path": {
        "dev": dev_data_path,
        "oot": oot_data_path
    },
    "output_path": {
        "dst": feature_analysis_dir,
        "dst_outs": feature_outputs_dir,
        "use_feature_path": use_feature_path,
        "feature_json_path": feature_json_path,
        "feature_startup_path": feature_startup_path
    },
    "settings": {
        "id_col":id_col,
        "date_col":date_col,
        "label":label,
        'base_fea':base_fea,
        "iv_thresh": iv_thresh,
        "psi_thresh": psi_thresh,  
        "miss_thresh": miss_thresh, 
        "corr_thresh": corr_thresh,
        "use_null_importance": use_null_importance,
        "top_n": top_n,
        "upload_fea_list": upload_fea_list
    }
}
feaure_select_json = json.dumps(feaure_select_json)

#--------------模块4：模型训练参数---------------
model_train_json_str={         
    "data_path":{
        "dev":dev_data_path,
        "oot":oot_data_path,
        "feature_dic_path":feature_dic_path,
        "use_fea_path":use_feature_path},
    "output_path":{
        "dst":model_dir,
        "dst_model":model_file_dir,
        "dst_outs":model_output_dir,
        "pkl_model_path":pkl_model_path,
        "pmml_model_path":pmml_model_path,
        "feature_importance_path":feature_importance_path,
        "model_params_path":model_params_path,
        'model_fea_path':model_fea_path,
        'model_json_path':model_json_path,
        'model_startup_path':model_startup_path},
    "settings":{
        "id_col":id_col,
        "label":label,
        'model_select':model_select, 
        'n_trials':n_trials,          
        'ks_threshold_min': ks_threshold_min,  
        'ks_threshold_max': ks_threshold_max,  
        'ks_gap_threshold':ks_gap_threshold,     
        'filter_zero_importance':filter_zero_importance,
        'RUN_HPO':RUN_HPO,
        "model_param":model_param,
        "model_selection_params":model_selection_params
         }
      }
model_train_json_str=json.dumps(model_train_json_str)

## ----------- 模块5：模型评估参数配置 ------------------
eva_json_str={
    "model_file":{
        "pkl_mod": pkl_model_path, ## 模型pkl文件
        "pmml_mod": pmml_model_path, ## 模型pkl文件
        "use_fea": model_fea_path
            },
    "data_path":{
        "oot": oot_data_path,
        "psi": psi_data_path, 
        "dev": dev_data_path },
    "output_path":{
        "dst":evaluation_dir,
        "dst_data":eval_data_dir,
        "dst_rp":eval_report_dir,
        "dst_outs":eval_outputs_dir,
        'eval_json_path':eval_json_path,
        'eval_startup_path':eval_startup_path,
        'model_evaluation_report_path':model_evaluation_report_path
        },
    "settings":{
         "id_col":id_col,
         "date_col":date_col,
         "label":label,
         "channel_col":channel_col,
         "base_fea":base_fea,
         "bin_num":bin_num_lift,
         'OUT_REPORT':GEN_EVAL_REPORT, 
         'OUTPUT':SAVE_EVAL_OUTPUT, 
         'IF_RUN':RUN_EVAL
        },
    "score":{
        "pdo": pdo, 
        "theta": theta, 
        "p0": p0,
        "cut_points":cut_points,
        "prob_col": prob_col, 
        "score_col": score_col 
        },
    "report_all":{
        "dq_report_path":dq_report_path,
        "feature_importance_path":feature_importance_path,
        "target_stats_report_path":target_stats_report_path,
        "feature_outputs_dir":feature_outputs_dir      
    }
}
eva_json_str = json.dumps(eva_json_str)