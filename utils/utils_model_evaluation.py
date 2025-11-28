import os  
import json
import pandas as pd
import traceback
import warnings
warnings.filterwarnings("ignore")

# import sys
# sys.path.append('/PycharmProjects/autoML-pipline/utils/')
from .utils1 import *
from .pyce1 import *
from .utils_data_quality import identify_types
from copy import deepcopy
import pickle

def get_auc_ks(dataset_type,df, target, col, pred):
    auc, ks, flag = [], [], []
    total_num, good_num, bad_num, bad_ratio = [], [], [], []
    item = list(sorted(df[col].unique()))
    for i in item:
        tmp = df[df[col] == i]
        total_num.append(tmp.shape[0])
        bad_num.append(tmp[target].sum())
        good_num.append(tmp.shape[0] - tmp[target].sum())
        bad_ratio.append(tmp[target].mean())
        # 新增：判断类别数
        if tmp[target].nunique() < 2:
            auc.append(None)
            ks.append(None)
        else:
            auc.append(roc_auc_score(tmp[target], tmp[pred]))
            ks.append(ks_score(tmp[target], tmp[pred]))
        flag.append(i)
    return pd.DataFrame({
        'dataset':dataset_type,
        'col': flag,
        'total': total_num,
        'good': good_num,
        'bad': bad_num,
        'bad_ratio': bad_ratio,
        'auc': auc,
        'ks': ks
    })

def get_final_result(dataset_type, df, target, pred, col, dst):
    """
    获取模型评估结果
    
    参数:
        dataset_type : str,数据类型[dev|oot]
        df: DataFrame, 输入数据
        target: str, 目标变量列名
        pred: str, 预测概率列名
        col: str, 分组列名
        dst: str, 输出目录
    """
    # 获取整体表现
    df['overall'] = 'overall'
    overall_df = get_auc_ks(dataset_type, df, col='overall', target=target, pred=pred)
    # 获取分渠道表现
    channel_df = get_auc_ks(dataset_type, df, col=col, target=target, pred=pred)
    
    # 保存
    overall_df.to_csv(os.path.join(dst, f'{dataset_type}_overall_auc_ks.csv'), index=False)
    channel_df.to_csv(os.path.join(dst, f'{dataset_type}_{col}_auc_ks.csv'), index=False)
    return overall_df, channel_df

def evaluate_psi(train_df, test_df, pred_col):
    psi = Psi()
    psi.fit(train_df[pred_col].values)
    return psi.transform(test_df[pred_col].values)[0]


def model_pred(df, mod, mod_var, prob_col):
    """
    对输入的df数据进行模型预测并打分

    参数：
        df: 原始数据 DataFrame
        mod: 已训练模型
        mod_var: 模型使用的特征名列表
        prob_col: 预测概率列名

    返回：
        包含预测概率和评分的新 df
    """
    df = df.copy()
    # 模型预测概率
    df[prob_col] = mod.predict_proba(df[mod_var])[:, 1]
    return df

def lift_analysis(test_df, label_col, pred_col,bin_num):
    base = test_df[label_col].mean()
    # lift_low = test_df[test_df[score_col] <= 450][label_col].mean() / base
    # lift_high = test_df[test_df[score_col] > 700][label_col].mean() / base
    # lift_ratio = lift_low / lift_high
    # print("score>700:",lift_high,"score<=450:",lift_low)

    lift_20 = sta_groups(test_df[label_col], test_df[pred_col], bins_num=bin_num, labels=list(range(bin_num, 0, -1)))
    return {
        'base_rate': base,
        'lift_top_20': lift_20
    }

def generate_evaluation_json(train_overall_df, test_overall_df, test_channel_df, psi_value, lift_res, output_path, train_date_range, test_date_range, date_col):
    """
    生成模型评估的JSON报告
    
    参数:
        train_overall_df: DataFrame, 训练集整体模型效果数据
        test_overall_df: DataFrame, 测试集整体模型效果数据
        test_channel_df: DataFrame, 测试集分渠道模型效果数据
        psi_value: float, 模型PSI值
        lift_res: dict, Lift分析结果
        output_path: str, 输出路径
        train_date_range: str, 训练集时间范围
        test_date_range: str, 测试集时间范围
        date_col: str, 日期列名
    """
    try:
        # 1. 模型效果评估
        model_performance = {
            "模型效果评估": []
        }
        
        # 处理训练集数据
        train_row = train_overall_df.iloc[0]
        train_performance = {
            "数据集": "训练集",
            "时间范围": train_date_range,
            "样本总数": int(train_row['total']),
            "好样本": int(train_row['good']),
            "坏样本": int(train_row['bad']),
            "坏率": f"{train_row['bad_ratio']:.2%}",
            "AUC": f"{train_row['auc']:.6f}",
            "KS": f"{train_row['ks']:.6f}"
        }
        model_performance["模型效果评估"].append(train_performance)
        
        # 处理测试集数据
        test_row = test_overall_df.iloc[0]
        test_performance = {
            "数据集": "测试集",
            "时间范围": test_date_range,
            "样本总数": int(test_row['total']),
            "好样本": int(test_row['good']),
            "坏样本": int(test_row['bad']),
            "坏率": f"{test_row['bad_ratio']:.2%}",
            "AUC": f"{test_row['auc']:.6f}",
            "KS": f"{test_row['ks']:.6f}"
        }
        model_performance["模型效果评估"].append(test_performance)
        
        # 2. 模型稳定性评估
        stability = {
            "模型稳定性评估": {
                "模型PSI值": f"{psi_value:.4f}",
                "稳定性评估": "模型稳定性良好" if psi_value < 0.1 else "模型稳定性较差"
            }
        }
        
        # 3. 分渠道模型效果
        channel_performance = {
            "分渠道模型效果": []
        }
        
        for _, row in test_channel_df.iterrows():
            channel = {
                "渠道": row['col'],
                "样本总数": int(row['total']),
                "好样本": int(row['good']),
                "坏样本": int(row['bad']),
                "坏率": f"{row['bad_ratio']:.2%}",
                "AUC": f"{row['auc']:.4f}",
                "KS": f"{row['ks']:.4f}"
            }
            channel_performance["分渠道模型效果"].append(channel)

        # 4. Lift分析结果
        lift_df = lift_res['lift_top_20']
        total_bins = len(lift_df)
        
        # 根据实际分箱数量计算百分比
        # 注意：由于duplicates='drop'，实际分箱数量可能少于预期
        if len(lift_df) > 0:
            # 计算第一箱的实际样本比例
            first_bin_ratio = lift_df.iloc[0]['acc_num_ration']  # 累积样本比例
            first_bin_pct = round(first_bin_ratio * 100, 1)
        else:
            first_bin_pct = 0
            
        if len(lift_df) > 1:
            # 计算前两箱的实际样本比例
            second_bin_ratio = lift_df.iloc[1]['acc_num_ration']  # 累积样本比例
            second_bin_pct = round(second_bin_ratio * 100, 1)
        else:
            second_bin_pct = first_bin_pct
        
        
        # 确保有足够的分箱数据
        first_lift = lift_df.iloc[0]['lift'] if len(lift_df) > 0 else 0
        second_lift = lift_df.iloc[1]['lift'] if len(lift_df) > 1 else 0
        lift_analysis = {
            "Lift分析结果": {
                "Lift分析概览": {
                    f"前{first_bin_pct}%样本的Lift值": f"{first_lift:.2f}",
                    f"前{second_bin_pct}%样本的Lift值": f"{second_lift:.2f}"
                },
                "分箱Lift分析": []
            }
        }
        
        # 获取前10箱的Lift分箱结果
        lift_df = lift_res['lift_top_20'].head(10)
        for _, row in lift_df.iterrows():
            lift_info = {
                "分箱": int(row['level']),  # 使用level列
                "分数区间": row['range'],  # 使用range列
                "样本数": int(row['num']),
                "好样本": int(row['good']),
                "坏样本": int(row['bad']),
                "坏率": f"{row['bad_ration']:.2%}",  # 使用bad_ration列
                "基准坏率": f"{lift_res['base_rate']:.2%}",
                "Lift": f"{row['lift']:.2f}"
            }
            lift_analysis["Lift分析结果"]["分箱Lift分析"].append(lift_info)
        
        # 整合所有结果
        report_data = {
            "report_title": "模型评估报告",
            "model_performance": model_performance,
            "stability": stability,
            "channel_performance": channel_performance,
            "lift_analysis": lift_analysis
        }
        
        # 转换为JSON
        json_str = json.dumps(report_data, ensure_ascii=False, indent=2)
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        return output_path
            
    except Exception as e:
        print(f"Error generating model evaluation JSON report: {str(e)}")
        return None

def evaluation_func(dev_df, oot_df, psi_df, config):
    """
    模型评估函数
    
    参数:
        dev_df: DataFrame, 训练集数据
        oot_df: DataFrame, 测试集数据  
        psi_df: DataFrame, PSI数据集
        config: dict, 配置参数
    """
    msg='evaluation: succ'
    try:
        print("=" * 60)
        print("开始模型评估流程")
         ## 基本参数导入
        id_col = config['settings']['id_col']
        date_col = config['settings']['date_col']
        label = config['settings']['label']
        channel_col = config['settings']['channel_col']
        base_fea = config['settings']['base_fea']
        bin_num = config['settings']['bin_num']     
        OUT_REPORT = config['settings']['gen_report']
        OUTPUT = config['settings']['save_output']
        IF_RUN = config['settings']['run_eval']

        if IF_RUN:
            ## 导入模型
            with open(config['model_file']['pkl_model_path'],'rb') as f:
                mod = pickle.load(f)
            with open(config['model_file']['model_fea_path'], encoding='utf-8') as f:
                mod_var = json.load(f)
            print('步骤1/8: 模型导入完成')

            ## 模型分数转换参数
            pdo = config['score']['pdo']
            theta = config['score']['theta']
            p0 = config['score']['p0']
            cut_points = config['score']['cut_points']
            prob_col = config['score']['prob_col']
            score_col = config['score']['score_col']

            ## 输出路径
            dst_output = config['output_path']['dst_outs']
            dst_report = config['output_path']['dst_rp']
            dst_data = config['output_path']['dst_data']
            os.makedirs(dst_output, exist_ok=True)
            os.makedirs(dst_report, exist_ok=True)
            os.makedirs(dst_data, exist_ok=True)
            eval_json_path = config['output_path']['eval_json_path']
            final_report_path = config['output_path']['model_evaluation_report_path']
            print('步骤2/8: 输出目录创建完成')

            ## 区分数值型特征和类别型特征
            num_fea, cat_fea = identify_types(dev_df[mod_var])
            for i in cat_fea:
                dev_df[i] = dev_df[i].astype('category')
                oot_df[i] = oot_df[i].astype('category')
                psi_df[i] = psi_df[i].astype('category')
            # 模型概率预测
            df_oot = model_pred(df=oot_df, mod=mod, mod_var=mod_var, prob_col=prob_col)
            df_dev = model_pred(df=dev_df, mod=mod, mod_var=mod_var, prob_col=prob_col)
            df_psi = model_pred(df=psi_df, mod=mod, mod_var=mod_var, prob_col=prob_col)

            ##======后添加的
            TEST_PRED_100_CSV = os.path.join(dst_output, 'test_pred_100.csv')
            TEST_PRED_5_JSON = os.path.join(dst_output, 'test_pred_5.json')
            USE_FEA_CSV = os.path.join(dst_output, 'use_fea_list.csv')
            # 1. 取df_oot前100行，按mod_var和prob_col输出
            df_oot_100 = df_oot.head(100).copy()
            cols = mod_var + [prob_col]
            df_oot_100[cols].to_csv(TEST_PRED_100_CSV, index=False)
            # 2. 随机抽5行保存为json
            sample_df = df_oot_100[cols].sample(n=5, random_state=42)
            sample_json = sample_df.to_dict(orient='records')
            with open(TEST_PRED_5_JSON, 'w', encoding='utf-8') as f:
                json.dump(sample_json, f, ensure_ascii=False, indent=2)
            # 3. mod_var转csv
            use_fea_df = pd.DataFrame({'Feature': mod_var})
            use_fea_df.to_csv(USE_FEA_CSV, index=False)
            print(f'已生成100条预测csv、5条json和特征csv，路径可在代码中自定义。')

            ##模型PSI
            psi_value = evaluate_psi(df_dev, df_psi, pred_col=prob_col)
            print(f"         模型PSI: {psi_value:.4f}")
            print('步骤3/8: 模型稳定性(PSI)评估完成')

            ### 7.04 KS 结果计算 + 7.05 AUC 结果计算
            # 训练集评估
            train_temp = df_dev[base_fea+[prob_col]]
            train_temp['overall'] = 'overall'
            train_overall_df, train_channel_df = get_final_result('dev', train_temp, target=label, pred=prob_col, col=channel_col, dst=dst_output) 
            # 测试集评估
            test_temp = df_oot[base_fea+[prob_col]]
            test_temp['overall'] = 'overall'
            test_overall_df, test_channel_df = get_final_result('oot', test_temp, target=label, pred=prob_col, col=channel_col, dst=dst_output)
            print('步骤4/8: 模型性能(KS/AUC)评估完成')
            print(f"         训练集: AUC={train_overall_df.iloc[0]['auc']:.6f}, KS={train_overall_df.iloc[0]['ks']:.6f}")
            print(f"         测试集: AUC={test_overall_df.iloc[0]['auc']:.6f}, KS={test_overall_df.iloc[0]['ks']:.6f}")

            #### 7.06 Lift 结果计算
            lift_res = lift_analysis(test_temp, label_col=label, pred_col=prob_col, bin_num=bin_num)
            lift_res['lift_top_20'].to_csv(os.path.join(dst_output, f'bin{bin_num}_lift_res.csv'), index=False)
            print('步骤5/8: Lift分析完成')

            # 生成模型评估JSON报告
            # 确保日期列为datetime类型
            df_dev[date_col] = pd.to_datetime(df_dev[date_col], format='%Y%m%d', errors='coerce')
            df_oot[date_col] = pd.to_datetime(df_oot[date_col], format='%Y%m%d', errors='coerce')
            
            # 获取训练集和测试集的时间范围
            train_date_range = f"{df_dev[date_col].min().strftime('%Y%m%d')}-{df_dev[date_col].max().strftime('%Y%m%d')}"
            test_date_range = f"{df_oot[date_col].min().strftime('%Y%m%d')}-{df_oot[date_col].max().strftime('%Y%m%d')}"
            generate_evaluation_json(
                train_overall_df=train_overall_df,
                test_overall_df=test_overall_df,
                test_channel_df=test_channel_df,
                psi_value=psi_value,
                lift_res=lift_res,
                output_path=eval_json_path,
                train_date_range=train_date_range,
                test_date_range=test_date_range,
                date_col=date_col
            )
            print('步骤6/8: 评估报告JSON生成完成')
            
            # 模型分数转换
            df_dev[score_col] = df_dev[prob_col].apply(lambda x:Transfer2Score(x,p0=p0,PDO=pdo,theta=theta))
            df_oot[score_col] = df_oot[prob_col].apply(lambda x:Transfer2Score(x,p0=p0,PDO=pdo,theta=theta))
            df_psi[score_col] = df_psi[prob_col].apply(lambda x:Transfer2Score(x,p0=p0,PDO=pdo,theta=theta))
            
            # 保存包含基础特征、入模特征、预测概率和分数的数据集，用于复现结果
            # 添加入模特征，并将日期列重命名为month
            dev_reproduce = df_dev[base_fea + mod_var + [prob_col, score_col]].copy()
            oot_reproduce = df_oot[base_fea + mod_var + [prob_col, score_col]].copy()
            psi_reproduce = df_psi[base_fea + mod_var + [prob_col, score_col]].copy()
            
            # 将日期列重命名为month，并精确到月份
            dev_reproduce['month'] = pd.to_datetime(dev_reproduce[date_col], format='%Y%m%d', errors='coerce').dt.to_period('M')
            oot_reproduce['month'] = pd.to_datetime(oot_reproduce[date_col], format='%Y%m%d', errors='coerce').dt.to_period('M')
            psi_reproduce['month'] = pd.to_datetime(psi_reproduce[date_col], format='%Y%m%d', errors='coerce').dt.to_period('M')
            
            # 保存复现用数据集
            dev_reproduce.to_csv(os.path.join(dst_data, 'df_dev_retrain.csv'), index=False)
            oot_reproduce.to_csv(os.path.join(dst_data, 'df_oot_retrain.csv'), index=False)
            psi_reproduce.to_csv(os.path.join(dst_data, 'df_psi_retrain.csv'), index=False)
            
            # 分数转换的序行结果
            df_score = get_sta_groups(df_oot[label], df_oot[score_col],cut_points)
            print('步骤7/8: 分数转换完成')
            
            ### 报告汇总
            if OUT_REPORT:
                dic_rp = config['report_all']
                # ========== 固定保持不变的文件 【数据质量|违约率|单变量分箱|lift分箱】==========
                if os.path.exists(dic_rp['data_quality_report_path']):
                    df_quality = pd.read_excel(dic_rp['data_quality_report_path']) #数据质量报告
                else:
                    df_quality = pd.DataFrame()
                if os.path.exists(dic_rp["target_stats_report_path"]):
                    df_default_dist = pd.read_csv(dic_rp["target_stats_report_path"])##违约率报告
                else:
                    df_default_dist = pd.DataFrame()
                
                mono_var_path = dic_rp['mono_var_path']
                if os.path.exists(mono_var_path):
                    df_mono_var = pd.read_csv(mono_var_path)   ##取单变量分箱结果
                else:
                    df_mono_var = pd.DataFrame()

                df_lift = lift_res['lift_top_20']
                # ========== 入模特征分析：合并特征分析结果 ==========
                single_var_path = dic_rp['single_var_path']
                if os.path.exists(single_var_path):
                    df_feat_analysis = pd.read_csv(single_var_path)
                else:
                    df_feat_analysis = pd.DataFrame()
                feature_importance_path = dic_rp['feature_importance_path']
                if os.path.exists(feature_importance_path):
                    df_importance = pd.read_csv(feature_importance_path)
                else:
                    df_importance = pd.DataFrame()
                # 合并分析和重要性（用Feature字段）
                df_feat_merged = df_importance.merge(df_feat_analysis, on='Feature', how='left')
                
                # ========== 模型效果合并 ==========
                df_perf_overall = pd.concat([train_overall_df, test_overall_df], ignore_index=True)
                df_perf_channels = pd.concat([train_channel_df, test_channel_df], ignore_index=True)
                df_model_effect_all = pd.concat([df_perf_overall, df_perf_channels], ignore_index=True)
                # ========== 汇总写入 Excel ==========
                with pd.ExcelWriter(final_report_path,engine='openpyxl') as writer:
                    df_quality.to_excel(writer, sheet_name="01_数据质量报告", index=False)
                    df_default_dist.to_excel(writer, sheet_name="02_违约率分布", index=False)
                    df_feat_merged.to_excel(writer, sheet_name="03_入模特征分析", index=False)
                    df_mono_var.to_excel(writer, sheet_name="04_单变量分箱结果", index=False)
                    df_model_effect_all.to_excel(writer, sheet_name="05_模型效果", index=False)
                    df_lift.to_excel(writer, sheet_name="06_Lift分箱", index=False)
                    
                    # 创建分数转换参数说明DataFrame
                    score_params_df = pd.DataFrame([
                        ['分数转换参数说明', ''],
                        ['基准分数点(p0)', p0],
                        ['分数翻倍所需的好样本比例(pdo)', pdo],
                        ['基准好坏比(theta)', theta],
                        ['分数切点', str(cut_points)],
                        ['', ''],
                        ['分数转换结果', '']
                    ])
                    score_params_df.to_excel(writer, sheet_name="07_分数转换", index=False, header=False)
                    # 获取当前sheet的writer对象
                    workbook = writer.book
                    worksheet = writer.sheets["07_分数转换"]
                    # 找到下一个空行
                    startrow = score_params_df.shape[0]
                    # 写df_score，保留表头
                    df_score.to_excel(writer, sheet_name="07_分数转换", index=False, header=True, startrow=startrow)      
                print('步骤8/8: 模型报告生成完成')
                 # 步骤9/9: 自动生成并执行串行验证
                print('步骤9/9: 开始自动生成并执行串行验证...')
                try:
                    from .retrain_code_generator import auto_generate_and_execute_retrain
                    with open('outputs/model_train/outputs/model_params.json', 'r', encoding='utf-8') as f:
                        model_params = json.load(f)
                    dev_data_path = dst_data+'/df_dev_retrain.csv'
                    oot_data_path = dst_data+'/df_oot_retrain.csv'
                    psi_data_path = dst_data+'/df_psi_retrain.csv'
                    print(dst_output)
                    output_path = dst_output+'/retrain_validationRes.xlsx'
                    print(output_path)
                    success = auto_generate_and_execute_retrain(
                        dev_data_path=dev_data_path,
                        oot_data_path=oot_data_path,
                        psi_data_path=psi_data_path,
                        use_features=mod_var,
                        model_params=model_params,
                        p0=p0,
                        pdo=pdo,
                        theta=theta,
                        label=label,
                        output_path=output_path,
                        output_dir=dst_output
                    )
                    if success:
                        print("✓ 串行验证自动生成并执行成功!")
                    else:
                        print("✗ 串行验证自动生成并执行失败!")
                except Exception as e:
                    print(f"自动生成并执行串行验证时发生错误: {str(e)}")
                    print(traceback.format_exc())
            return True
        return False
    except:
        msg=traceback.format_exc()
        print(msg)
        return False



