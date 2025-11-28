#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动生成retrain_code.py文件的代码生成器
每次执行模型评估时自动生成最新的串行验证代码
"""

import os
from datetime import datetime
    
def generate_retrain_code(
    dev_data_path,
    oot_data_path,
    psi_data_path,
    use_features,
    model_params,
    p0,
    pdo,
    theta,
    label,
    output_path,
    output_dir
):
    """生成代码内容"""
    code_template = f'''###=========================导入必要的包======================================
import sys
import pandas as pd
import numpy as np
import datetime
import time as time
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns',70)
pd.set_option('display.max_rows',500)
pd.set_option('display.float_format',lambda x : '%0.5f' %x)
pd.set_option('display.max_colwidth',1000)
np.set_printoptions(suppress=True , precision=20,threshold =10 , linewidth =40)
import warnings 
warnings.filterwarnings('ignore')
# import joblib
# import pickle
from copy import deepcopy
import lightgbm as lgb
from sklearn import  metrics
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import roc_auc_score,roc_curve

###=========================function======================================
def ks_metric(y_true,y_pred):
    return 'ks',ks_score(y_true,y_pred),True

def ks_score(true,preds):
    fpr,tpr,thre=roc_curve(true,preds,pos_label=1)
    return abs(fpr-tpr).max()

def ks_score2(true,preds,typ='score',clas='dev'):
    fpr,tpr,thre=roc_curve(true,preds,pos_label=1)
    ks=abs(fpr-tpr).max()
    if typ=='score':
        auc_train=1-roc_auc_score(true,preds)
    else:
        auc_train=roc_auc_score(true,preds)
    num=len(true)
    num1 = sum(true)
    dd=[[clas ,num,num1,round(num1/num,4),auc_train,ks]]
    dd=pd.DataFrame(dd)
    dd.columns=['类别','cnt','bad_cnt','bad_ratio','auc','ks']
    return dd

# flag=1:等频, flag=2:等距
def feature_psi(dtrain,dtest,ft,flag=1, bin_step=0.05, group=20):    
    max_n = max(dtrain[ft].max(),dtest[ft].max()) + 0.1
    min_n = min(dtrain[dtrain[ft]>-1][ft].min(), dtest[dtest[ft]>-1][ft].min())
    bins_split=[-10000, min_n]
    if flag==1:
        perc = np.arange(0,1,bin_step)
        node = list(dtrain[dtrain[ft]>-1][ft].quantile(perc))
        node = node[1:]
        bins_split = list(bins_split + list(np.unique(node)))
        bins_split.append(max_n + 1)   
    if flag==2:
        step_n = (max_n-min_n)/group
        for i in range(group):
            tmp = min_n + i * step_n
            bins_split.append(tmp)
        tmp_max = max(min_n + group*step_n, max_n)
        bins_split.append(tmp_max)
    train_prob_ = dtrain[ft]
    test_prob_ = dtest[ft] 
    cutbin_ = list(np.unique(bins_split))
    train_probbin_ = pd.concat([pd.cut(train_prob_,cutbin_,right=False,duplicates='drop'),dtrain[ft]],axis=1)
    test_probbin_ = pd.concat([pd.cut(test_prob_,cutbin_,right=False,duplicates='drop'),dtest[ft]],axis=1)
    bin_index = 'bin_'+ft
    train_probbin_.columns=[bin_index,'train']
    train_probbin_.train.fillna(0,inplace=True)
    test_probbin_.columns=[bin_index,'test']
    test_probbin_.test.fillna(0,inplace=True)
    tr_count_ = train_probbin_.groupby(bin_index).count()
    te_count_ = test_probbin_.groupby(bin_index).count()
    _psi_ = pd.concat([tr_count_,te_count_],axis=1)
    train_sum = sum(_psi_['train'])
    test_sum = sum(_psi_['test'])
    _psi_['train_prec']=_psi_.apply(lambda x : np.round(x.train/train_sum,8) if x.train >0 else  np.round((x.train+1)/train_sum,8), axis=1)
    _psi_['test_prec']=_psi_.apply(lambda x : np.round(x.test/test_sum,8) if x.test >0 else  np.round((x.test+1)/test_sum,8), axis=1 )
    _psi_['psi']=(_psi_['train_prec']-_psi_['test_prec'])*(_psi_['train_prec']/_psi_['test_prec']).apply(lambda x:math.log(x,math.e))
    _psi_['psi'] = np.round(_psi_['psi'], 4)
    _psi_['psi'].replace(to_replace=np.inf,value=0,inplace=True)
    _psi_.psi.fillna(0,inplace=True)
    #print(ft +' The PSI is: %f' % sum(_psi_['psi']))
    _psi_.reset_index(inplace=True)
    #return sum(_psi_['psi'])
    return _psi_

def get_distr(dt, ft, group=10):
    
    max_n=dt[ft].max()
    min_n=dt[dt[ft]>-1][ft].min()
    
    bin_split=[round(min_n-0.0001,4)]
    
    step_n=(max_n-min_n)/group
    for i in range (group):
        temp=round(min_n+step_n+i*step_n,4)
        bin_split.append(temp)
    # temp_max=round(max(min_n+step_n*group, max_n),4)+0.0001
    # bin_split.append(temp_max)
    
    bin_split=list(np.unique(bin_split))
    bin_split.sort()
    bin_split[-1] = 1000.0
    bin_split[0] = 0.0
    binned=pd.cut(dt[ft], bins=bin_split)
    ft_binned='binned'+'_'+ft
    dt[ft_binned]=binned

    return dt, bin_split
def Transfer2Score(prob=0,p0=600,PDO=50,theta=20):
    import math
    if prob==1:
        p=0.00000001
    elif prob==0:
        p=0.99999999
    else:
        p=1-prob
    B=PDO/(math.log(0.5,math.e))
    A=p0+B*(math.log(theta,math.e))
    odds=p/(1-p)   
    score = round(A-B*(math.log(odds,math.e)))
    return max(0,min(1000,score))

###=========================Config Set======================================

#1/5: 保持和离线模型一致
base_point={p0}  # 从model_eval_config.json中的p0
theta={theta}       # 从model_eval_config.json中的theta
PDO={pdo}          # 从model_eval_config.json中的pdo

###########重要！！！！需要根据各模型情况，手动修改模型名称，业务背景及算法##########################
#2/5: 模型名称，业务背景及算法#
model_name='demomob3-retrain'
yw_bj='测试重训练代码'
model_alg='LGBM'

#==============offline debug=======================
#3/5: 串行验证数据路径【上传时修改为入参形式】
dev_data_path='{dev_data_path}'
oot_data_path='{oot_data_path}'
psi_data_path='{psi_data_path}'
output_path='{output_path}'

#dev_data_path=sys.argv[1]
#oot_data_path=sys.argv[2]
#psi_data_path=sys.argv[3]
#output_path=sys.argv[4]

dev_data = pd.read_csv(dev_data_path)
oot_data = pd.read_csv(oot_data_path)
psi_data = pd.read_csv(psi_data_path)

###########重要！！！！需要根据各模型Y标签，手动修改##########################
Target='{label}'

##########重要！！！！需要根据各模型候选特征，手动修改##########################
#4/5: 入模特征
var_select_last = {use_features}
print(f"入模特征数量: {{len(var_select_last)}}")
print("入模特征列表:", var_select_last)

###########重要！！！！需要根据各模型训练参数，手动修改##########################
#5/5:模型参数
param = {model_params}



# #============识别类别型特征并转换为category类型（与训练时保持一致）===========
cat_fea = []
for feat in var_select_last:
    if dev_data[feat].dtype == 'object':
        cat_fea.append(feat)
        # 对原始数据也转换（用于后续预测）
        dev_data[feat] = dev_data[feat].astype('category')
        oot_data[feat] = oot_data[feat].astype('category')
        psi_data[feat] = psi_data[feat].astype('category')
if len(cat_fea) > 0:
    print(f"类别型特征数量: {{len(cat_fea)}}")
    print(f"类别型特征: {{cat_fea}}")
else:
    print("未发现类别型特征")

dev=dev_data[dev_data[Target].isin([0,1])]
oot=oot_data[oot_data[Target].isin([0,1])]
psi=psi_data[psi_data[Target].isin([0,1])]
print(dev.shape)
print(oot.shape)
print(psi.shape)

# #=========================================predict=======================================
clf = lgb.LGBMClassifier(**param)
model_lgb =deepcopy(clf) 
model_lgb.fit(dev[var_select_last],dev[Target], categorical_feature=cat_fea)

lgb_pred_train=model_lgb.predict_proba(dev[var_select_last])[:,1]
lgb_auc_train=roc_auc_score(dev[Target],lgb_pred_train)
lgb_ks_train=ks_score(dev[Target],lgb_pred_train)
print('--------------------------------')
print('lgb auc_train : {{}}'.format(lgb_auc_train))
print('lgb ks_train : {{}}'.format(lgb_ks_train))
print('--------------------------------')

lgb_pred_oot=model_lgb.predict_proba(oot[var_select_last])[:,1]
lgb_auc_oot=roc_auc_score(oot[Target],lgb_pred_oot)
lgb_ks_oot=ks_score(oot[Target],lgb_pred_oot)
print('--------------------------------')
print('lgb auc_oot : {{}}'.format(lgb_auc_oot))
print('lgb ks_oot : {{}}'.format(lgb_ks_oot))
print('--------------------------------')

########################################################################### 模型评估
################ 变量重要性
model=model_lgb
imp_df = pd.DataFrame()
imp_df["Feature"] = model_lgb.booster_.feature_name()
#imp_df["importance_gain"] = alg.booster_.feature_importance(importance_type='gain')
imp_df["importance_split"] = model_lgb.booster_.feature_importance(importance_type='split')
#imp_df['gain_ratio']=imp_df['importance_gain']/sum(imp_df['importance_gain'])
imp_df['split_ratio']=imp_df['importance_split']/imp_df['importance_split'].sum()
imp_df =imp_df.sort_values(by='importance_split',ascending=False)

var_importance=imp_df[imp_df['split_ratio']>0][['Feature','split_ratio']]
var_importance.rename(columns={{'split_ratio': 'importance'}}, inplace=True)

################ 概率预测及分数转换
dev['prob']=model_lgb.predict_proba(dev[var_select_last])[:,1] 
oot['prob']=model_lgb.predict_proba(oot[var_select_last])[:,1] 
dev_data['prob']=model_lgb.predict_proba(dev_data[var_select_last])[:,1] 
oot_data['prob']=model_lgb.predict_proba(oot_data[var_select_last])[:,1] 
psi_data['prob']=model_lgb.predict_proba(psi_data[var_select_last])[:,1] 

###########重要！！！！需要根据各模型转分参数，手动修改##########################

dev['score'] =dev['prob'].apply(lambda x : Transfer2Score(prob=x,p0=base_point,PDO=PDO,theta=theta)) 
oot['score'] =oot['prob'].apply(lambda x : Transfer2Score(prob=x,p0=base_point,PDO=PDO,theta=theta)) 
dev_data['score'] =dev_data['prob'].apply(lambda x : Transfer2Score(prob=x,p0=base_point,PDO=PDO,theta=theta)) 
oot_data['score'] =oot_data['prob'].apply(lambda x : Transfer2Score(prob=x,p0=base_point,PDO=PDO,theta=theta)) 
psi_data['score'] =psi_data['prob'].apply(lambda x : Transfer2Score(prob=x,p0=base_point,PDO=PDO,theta=theta)) 

############################### 模型效果评估 

dev_prob = ks_score2(dev[Target],dev['prob'],typ='prob',clas='dev')
oot_prob = ks_score2(oot[Target],oot['prob'],typ='prob',clas='oot')

dev_score = ks_score2(dev[Target],dev['score'],typ='score',clas='dev')
oot_score = ks_score2(oot[Target],oot['score'],typ='score',clas='oot')

################ 模型稳定性
import math
psi_dev_oot_prob= feature_psi(dev_data,oot_data,'prob',flag=1, bin_step=0.1, group=10)
print(f"Dev-OOT概率PSI: {{psi_dev_oot_prob.psi.sum():.4f}}")
psi_dev_psi_prob= feature_psi(dev_data,psi_data,'prob',flag=1, bin_step=0.1, group=10)
print(f"Dev-PSI概率PSI: {{psi_dev_psi_prob.psi.sum():.4f}}")

psi_dev_oot_score= feature_psi(dev_data,oot_data,'score',flag=1, bin_step=0.1, group=10)
print(f"Dev-OOT分数PSI: {{psi_dev_oot_score.psi.sum():.4f}}")
psi_dev_psi_score= feature_psi(dev_data,psi_data,'score',flag=1, bin_step=0.1, group=10)
print(f"Dev-PSI分数PSI: {{psi_dev_psi_score.psi.sum():.4f}}")


model_result=pd.DataFrame({{'模型名称':model_name,'业务背景':yw_bj,'模型算法':model_alg,'概率KS_dev':dev_prob['ks'],'概率KS_oot':oot_prob['ks'],'概率AUC_dev':dev_prob['auc'],'概率AUC_oot':oot_prob['auc'],'概率PSI_dev_oot':psi_dev_oot_prob.psi.sum(),\
           '概率PSI_dev_psi':psi_dev_psi_prob.psi.sum(),'分数KS_dev':dev_score['ks'],'分数KS_oot':oot_score['ks'],\
           '分数AUC_dev':dev_score['auc'],'分数AUC_oot':oot_score['auc'],'分数PSI_dev_oot':psi_dev_oot_score.psi.sum(),'分数PSI_dev_psi':psi_dev_psi_score.psi.sum()}})

################ 评分单调性
#违约率分布-score

data_all, bin_split_all=get_distr(dev_data, 'score',11)
dev_data.pivot_table(index='binned_score',values='score',columns=Target,aggfunc=np.count_nonzero,margins=True)
bin_manual=bin_split_all

ft='score'
ft_binned='binned'+'_'+ft
target=Target

dt=dev_data
dt[ft_binned]=pd.cut(dt[ft],bins=bin_manual)
tmp=dt.pivot_table(index=ft_binned,values=ft,columns=Target,aggfunc=np.count_nonzero,margins=True)
tmp=pd.DataFrame(tmp)
tmp=tmp.fillna(0)
tmp.rename(columns={{0.0: 'dev_0',1.0: 'dev_1',2.0: 'dev_2','All':'dev_all_gbi'}}, inplace=True)
result=tmp

dt=oot_data
dt[ft_binned]=pd.cut(dt[ft],bins=bin_manual)
tmp=dt.pivot_table(index=ft_binned,values=ft,columns=Target,aggfunc=np.count_nonzero,margins=True)
tmp=pd.DataFrame(tmp)
tmp=tmp.fillna(0)
tmp.rename(columns={{0.0: 'oot_0',1.0: 'oot_1',2.0: 'oot_2','All':'oot_all_gbi'}}, inplace=True)
result=result.merge(tmp, how='left',on=ft_binned)

dt=psi_data
dt[ft_binned]=pd.cut(dt[ft],bins=bin_manual)
tmp=dt.pivot_table(index=ft_binned,values=ft,columns=Target,aggfunc=np.count_nonzero,margins=True)
tmp=pd.DataFrame(tmp)
tmp=tmp.fillna(0)
tmp.rename(columns={{0.0: 'psi_0',1.0: 'psi_1',2.0: 'psi_2','All':'psi_all'}}, inplace=True)
result=result.merge(tmp, how='left',on=ft_binned)

result['dev_all']=result['dev_0']+result['dev_1']
result['oot_all']=result['oot_0']+result['oot_1']


result=result.reset_index()

dev_sum = sum(result[result['binned_score']!='All']['dev_all'])
oot_sum = sum(result[result['binned_score']!='All']['oot_all'])
psi_sum = sum(result[result['binned_score']!='All']['psi_all'])
print(f"Dev样本总数: {{dev_sum}}")
print(f"OOT样本总数: {{oot_sum}}")
print(f"PSI样本总数: {{psi_sum}}")

result['Total_pcnt_dev']=result.apply(lambda x : np.round(x['dev_all']/dev_sum,8) ,axis=1)
result['Total_pcnt_oot']=result.apply(lambda x : np.round(x['oot_all']/oot_sum,8), axis=1)
result['Total_pcnt_psi']=result.apply(lambda x : np.round(x['psi_all']/psi_sum,8) , axis=1)


result['Bad_rate_dev']=result.apply(lambda x : np.round(x['dev_1']/x['dev_all'],8) if x['dev_all'] >0 else  0.0, axis=1)
result['Bad_rate_oot']=result.apply(lambda x : np.round(x['oot_1']/x['oot_all'],8) if x['oot_all'] >0 else  0.0, axis=1)
result['all']=result['dev_0']+result['dev_1']+result['oot_0']+result['oot_1']+result['psi_all']
print(f"Dev好样本总数: {{result[result['binned_score']!='All']['dev_0'].sum()}}")
print(f"Dev坏样本总数: {{result[result['binned_score']!='All']['dev_1'].sum()}}")
print(f"OOT好样本总数: {{result[result['binned_score']!='All']['oot_0'].sum()}}")
print(f"OOT坏样本总数: {{result[result['binned_score']!='All']['oot_1'].sum()}}")
print(f"PSI好样本总数: {{result[result['binned_score']!='All']['psi_0'].sum()}}")
print(f"PSI坏样本总数: {{result[result['binned_score']!='All']['psi_1'].sum()}}")

result_score=result
result_score.rename(columns={{'binned_score':'bin','psi_all':'psi'}},inplace=True)
result_score=result_score[result_score.bin!='All']
result_score=result_score[['bin','dev_0','dev_1','oot_0','oot_1','psi','all','Total_pcnt_dev','Bad_rate_dev','Total_pcnt_oot','Bad_rate_oot','Total_pcnt_psi']]

result_score=result_score[['bin','dev_0','dev_1','oot_0','oot_1','psi','all','Total_pcnt_dev','Bad_rate_dev','Total_pcnt_oot','Bad_rate_oot','Total_pcnt_psi']]

################## 保存结果到excel
with pd.ExcelWriter(output_path,engine='openpyxl') as writer: #,model='a'
    model_result=pd.DataFrame(model_result)
    var_importance=pd.DataFrame(var_importance)
    #result_prob=pd.DataFrame(result_prob)
    result_score=pd.DataFrame(result_score)
    
    model_result.to_excel(writer,index=False,sheet_name="固定内容",header=True)
    var_importance.to_excel(writer,index=False,sheet_name="变量分析",header=True)
    #result_prob.to_excel(writer,index=False,sheet_name="概率单调性",header=True)    
    result_score.to_excel(writer,index=False,sheet_name="评分单调性",header=True)  
print(f"串行验证完成，结果已保存到: {{output_path}}")
'''
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'retrain_code.py')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code_template)
    print(f"串行验证代码已生成: {filepath}")
    return filepath

def auto_generate_and_execute_retrain(
    dev_data_path,
    oot_data_path,
    psi_data_path,
    use_features,
    model_params,
    p0,
    pdo,
    theta,
    label,
    output_path,
    output_dir
):
    """
    自动生成并执行串行验证代码
    参数:
        dev_data_path: 训练集路径
        oot_data_path: 测试集路径
        psi_data_path: PSI集路径
        use_features: 入模特征列表
        model_params: 模型参数字典
        p0, pdo, theta: 分数转换参数
        label: 标签名
        output_path: 串行验证结果输出路径
        output_dir: 输出目录
    返回:
        bool: 执行是否成功
    """
    try:
        retrain_file_path = generate_retrain_code(
            dev_data_path,
            oot_data_path,
            psi_data_path,
            use_features,
            model_params,
            p0,
            pdo,
            theta,
            label,
            output_path,
            output_dir
        )
        import subprocess
        import sys
        print(f"开始执行串行验证: {retrain_file_path}")
        output_path
        result = subprocess.run([sys.executable, retrain_file_path], 
                              capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print("✓ 串行验证执行成功!")
            print("执行输出:")
            print(result.stdout)
            return True
        else:
            print("✗ 串行验证执行失败!")
            print("错误信息:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"自动生成和执行串行验证时发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False