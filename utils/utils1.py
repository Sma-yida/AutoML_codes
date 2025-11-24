############################
#### luoshichao ############
#### lsc_utils  ############
############################

import numpy as np
import pandas as pd
import math
#from pandas.api.types import is_numeric_dtype
import time
import gc
import os
from contextlib import contextmanager

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
#import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import roc_auc_score,roc_curve
import seaborn as sns

#import sys
#sys.path.append('E:/jdxb/utils/')
#from pyce1 import *
#from woe_iv import *

def reduce_mem_usage(df,verbose=True):
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if (c_min > np.iinfo(np.int8).min
                        and c_max < np.iinfo(np.int8).max):
                    df[col] = df[col].astype(np.int8)
                elif (c_min > np.iinfo(np.int16).min
                      and c_max < np.iinfo(np.int16).max):
                    df[col] = df[col].astype(np.int16)
                elif (c_min > np.iinfo(np.int32).min
                      and c_max < np.iinfo(np.int32).max):
                    df[col] = df[col].astype(np.int32)
                elif (c_min > np.iinfo(np.int64).min
                      and c_max < np.iinfo(np.int64).max):
                    df[col] = df[col].astype(np.int64)
            else:
                if (c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max):
                    df[col] = df[col].astype(np.float16)
                elif (c_min > np.finfo(np.float32).min
                      and c_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    reduction = (start_mem - end_mem) / start_mem

    msg = f'Mem. usage decreased to {end_mem:5.2f} MB ({reduction * 100:.1f} % reduction)'
    if verbose:
        print(msg)

    return df

@contextmanager
def timer(title, new_line=True):
    """
    USAGE:
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how="left", on="SK_ID_CURR")
        del bureau
        gc.collect()
    """
    t0 = time.time()
    yield
    print(f"{title} - done in {time.time() - t0:.0f}s")
    if new_line:
        print()

#-------------------------识别类型----------------------------------------------------
def Identify_types(data,remove_fea=[],show_print=False):
    if show_print:
        print('=='*20+'\n')
        print(data.dtypes.value_counts())
        print('=='*20)
    cat_fea=data.select_dtypes(include='object').columns.tolist()
    num_fea=[i for i in data.columns.tolist() if i not in list(set(cat_fea+remove_fea))]
    return num_fea,cat_fea

#num_fea,cat_fea=Identify_types(data,remove_fea=[],show_print=True)



#-------------------------基本过滤(missing,unique,std)--------------------------------------
def missing(data,features):
    total=data[features].isnull().sum().sort_values(ascending=False)
    percent=total/len(data)
    missing_data=pd.concat([total,percent],axis=1,keys=['Total','Percent'])
    missing_data=missing_data.reset_index()
    missing_data.columns=['feature','missing_num','missing_rate']
    return missing_data

#missing_data=missing(data,features=num_fea)
#miss_drop=list(missing_data[missing_data.Percent>0.85].index)


def uni(data,features):
    temp=[data[item].nunique()  for item in features]
    feature_unique=pd.DataFrame()
    feature_unique['feature']=features
    feature_unique['unique']=temp
    feature_unique=feature_unique.sort_values(by='unique',ascending=False)
    feature_unique.reset_index(drop=True,inplace=True)
    return feature_unique

#feature_unique=uni(data,features=num_fea)
#unique_drop=list(feature_unique.loc[feature_unique['unique']==1,'feature'])


def std(data,features):
    temp=[data[item].std()  for item in features]
    feature_std=pd.DataFrame()
    feature_std['feature']=features
    feature_std['std']=temp
    feature_std=feature_std.sort_values(by='std',ascending=False)
    feature_std.reset_index(drop=True,inplace=True)
    return feature_std

#feature_std=std(data,features=num_fea)
#std_drop=list(feature_std.loc[feature_std['std']==0,'feature'])

def top_values_cols(data,cat_fea,threshold=0.9):
    big_top_value_cols=[col for col in cat_fea if data[col].value_counts(dropna=False, normalize=True).values[0] > threshold]
    return big_top_value_cols




#-------------------------相关性过滤----------------------------------------------------
#corr_matrix=data[num_fea].corr()
def corr(corr_matrix,correlation_threshold):
    # Extract the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

    # Select the features with correlations above the threshold
    # Need to use the absolute value
    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]
    return to_drop
    
#corr_drop=corr(corr_matrix,correlation_threshold=0.95)
#print(corr_drop)

#-------------------------相关性矩阵可视化----------------------------------------------
def plot_corr(corr):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Compute the correlation matrix
    
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 12))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, annot=True, fmt=".4g",
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    ax.set_ylim(corr.shape[0], 0)
    plt.yticks(rotation=0)

#plot_corr(corr)

#-------------------------交叉表分析-----------------------------------------
def Crosstab_1D(data,i='app_month',j='mob3_type',GREY=True,SAVE=True,SAVE_PATH='tmp/app_mob3.csv'):
    tmp=pd.crosstab(data[i],data[j],margins=True)
    if GREY:
        tmp.columns=['good','bad','grey','total']
    else:
        tmp.columns=['good','bad','total']
    tmp.reset_index(inplace=True)
    tmp['bad_ratio']=tmp['bad']/tmp['total']
    if SAVE:
        tmp.to_csv(SAVE_PATH,index=None)
    return tmp

def Crosstab_2D(data,i=['cust_seg2','app_month'],j='mob3_type',GREY=True,SAVE=True,SAVE_PATH='tmp/app_mob3.csv'):
    tmp=pd.crosstab([data[i[0]],data[i[1]]],data[j],margins=True)
    if GREY:
        tmp.columns=['good','bad','grey','total']
    else:
        tmp.columns=['good','bad','total']
    tmp.reset_index(inplace=True)
    tmp['bad_ratio']=tmp['bad']/tmp['total']
    if SAVE:
        tmp.to_csv(SAVE_PATH,index=None)
    return tmp 


def Crosstab_3D(data,i=['channel','product_name','app_month'],j='mob3_type',GREY=True,SAVE=True,SAVE_PATH='tmp/app_mob3.csv'):
    tmp=pd.crosstab([data[i[0]],data[i[1]],data[i[2]]],data[j],margins=True)
    if GREY:
        tmp.columns=['good','bad','grey','total']
    else:
        tmp.columns=['good','bad','total']
    tmp.reset_index(inplace=True)
    tmp['bad_ratio']=tmp['bad']/tmp['total']
    if SAVE:
        tmp.to_csv(SAVE_PATH,index=None)
    return tmp
    
    
'''    
if not os.path.exists('tmp'):
        os.mkdir('tmp')
        
    
# ###    Vars Profiling，Recoding & Split   ###
#############################################
recoding_prefix='r_'
out_feature_recoding="tmp/features_recoding.txt"        # output python recoding script to file
out_features_profile="tmp/features_profile.csv"         # output feature profiling to file
out_features_statistics="tmp/features_statistics.csv"   # output feature statistics to file
woe_only = True

df_profile,df_statistics,statement_recoding = features_prof_recode(
     Xcont=train[use_fea], # set as pd.DataFrame() if none
     Xnomi=pd.DataFrame(), # set as pd.DataFrame() if none
     Y=train[label], # Y will be cut by median if non-binary target
     event=1, max_missing_rate=0.95, recoding_std=False, recoding_woe=True,  recoding_prefix=recoding_prefix,
     prof_cut_group = 10, monotonic_bin=False , prof_tree_cut=True, prof_min_p= 0.05, prof_threshold_cor=1,class_balance=True)

# use_fea=df_statistics.loc[df_statistics.iv>0.02,'variable']


### write python recoding script to file
write_recoding_txt(statement_recoding, file = out_feature_recoding, encoding = "utf-8")

### write feature profiling to file
df_profile.to_csv(out_features_profile , encoding = "gb2312")

### write feature statistics to file
df_statistics.to_csv(out_features_statistics , encoding = "gb2312")

### read in recoding script and execute recoding on model data
#data_recoded = exec_recoding(train, recoding_txt= out_feature_recoding , encoding = "utf-8")

if woe_only:
    recoding_prefix_selected = recoding_prefix + "woe_"
else:
    recoding_prefix_selected = recoding_prefix 

'''    

from pandas.core.frame import DataFrame
# import prestodb
import pandas as pd
import numpy as np

#----DataFrame 显示完全----
pd.options.display.max_columns=None   
pd.options.display.max_rows=None

#----多个命令显示---------
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity='all'

# conn=prestodb.dbapi.connect(
#     host='10.33.7.39',
#     port=8787,
#     user='luoshichao',
#     catalog='bmr',
#     schema='default',
# )
# print("====the conn is success=======")

# def get_from_hive(sql):
#     cur = conn.cursor()
#     cur.execute(sql)
#     df = DataFrame(cur.fetchall()).apply(pd.to_numeric,errors='ignore')
#     df.columns = list(pd.DataFrame(cur.description).iloc[:,0])
#     return df


### 固定随机数种子
import random
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    
seed_everything(42)


from pathlib import Path
import numpy as np
import pandas as pd
import time
import gc
import os
from contextlib import contextmanager

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
#import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import roc_auc_score,roc_curve

def ks_score(true,preds):
    fpr,tpr,thre=roc_curve(true,preds,pos_label=1)
    return abs(fpr-tpr).max()


def ks_metric(y_true,y_pred):
    return 'ks',ks_score(y_true,y_pred),True

# 2) AUC评估指标函数
def auc_score(y_true, y_pred_proba):
    """AUC评估指标函数"""
    auc_tmp = roc_auc_score(y_true, y_pred_proba)
    return auc_tmp


def Transfer2Score(prob=0,p0=600,PDO=40,theta=6.63):
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


####=============================follow is plot part=====================================


def plot_auc(y_train,pred_train,y_val,pred_val,y_oot,pred_oot):
    # 模型效果评估I：AUC（C值）
    #from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import roc_curve, auc
    import matplotlib as mpl
    mpl.style.use('ggplot')
    import matplotlib.pyplot as plt


    fpr_train,recall_train,thresholds_train=roc_curve(y_train, pred_train)
    fpr_val,recall_val,thresholds_val=roc_curve(y_val, pred_val)
    fpr_oot,recall_oot,thresholds_oot=roc_curve(y_oot, pred_oot)

    plt.rcdefaults()#重置rc所有参数，初始化
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize =(8,6))

    roc_auc_train = auc(fpr_train, recall_train)
    roc_auc_val = auc(fpr_val, recall_val)
    roc_auc_oot = auc(fpr_oot, recall_oot)
    ks_train=max(recall_train-fpr_train)
    ks_val=max(recall_val-fpr_val)
    ks_oot=max(recall_oot-fpr_oot)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_train, recall_train, 'b', label='TRAIN: C = %0.3f  ' % roc_auc_train+"KS = %0.3f"% ks_train)
    plt.plot(fpr_val, recall_val, 'r', label='VAL:     C = %0.3f  ' % roc_auc_val+"KS = %0.3f" % ks_val)
    plt.plot(fpr_oot, recall_oot, 'g', label='OOT:    C = %0.3f  ' % roc_auc_oot+"KS = %0.3f" % ks_oot)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall-out')
    plt.show()
    

def plot_ks(y_true, y_pred,name):
    """Plot KS curve.

    Parameters:
    -----------
    y_true: array_like, true binary labels

    y_pred: array_like, predicted probability estimates 

    Returns:
    --------

    """
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    ks = max(tpr - fpr)
    x = [1.0*i/len(tpr) for i in range(len(tpr))]
    
    cut_index = (tpr - fpr).argmax()
    cut_x = 1.0*cut_index/len(tpr)
    cut_tpr = tpr[cut_index]
    cut_fpr = fpr[cut_index]
    plt.rcdefaults()#重置rc所有参数，初始化
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.rc('figure', figsize=(8,6))
    plt.plot(x, tpr, color='blue', lw=2, label='True Positive Rate')
    plt.plot(x, fpr, color='green', lw=2, label='False Positive Rate')    
    plt.plot([cut_x,cut_x], [cut_tpr,cut_fpr], color='firebrick', ls='--')  
    plt.text(0.45, 0.3, 'KS = %0.3f' % ks, fontsize=14, color='firebrick')       
    plt.xlabel('Proportion')
    plt.ylabel('Rate')
    plt.title(name+' KS Curve')
    plt.legend(loc="lower right")
    plt.show()
    
def plot_lift(y_true, y_pred,name, ncut=20):
    """Plot Lift curve.

    Parameters:
    -----------
    y_true: array_like, true binary labels

    y_pred: array_like, predicted probability estimates 

    Returns:
    --------

    """
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    y_true, y_pred = pd.Series(list(y_true)), pd.Series(list(y_pred))
    try:
        qcut_x=[i+1/ncut for i in np.arange(0,1,1/ncut)]
        qcut=pd.qcut(-y_pred,q=ncut,labels=np.arange(0,1,1/ncut),retbins=False)
    except:
        qcut=pd.qcut(-y_pred,q=ncut,duplicates='drop',retbins=False)
        ncut=qcut.nunique()
        #qcut_x=[i+1/ncut for i in np.arange(0,1,1/ncut)]
        qcut_x=list(np.linspace(0,1,num=ncut,endpoint=False))
    overall_resp=np.mean(y_true)
    sample_size=[1 for i in range(len(y_true))]
    qcut_cumresp=pd.DataFrame({'response':y_true}).groupby(qcut).agg(sum).sort_index().cumsum()
    qcut_cumsamp=pd.DataFrame({'count':sample_size},index=y_true.index).groupby(qcut).agg(sum).sort_index().cumsum()
    lift=qcut_cumsamp.join(qcut_cumresp)
    lift['lift']=lift['response']/(lift['count']*overall_resp)
    cut_x1=1*(1/ncut)
    cut_x2=5*(1/ncut)
    cut_x3=10*(1/ncut)
    cut_y1=lift.iloc[0]['lift']  # get_value/_get_value
    cut_y2=lift.iloc[4]['lift']
    cut_y3=lift.iloc[9]['lift']
 
    plt.rcdefaults()#重置rc所有参数，初始化
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.rc('figure', figsize=(8,6))
    plt.plot(qcut_x, lift.lift, color='blue', lw=2, label='Lift Curve') 
    plt.plot([cut_x1], [cut_y1],'o',color='red', label='Top '+ str(int((100*(1/ncut)))) +'pct Lift: %0.1f' % cut_y1) 
    plt.plot([cut_x2], [cut_y2],'o',color='firebrick', label='Top '+ str(int((500*(1/ncut)))) +'pct Lift: %0.1f' % cut_y2) 
    plt.plot([cut_x3], [cut_y3],'o',color='green', label='Top '+ str(int((1000*(1/ncut)))) +'pct Lift: %0.1f' % cut_y3) 
    plt.xlabel('Proportion')
    plt.ylabel('Lift')
    plt.title(name+' Lift Curve')
    plt.legend(loc="best")
    plt.show()
    return lift

def bins_freq(data, num_of_bins=10, labels=None):
    '''
    分箱 - 按照相同的频率分箱
    
    data:    list.Series 数据，用于分箱，一般为分数的连续值
    num_of_bins: 箱子的个数
    labels:  个数和bins的个数相等
    
    return:  分箱后的label
    '''
    data = pd.Series(data)
    
    # 检查数据是否全为相同值
    if data.nunique() <= 1:
        error_msg = f"错误: 数据中只有 {data.nunique()} 个唯一值，无法进行等频分箱"
        raise ValueError(error_msg)
    
    # 检查唯一值数量是否少于分箱数量
    unique_count = data.nunique()
    if unique_count < num_of_bins:
        error_msg = f"错误: 数据中只有 {unique_count} 个唯一值，少于分箱数量 {num_of_bins}，无法进行等频分箱"
        raise ValueError(error_msg)
    
    # 检查数据量是否足够
    if len(data) < num_of_bins:
        error_msg = f"错误: 数据量 {len(data)} 少于分箱数量 {num_of_bins}，无法进行等频分箱"
        raise ValueError(error_msg)
    
    try:
        # 进行等频分箱
        if labels == None:
            r = pd.qcut(data, q=np.linspace(0, 1, num_of_bins+1), precision=10, retbins=True, duplicates='drop')
        else:
            # 先检查实际分箱数量，然后调整labels
            temp_r = pd.qcut(data, q=np.linspace(0, 1, num_of_bins+1), precision=10, retbins=True, duplicates='drop')
            actual_bins = len(temp_r[1]) - 1
            
            # 调整labels数量以匹配实际分箱数量
            if actual_bins < num_of_bins:
                if len(labels) > actual_bins:
                    labels = labels[:actual_bins]
                else:
                    # 如果labels不够，生成新的labels
                    labels = list(range(actual_bins, 0, -1))
            
            # 使用调整后的labels进行分箱
            r = pd.qcut(data, q=np.linspace(0, 1, num_of_bins+1), precision=10, retbins=True, labels=labels, duplicates='drop')
        
        # 统一检查实际分箱数量并显示警告
        actual_bins = len(r[1]) - 1
        if actual_bins < num_of_bins:
            warning_msg = f"警告: 由于重复值，实际分箱数量为 {actual_bins}，少于预期 {num_of_bins}"
            print(warning_msg)
        
        return r[0]
        
    except Exception as e:
        error_msg = f"等频分箱失败: {e}"
        raise ValueError(error_msg)
        

def bins_points(data, cut_points, labels=None):
    '''
    分箱 - 按照给定的几个cut points分箱

    labels:  个数比cut_points的个数少1
    cut_points: 必须倒掉递增

    return:  分箱后的label
    '''
    if float('inf') not in cut_points:
        # cut_points = [min(data)] + cut_points + [max(data)]
        cut_points = [-np.inf] + cut_points + [np.inf]

    if labels == None:
        r = pd.cut(data, bins=cut_points, include_lowest=True)
    else:
        r = pd.cut(data, bins=cut_points, labels=labels, include_lowest=True)
    return r


def sta_groups(y_true, y_pred,cut_points=None, bins_num=10,labels=list(range(10,0,-1))):
    # bins 合并在一起
    df = pd.DataFrame()
    df['y_true']=list(y_true)
    df['y_pred']=list(y_pred)
    if cut_points:
        df['level'] = bins_points(df.y_pred, cut_points=cut_points, labels=labels).astype(int)
        df['range'] = bins_points(df.y_pred, cut_points=cut_points).astype(str)
    else:
        try:
            # 尝试等频分箱
            df['level'] = bins_freq(df.y_pred, num_of_bins=bins_num, labels=labels).astype(int)
            df['range'] = bins_freq(df.y_pred, num_of_bins=bins_num).astype(str)
        except ValueError as e:
            # 如果等频分箱失败，给出明确的错误信息
            error_msg = f"等频分箱失败，无法进行统计分组分析: {e}"
            raise ValueError(error_msg)


    temp=pd.crosstab([df['level'],df['range']],df['y_true']).reset_index()
    temp.columns=['level','range','good','bad']
    temp['num']=temp['good']+temp['bad']
    temp['bad_ration']=temp['bad']/temp['num']
    temp['acc_bad_ration']=temp['bad'].cumsum()/temp['bad'].sum()
    temp['acc_num_ration']=temp['num'].cumsum()/temp['num'].sum()
    temp['acc_precison']=temp['bad'].cumsum()/temp['num'].cumsum()
    temp=temp[['level','range','num','good','bad','bad_ration','acc_bad_ration','acc_precison','acc_num_ration']]
    temp['lift']=temp['acc_precison']/temp['acc_precison'].min()
    return temp


def sorting_ability(df,name_label,name, text='', is_asce=False):
    width, lw = 0.5, 2
    fig, ax1 = plt.subplots(figsize = (8, 6), dpi=100)
    ax2 = ax1.twinx()                                    # 创建第二个坐标轴（设置 xticks 的时候使用 ax1）

#     # 柱状图
    ax2.bar(df['level'], df['num'], width, alpha=0.5)

#     折线图
    ax1.plot(df['level'], df['acc_bad_ration'], linestyle='--', lw=lw, label=name_label, marker='o')
    for a,b in zip(df['level'],df['acc_bad_ration']):
        ax1.text(a, b, '%.2f' % b, ha='center', va= 'bottom',fontsize=9)
    
    ax1.plot(df['level'], df['acc_precison'], linestyle='--', lw=lw, label=name, marker='o')
    for a,b in zip(df['level'],df['acc_precison']):
        ax1.text(a, b, '%.4f' % b, ha='center', va= 'top',fontsize=9)

    if is_asce:
        ax1.set_xlabel('Groups(Good-> Bad)')
    else:
        ax1.set_xlabel('Groups(Bad -> Good)')
    ax1.set_ylabel('Percentage')
    ax2.set_ylabel('The number of person')

    plt.xticks(rotation=90)
    plt.xticks(df['level'])

    ax1.legend(loc="upper left")
    plt.title(f'{text} '+name+' sorting ability')
    plt.show()
    
    

    
def Transfer2Score(prob=0,p0=600,PDO=40,theta=6.63):
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




from sklearn.base import BaseEstimator,TransformerMixin
from decimal import Decimal

class Psi(BaseEstimator,TransformerMixin):
    def __init__(self,q=10,max_th=10,min_th=-10,smooth=0.0001,range_ratio=0.2,precision=8):
        self.max_th=max_th
        self.min_th=min_th
        self.smooth=smooth
        self.range_ratio=range_ratio
        self.q=q
        self.precision=precision
        
    def fit(self,expect_score):
        expect_score=pd.Series(expect_score)
        self.bins=self.qcut(expect_score,q=self.q,range_ratio=self.range_ratio,precision=self.precision)
        self.expect_dist=pd.DataFrame(
              pd.cut(expect_score,bins=self.bins,include_lowest=True).value_counts()).reset_index()
        self.expect_dist.columns=['cat','expect_score']
        return self
    
    def transform(self,actual_score):
        actual_score=pd.Series(actual_score)
        s=pd.cut(actual_score,bins=self.bins,include_lowest=True).value_counts()+self.smooth
        s.columns=['actual_score']
        extreme_ratio=(len(actual_score)+float(len(s) * Decimal(str(self.smooth)))-float(
            Decimal(str(s.sum())).quantize(Decimal(str(self.smooth)))))/len(actual_score)
        s=s.reset_index()
        s.columns=['cat','actual_score']
        self.dist=pd.merge(self.expect_dist,s,on='cat')
        self.dist['expect_score_ratio']=self.dist.expect_score/self.dist.expect_score.sum()
        self.dist['actual_score_ratio']=self.dist.actual_score/self.dist.actual_score.sum()
        
        lg=np.log(self.dist['actual_score_ratio']/self.dist['expect_score_ratio'])
        lg[lg>self.max_th]=self.max_th
        lg[lg<self.min_th]=self.min_th
        p=np.sum((self.dist['actual_score_ratio'] - self.dist['expect_score_ratio']) * lg)
        
        return p,extreme_ratio
    
    @staticmethod
    def qcut(l,q=10,range_ratio=0.1,precision=8):
        _,cut_points=pd.qcut(l,q=q,retbins=True,precision=precision,duplicates='drop')
        span=cut_points[-1]-cut_points[0]
        cut_points[0]-=span * range_ratio
        cut_points[-1] += span * range_ratio
        
        return cut_points 
    
    
    
def get_bin_points(data,cut_points):
    '''
    data:pd.Series
    
    '''
    r = pd.cut(data, bins=cut_points, include_lowest=True).astype(str)
    
    return r


def get_sta_groups(y_true,y_pred,cut_points):
    df = pd.DataFrame()
    df['y_true']=list(y_true)
    df['y_pred']=list(y_pred)
    r=pd.cut(y_pred, bins=cut_points, include_lowest=False).astype(str)
    df['range'] = r
    
    temp=pd.crosstab(df['range'],df['y_true']).reset_index()
    temp.columns=['range','good','bad']
    temp['num']=temp['good']+temp['bad']
    temp['bad_ration']=temp['bad']/temp['num']
    temp['acc_bad_ration']=temp['bad'].cumsum()/temp['bad'].sum()
    temp['acc_num_ration']=temp['num'].cumsum()/temp['num'].sum()
    temp['num_ration']=temp['num']/temp['num'].sum()
    temp['acc_precison']=temp['bad'].cumsum()/temp['num'].cumsum()
    temp=temp[['range','num','good','bad','bad_ration','acc_bad_ration','acc_precison','acc_num_ration','num_ration']]
    temp['lift']=temp['acc_precison']/temp['acc_precison'].min()
    temp=temp.reset_index()
    temp['index']=temp['index']+1
    
    return temp


def sorting_ability_plot(df,name_label, text='', is_asce=False,item1='num_ration',item2='bad_ration'):
    width, lw = 0.5, 2
    fig, ax1 = plt.subplots(figsize = (8, 6), dpi=100)
    ax2 = ax1.twinx()                                    # 创建第二个坐标轴（设置 xticks 的时候使用 ax1）

#     # 柱状图
    ax2.bar(df['index'], df[item1], width, alpha=0.5)
#     for a,b in zip(df['index'],df[item1]):
#         ax1.text(a, b, '%.2f' % b, ha='center', va= 'bottom',fontsize=9)

    

#     折线图
    ax1.plot(df['index'], df[item2], linestyle='--', lw=lw, label=name_label, marker='o')
    for a,b in zip(df['index'],df[item2]):
        ax1.text(a, b, '%.3f' % b, ha='center', va= 'bottom',fontsize=9)
    

    if is_asce:
        ax1.set_xlabel('Groups(Good-> Bad)')
    else:
        ax1.set_xlabel('Groups(Bad -> Good)')
    ax1.set_ylabel('Percentage')
    ax2.set_ylabel('The number of person')

    #plt.xticks(rotation=90)
    #plt.xticks(list(df['index']),list(df['index']))
    
    
    plt.sca(ax1)
    plt.xticks(rotation=90)
    plt.xticks(list(df['index']),list(df['range']))
    #plt.setp(plt.gca().get_xticklabels(), rotation=-45, horizontalalignment='right')


    ax1.legend(loc="upper right")
    plt.title(f'{text} '+name_label)
    plt.show()
    

# sorting_ability(tmp,'bad_rate',text='XJ_Mob3',is_asce=False)


















    