# import packages
import pandas as pd
import numpy as np
from sklearn import tree
try:
    from sklearn.linear_model.logistic import LogisticRegression
except:
    from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
import math
from scipy import stats
from sklearn.utils.multiclass import type_of_target
#import Orange
import random
from math import *
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf


def get_tree_threshold_not_used(clf):
    '''
    Note: clf 是tree fit完的结果
    '''
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    #feature = clf.tree_.feature
    threshold_tmp = clf.tree_.threshold
    boundary = []
    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
            boundary.append(threshold_tmp[i])
    boundary.sort()
    return boundary
    
def get_tree_threshold(clf):    

    #from sklearn.tree import _tree
    #_tree.TREE_UNDEFINED  == -2
    
    th = clf.tree_.threshold
    f = clf.tree_.feature  #feature 简写为 f
    # 对于没有参与分裂的节点，dt默认会给-2,所以这里要根据dt.tree_.feature把-2踢掉
    threshold=sorted(th[np.where(f != -2)])
    
    return threshold


#---------------------no problem---------------------------------------------------
def VIF(Y,X):
    from patsy import dmatrices
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    """
    VIF for each features:
    :param X: 2-D numpy array explanatory features
    :param Y: 1-D numpy array target variable
    """
    df=X.copy()
    df['Y_target'] = Y
    # 'feature_1 + feature_2 ... feature_p'
    vif_features = [x for x in X.columns if x != 'intercept']
    features_formula = "+".join(vif_features)

    # get y and X dataframes based on this formula:
    # indirect_expenditures ~ feature_1 + feature_2 ... feature_p
    vif_y, vif_X = dmatrices('Y_target' + '~' + features_formula, df, return_type='dataframe')

    # For each Xi, calculate VIF and save in dataframe
    vif = pd.DataFrame(index=vif_X.columns) 
    vif["vif"] = [variance_inflation_factor(vif_X.values, i) for i in range(vif_X.shape[1])]
    vif=vif.drop('Intercept')
    return vif

#vif=VIF(df['b'],df)

#---------------------vif used(much easier)-------------------------------------------------    
def vif(X):    
    from statsmodels.stats.outliers_influence import variance_inflation_factor 
    from statsmodels.tools.tools import add_constant 
    X = add_constant(X) 
    vif=pd.DataFrame()    #vif=pd.DataFrame(X.columns.tolist(),columns=['name'])
    vif['name']=X.columns.tolist() 
    vif['vif']=[variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif    
  
#---------------------have some problem-------------------------------------------------   

def feature_selection_logistic(y, x, alpha=0.05, stepwise = True):
    """Logistic model designed by forward selection.

    Parameters:
    -----------
    x : pandas DataFrame with all possible predictors 

    y: response array
    
    alpha: significant level (p value) for model selection
    
    stepwise: True for stepwise selection, otherwise forward selection

    Returns:
    --------
    significant features
    """
    remaining = set(x.columns)
    
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and best_new_score<=alpha:
        scores_with_candidates = []
        for candidate in remaining:
            x_candidates = selected + [candidate]
            try:
                model_stepwise_forward = sm.Logit(y, sm.add_constant(x[x_candidates]).astype(float)).fit()
            except:
                x_candidates.remove(candidate)
                print("\n\t feature " + candidate + " selection exception occurs")
                continue
            score = model_stepwise_forward.pvalues[candidate]
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort(reverse=True)
        best_new_score, best_candidate = scores_with_candidates.pop()
        if best_new_score <= alpha:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            print(best_candidate +' enters: pvalue: ' + str(best_new_score))
        
        #print(f'selected features are {selected}')
        if stepwise:
            model_stepwise_backford=sm.Logit(y, sm.add_constant(x[selected]).astype(float)).fit()
            for i in selected:
                if model_stepwise_backford.pvalues[i]>alpha:
                    selected.remove(i)
                    print(i +' removed: pvalue: ' + str(model_stepwise_backford.pvalues[i]))
    
    model_end = sm.Logit(y, sm.add_constant(x[selected]).astype(float)).fit()
    log=str(model_end.summary())
    return selected,log



#final_vars,log = feature_selection_logistic(y_train,X_train,alpha=0.05, stepwise =True)
#print(log)


def feature_selection_linear(y, x, alpha=0.05, stepwise = True):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    x : pandas DataFrame with all possible predictors 

    y: response array
    
    alpha: significant level (p value) for model selection
    
    stepwise: True for stepwise selection, otherwise forward selection

    Returns:
    --------
    significant features
    """
    remaining = set(x.columns)
    
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and best_new_score<=alpha:
        scores_with_candidates = []
        for candidate in remaining:
            x_candidates = selected + [candidate]
            try:
                model_stepwise_forward = sm.OLS(endog=y, exog=sm.add_constant(x[x_candidates]).astype(float)).fit()
            except:
                x_candidates.remove(candidate)
                print("\n\t feature " + candidate + " selection exception occurs")
                continue
            score = model_stepwise_forward.pvalues[candidate]
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort(reverse=True)
        best_new_score, best_candidate = scores_with_candidates.pop()
        if best_new_score <= alpha:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            print(best_candidate +' enters: pvalue: ' + str(best_new_score))
        
        if stepwise:
            model_stepwise_backford=sm.OLS(endog=y, exog= sm.add_constant(x[selected]).astype(float)).fit()
            for i in selected:
                if model_stepwise_backford.pvalues[i]>alpha:
                    selected.remove(i)
                    print(i +' removed: pvalue: ' + str(model_stepwise_backford.pvalues[i]))

    model_end = sm.OLS(endog=y, exog=sm.add_constant(x[selected]).astype(float)).fit()
    log=str(model_end.summary())
    return selected,log


#final_vars,log = feature_selection_linear(y_train,X_train,alpha=0.05, stepwise =True)
#print(log)

def rule_learner( data = "",      # inputdata
                    dep_var = "",                                 # dependent variable name
                    threhold_lift = 2,                                     # threhold lift for significant rule
                    loop_num = 5,                                          # random loop times
                    min_covered_rate = 0.01,                               # minimum covered cover rate of record for each rule
                    max_feature_num = 2,                                   # max feature number for each rule
                    oversampled_dep_rate=0.5                               # oversampled dependent rate, set >= 0.5
                    ):
    
    """
    input params:
        data = Orange.data.Table("For_RuleLearner.csv")       # Orange inputdata
        dep_var = "Ind_Fraud"                                 # dependent variable name
        threhold_lift = 2                                     # threhold lift for significant rule
        loop_num = 5                                          # random loop times
        min_covered_rate = 0.01                               # minimum covered cover rate of record for each rule
        max_feature_num = 2                                   # max feature number for each rule
        oversampled_dep_rate=0.5                              # oversampled dependent rate, set >= 0.5
    output:
        rule_set = {'1.Rule': if-then-rules, 
                    '2.SampleDistribution': [0-sample, 1-sample],
                    '3.SampleProblility': 0-rate, 1-rate],
                    '4.Score': 1_predict_probalility,
                    '5.Lift': 1_predict_probalility/overall_1_probalility
                    '6.OverallDepRate':overall_1_probalility},
    """

    data_1=Orange.data.Table(data.domain,[d for d in data if (d[dep_var] == '1')])
    data_0=Orange.data.Table(data.domain,[d for d in data if (d[dep_var] == '0')])


    for i in range(loop_num):
        random.seed(i* random.randint(1000, 10000))
        oversample_data_0_record=int(len(data_1)*(1-oversampled_dep_rate)/(oversampled_dep_rate))
        data_0_oversampled=Orange.data.Table(data_0.domain, random.sample(data_0, oversample_data_0_record))
        data_oversampled=Orange.data.Table.concatenate([data_1,data_0_oversampled],axis=0)
        origin_dep_rate = len(data_1)/len(data)
        oversample_dep_rate=len(data_1)/len(data_oversampled)
        offset=log(1-origin_dep_rate)+log(oversample_dep_rate)-log(origin_dep_rate)-log(1-oversample_dep_rate)


        learner = Orange.classification.rules.CN2SDLearner()

        # consider up to 10 solution streams at one time
        learner.rule_finder.search_algorithm.beam_width = 10

        # continuous value space is constrained to reduce computation time
        learner.rule_finder.search_strategy.constrain_continuous = True

        # found rules must cover at least 15 examples
        learner.rule_finder.general_validator.min_covered_examples = min_covered_rate*len(data)*origin_dep_rate/oversample_dep_rate

        # found rules may combine at most 2 selectors (conditions)
        learner.rule_finder.general_validator.max_rule_length = max_feature_num


        # validator significance
        learner.rule_finder.significance_validator.default_alpha = 0.05
        learner.rule_finder.significance_validator.parent_alpha = 0.05

        learner.gamma=1
        classifier = learner(data_oversampled)

        rule_str_set=[]
        rule_set=[]
        for r in classifier.rule_list:
            logit=log(r.probabilities[1])-log(1-r.probabilities[1])
            score=1/(1+exp(offset-logit))
            if score>threhold_lift*origin_dep_rate and str(r) not in rule_str_set:
                rule_str_set.append(str(r))
                rule_set.append({'1.Rule':str(r),'2.SampleDistribution':list(r.curr_class_dist),
                                 '3.SampleProblility':list(r.probabilities),'4.Score':score,'5.Lift':score/origin_dep_rate,
                                 '6.OverallDepRate':origin_dep_rate})

    return rule_set

class WOE:
    def __init__(self):
        self._WOE_MIN = -20
        self._WOE_MAX = 20

    def woe(self, X, y, event=1):
        '''
        Calculate woe of each feature category and information value
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable which should be binary
        :param event: value of binary stands for the event to predict
        :return: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
                 numpy array of information value of each feature
        '''
        self.check_target_binary(y)
        X1 = self.feature_discretion(X)

        res_woe = []
        res_iv = []
        for i in range(0, X1.shape[-1]):
            x = X1[:, i]
            woe_dict, iv1 = self.woe_single_x(x, y, event)
            res_woe.append(woe_dict)
            res_iv.append(iv1)
        return np.array(res_woe), np.array(res_iv)

    def woe_single_x(self, x, y, event=1):
        '''
        calculate woe and information for a single feature
        :param x: 1-D numpy starnds for single feature
        :param y: 1-D numpy array target variable
        :param event: value of binary stands for the event to predict
        :return: dictionary contains woe values for categories of this feature
                 information value of this feature
        '''
        self.check_target_binary(y)

        event_total, non_event_total = self.count_binary(y, event=event)
        x_labels = pd.Categorical(x).categories.values
        woe_dict = {}
        iv = 0
        for x1 in x_labels:
            y1 = y[np.where(x == x1)[0]]
            event_count, non_event_count = self.count_binary(y1, event=event)
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            if rate_event == 0:
                woe1 = self._WOE_MIN
            elif rate_non_event == 0:
                woe1 = self._WOE_MAX
            else:
                woe1 = math.log(rate_event / rate_non_event)
            woe_dict[x1] = woe1
            iv += (rate_event - rate_non_event) * woe1
        return woe_dict, iv

    def woe_replace(self, X, woe_arr):
        '''
        replace the explanatory feature categories with its woe value
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param woe_arr: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
        :return: the new numpy array in which woe values filled
        '''
        if X.shape[-1] != woe_arr.shape[-1]:
            raise ValueError('WOE dict array length must be equal with features length')

        res = np.copy(X).astype(float)
        idx = 0
        for woe_dict in woe_arr:
            for k in woe_dict.keys():
                woe = woe_dict[k]
                res[:, idx][np.where(res[:, idx] == k)[0]] = woe * 1.0
            idx += 1

        return res

    def combined_iv(self, X, y, masks, event=1):
        '''
        calcute the information vlaue of combination features
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable
        :param masks: 1-D numpy array of masks stands for which features are included in combination,
                      e.g. np.array([0,0,1,1,1,0,0,0,0,0,1]), the length should be same as features length
        :param event: value of binary stands for the event to predict
        :return: woe dictionary and information value of combined features
        '''
        if masks.shape[-1] != X.shape[-1]:
            raise ValueError('Masks array length must be equal with features length')

        x = X[:, np.where(masks == 1)[0]]
        tmp = []
        for i in range(x.shape[0]):
            tmp.append(self.combine(x[i, :]))

        dumy = np.array(tmp)
        # dumy_labels = np.unique(dumy)
        woe, iv = self.woe_single_x(dumy, y, event)
        return woe, iv

        
    def combine(self, list):
        res = ''
        for item in list:
            res += str(item)
        return res

    def count_binary(self, a, event=1):
        event_count = (a == event).sum()
        non_event_count = a.shape[-1] - event_count
        return event_count, non_event_count

    def check_target_binary(self, y):
        '''
        check if the target variable is binary, raise error if not.
        :param y:
        :return:
        '''
        y_type = type_of_target(y)
        if y_type not in ['binary']:
            raise ValueError('Label type must be binary')

    def feature_discretion(self, X):
        '''
        Discrete the continuous features of input data X, and keep other features unchanged.
        :param X : numpy array
        :return: the numpy array in which all continuous features are discreted
        '''
        temp = []
        for i in range(0, X.shape[-1]):
            x = X[:, i]
            x_type = type_of_target(x)
            if x_type == 'continuous':
                x1 = self.discrete(x)
                temp.append(x1)
            else:
                temp.append(x)
        return np.array(temp).T

    def discrete(self, x):
        '''
        Discrete the input 1-D numpy array using 5 equal percentiles
        :param x: 1-D numpy array
        :return: discreted 1-D numpy array
        '''
        res = np.array([0] * x.shape[-1], dtype=int)
        for i in range(5):
            point1 = stats.scoreatpercentile(x, i * 20)
            point2 = stats.scoreatpercentile(x, (i + 1) * 20)
            x1 = x[np.where((x >= point1) & (x <= point2))]
            mask = np.in1d(x, x1)
            res[mask] = (i + 1)
        return res

    @property
    def WOE_MIN(self):
        return self._WOE_MIN
    @WOE_MIN.setter
    def WOE_MIN(self, woe_min):
        self._WOE_MIN = woe_min
    @property
    def WOE_MAX(self):
        return self._WOE_MAX
    @WOE_MAX.setter
    def WOE_MAX(self, woe_max):
        self._WOE_MAX = woe_max

### Continuous variable profiling and recoding
def univariate_cont_prof_recode(X, Y, event=1, recoding_std=True, recoding_prefix='r_', recoding_woe=True, 
                     prof_cut_group = 10, monotonic_bin=True , prof_tree_cut=True, prof_min_p= 0.05, prof_threshold_cor=0.1,class_balance=True):
    """
    profiling and recoding  for continuous feature based on binary target:
    :param X: 1-D numpy starnds for single feature
    :param Y: 1-D numpy array target variable
    :param event: target event, default 1
    :param recoding_std: True for standard recoding
    :param recoding_woe: True for woe recoding
    :param prof_cut_group: max bin groups
    :param prof_tree_cut: True for tree based (optimal) binning, False for equal cut
    :param prof_min_p: minimal sample rate in each bin, default 0.05, only available for tree_cut=True
    :param prof_threshold_cor: threshold spearman correlation of target_rank by groups, only avaible for tree_cut=False
    :output profile_df: feature profile dataframe
    :output Xstatistics: feature statistics dataframe
    :output statement_recoding: feature recoding script
    """
    Y_org = Y
    Y1_org = Y_org[pd.isnull(X)] 
    Y2_org = Y_org[-pd.isnull(X)]
    
    if type_of_target(Y) not in ['binary']: 
        Y = (Y >= Y.quantile(0.50))*1
        event = 1
        recoding_woe = False
    if event != 1:
        Y=1-Y
    
    X1 = X[pd.isnull(X)]
    Y1 = Y[pd.isnull(X)]                       
    X2 = X[-pd.isnull(X)]
    Y2 = Y[-pd.isnull(X)]
    r = 0
    min_samples_group=int(len(Y)*prof_min_p)
    
        
    # whether use balanced class weight
    if class_balance==True:
        class_weight='balanced'
    else:
        class_weight=None
    
    # non monotonic cut
    if not monotonic_bin:    # False：单调性否---执行
        # Tree Based Cut Point
        if prof_tree_cut == True:
            if type_of_target(Y) not in ['binary']:
                clf = tree.DecisionTreeRegressor(criterion = 'mse', max_leaf_nodes=prof_cut_group, min_samples_leaf=min_samples_group)
            else:
                clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_leaf_nodes=prof_cut_group, min_samples_leaf=min_samples_group,class_weight=class_weight)
            clf.fit(X2.values.reshape(-1, 1),Y2)
            
            # threshold = clf.tree_.threshold[clf.tree_.threshold>-2]  #有可能存在小于-2,或者真实情况可能就是-2的情况（概率低而已）
            # threshold = np.sort(threshold)
            # threshold = threshold.tolist()
            
            threshold=get_tree_threshold(clf)       
            threshold.append(X2.quantile(0))
            threshold.append(X2.quantile(1))
            cut_points=list(set(threshold))
            cut_points.sort()
        # Equal Cut (等频切分)
        else:
            cut_points=list(set(X2.quantile(i/prof_cut_group) for i in range(prof_cut_group+1))) #----0.1,0.2,0.3....1 分位数
            cut_points.sort()       
        d1 = pd.DataFrame({"X": X2, "Y": Y2, "Bucket": pd.cut(X2,cut_points,include_lowest=True)})
        d1_nan = pd.DataFrame({"X": X1, "Y": Y1, "Bucket": ['_MISSING_' for i in Y1]})
        d1 = pd.concat([d1_nan,d1]).reset_index(drop = True)
        d2 = d1.groupby('Bucket', as_index = True)
    
    # monotonic cut
    else:
        while np.abs(r) < prof_threshold_cor and prof_cut_group>1:
            # Tree Based Cut Point
            if prof_tree_cut == True:
                if type_of_target(Y) not in ['binary']:
                    clf = tree.DecisionTreeRegressor(criterion = 'mse', max_leaf_nodes=prof_cut_group, min_samples_leaf=min_samples_group)
                else:
                    clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_leaf_nodes=prof_cut_group, min_samples_leaf=min_samples_group,class_weight=class_weight)
                clf.fit(X2.values.reshape(-1, 1),Y2)
                
                # threshold = clf.tree_.threshold[clf.tree_.threshold>-2]
                # threshold = np.sort(threshold)
                # threshold = threshold.tolist()
                
                threshold=get_tree_threshold(clf)
                threshold.append(X2.quantile(0))
                threshold.append(X2.quantile(1))
                cut_points=list(set(threshold))
                cut_points.sort()
            # Equal Cut
            else:
                cut_points=list(set(X2.quantile(i/prof_cut_group) for i in range(prof_cut_group+1)))
                cut_points.sort()
            d1 = pd.DataFrame({"X": X2, "Y": Y2, "Bucket": pd.cut(X2,cut_points,include_lowest=True)})
            d1_r = d1.groupby('Bucket', as_index = True)
            r, p = stats.spearmanr(d1_r.mean().X, d1_r.mean().Y)
            d1_nan = pd.DataFrame({"X": X1, "Y": Y1, "Bucket": ['_MISSING_' for i in Y1]})
            d1 = pd.concat([d1_nan,d1]).reset_index(drop = True)
            d2 = d1.groupby('Bucket', as_index = True)
            prof_cut_group = prof_cut_group - 1
    
    # compute WOE and IV
    woe=WOE()
    woe_dict, iv=woe.woe_single_x(d1.Bucket,d1.Y, event=1)
    
    d3 = pd.DataFrame()
    d3['min'] = d2.min().X
    d3['max'] = d2.max().X
    d3['depvar_n'] = d2.sum().Y
    d3['count'] = d2.count().Y
    d3['proportion'] = d2.count().Y/len(Y)
    d3['depvar_rate'] = d2.mean().Y    
    d3['group'] = pd.Categorical(d1["Bucket"]).categories.values
    d3['variable'] = [X.name for i in pd.Categorical(d1["Bucket"]).categories.values]
    d3['woe'] = list(woe_dict.values())
    d3['iv'] = iv
    d4 = (d3.sort_values(by = 'min')).reset_index(drop = True)

    d4_total=pd.DataFrame({'index':[1]})          
    d4_total['variable'] = X.name
    d4_total['group'] = "TOTAL"
    d4_total['min'] = np.nan
    d4_total['max'] = np.nan
    d4_total['depvar_n'] = Y.sum()
    d4_total['count'] = Y.count()
    d4_total['proportion'] = Y.count()/len(Y)
    d4_total['depvar_rate'] = Y.mean()
    d4_total['woe'] = np.nan
    d4_total['iv'] = iv
    d4_total = d4_total.drop('index',axis=1)    
    
    
    d5 = pd.concat([d4_total,d4])
    d5['lift'] = d5['depvar_rate']*100/(Y.mean())
    columns_pos = ['iv','variable','group','min','max','count','proportion','depvar_n','depvar_rate','lift','woe']
    d5 = d5.loc[:,columns_pos].reset_index(drop = True)
    
    ### Recoding tool 
    
    # statistics
    Xmean = X2.mean()
    Xmin = X2.quantile(0)
    Xmax = X2.quantile(1)
    Xp1  = X2.quantile(0.01)
    Xp99  = X2.quantile(0.99)
    Xp25  = X2.quantile(0.25)
    Xmedian  = X2.quantile(0.50)
    Xp75  = X2.quantile(0.75)
    Xiqr = max((Xp75-Xp25),(Xp99-Xp75),(Xp25-Xp1))
    Xupperbound = max(min((Xp75+1.5*Xiqr),Xmax),Xp99)
    Xlowerbound = min(max((Xp25-1.5*Xiqr),Xmin),Xp1)
    if Xupperbound==Xlowerbound:
        Xupperbound=Xmax
        Xlowerbound=Xmin
    Xmidrange = (Xmax+Xmin)*1.0/2
    
    # ER missing imputation
    Xbest_trans = 'origin'
    impute_er = Xmedian

    X2sq = X2.apply(np.square)
    X2sr = X2.apply(lambda x: np.sqrt(max(x,0)))
    X2log = X2.apply(lambda x: np.log(max(x,0.00001)))

    X2sq_min = X2sq.quantile(0)
    X2sq_max = X2sq.quantile(1)
    X2sr_min = X2sr.quantile(0)
    X2sr_max = X2sr.quantile(1)
    X2log_min = X2log.quantile(0) 
    X2log_max = X2log.quantile(1)

    df_Xtrans=pd.DataFrame({'X2':X2,'X2sq':X2sq,'X2sr':X2sr,'X2log':X2log})
    Xtrans_F,Xtrans_p = f_regression(df_Xtrans,Y2_org)

    if len(X1)>0:
        Ymean_miss=max(min(Y1.mean(),0.99999),0.00001)

        if type_of_target(Y2_org) not in ['binary']: 
            Ymean_miss=Y1_org.mean()
            impute_m = LinearRegression()
            if np.argmin(Xtrans_p) == 0:
                impute_m.fit(pd.DataFrame(X2),Y2_org)
                Xbest_trans = 'origin'
                impute_m_coef,impute_m_intercept = impute_m.coef_[0],impute_m.intercept_
                impute_er = (Ymean_miss-impute_m_intercept)/impute_m_coef

            if np.argmin(Xtrans_p) == 1:
                impute_m.fit(pd.DataFrame(X2sq),Y2_org)
                Xbest_trans = 'square'
                impute_m_coef,impute_m_intercept = impute_m.coef_[0],impute_m.intercept_
                impute_er = sqrt(max((Ymean_miss-impute_m_intercept)/impute_m_coef,0))

            if np.argmin(Xtrans_p) == 2:         
                impute_m.fit(pd.DataFrame(X2sr),Y2_org)
                Xbest_trans = 'sqrt'
                impute_m_coef,impute_m_intercept = impute_m.coef_[0],impute_m.intercept_
                impute_er = ((Ymean_miss-impute_m_intercept)/impute_m_coef)*((Ymean_miss-impute_m_intercept)/impute_m_coef)

            if np.argmin(Xtrans_p) == 3:
                impute_m.fit(pd.DataFrame(X2log),Y2_org)
                Xbest_trans = 'log'
                impute_m_coef,impute_m_intercept = impute_m.coef_[0],impute_m.intercept_
                impute_er = np.exp((Ymean_miss-impute_m_intercept)/impute_m_coef)

        else:
            impute_m = LogisticRegression()
            if np.argmin(Xtrans_p) == 0:
                impute_m.fit(pd.DataFrame(X2),Y2)
                Xbest_trans = 'origin'
                impute_m_coef,impute_m_intercept = impute_m.coef_[0][0],impute_m.intercept_[0]
                impute_er = (log(Ymean_miss/(1-Ymean_miss))-impute_m_intercept)/impute_m_coef

            if np.argmin(Xtrans_p) == 1:
                impute_m.fit(pd.DataFrame(X2sq),Y2)
                Xbest_trans = 'square'
                impute_m_coef,impute_m_intercept = impute_m.coef_[0][0],impute_m.intercept_[0]
                impute_er = sqrt(max((log(Ymean_miss/(1-Ymean_miss))-impute_m_intercept)/impute_m_coef,0))

            if np.argmin(Xtrans_p) == 2:        
                impute_m.fit(pd.DataFrame(X2sr),Y2)
                Xbest_trans = 'sqrt'
                impute_m_coef,impute_m_intercept = impute_m.coef_[0][0],impute_m.intercept_[0]
                impute_er = ((log(Ymean_miss/(1-Ymean_miss))-impute_m_intercept)/impute_m_coef)*((log(Ymean_miss/(1-Ymean_miss))-impute_m_intercept)/impute_m_coef)

            if np.argmin(Xtrans_p) == 3:
                impute_m.fit(pd.DataFrame(X2log),Y2)
                Xbest_trans = 'log'
                impute_m_coef,impute_m_intercept = impute_m.coef_[0][0],impute_m.intercept_[0]
                impute_er = np.exp((log(Ymean_miss/(1-Ymean_miss))-impute_m_intercept)/impute_m_coef)

    if pd.isnull(impute_er):
        impute_er = Xmedian
    if impute_er<Xlowerbound:
        impute_er = Xlowerbound
    if impute_er>Xupperbound:
        impute_er = Xupperbound
                 
    Xstatistics = pd.DataFrame([{
        'variable' : X.name,
        'missingrate': len(Y1)/len(Y),
        'mean':Xmean,
        'median':Xmedian,
        'min':Xmin,
        'max':Xmax,
        'upperbound':Xupperbound,
        'lowerbound':Xlowerbound,
        'midrange':Xmidrange,
        'besttrans':Xbest_trans,
        'erimpute':impute_er,
        'iv': iv
    }],columns = ['variable', 'missingrate', 'mean','median','min','max','upperbound','lowerbound','midrange','besttrans','erimpute','iv'])
    
    ### recoding output
    varname_org = X.name
    recoding_varname_org = recoding_prefix + X.name.lower()
    recoding_varname_sq = recoding_prefix + 'sq_' + X.name.lower()
    recoding_varname_sr = recoding_prefix + 'sr_' + X.name.lower()
    recoding_varname_log = recoding_prefix + 'log_' + X.name.lower()
    recoding_varname_woe = recoding_prefix + 'woe_' + X.name.lower()
    recoding_varname_org_std = recoding_prefix + 'std_' + X.name.lower()
    recoding_varname_sq_std = recoding_prefix + 'sq_' + 'std_' + X.name.lower()
    recoding_varname_sr_std = recoding_prefix + 'sr_' + 'std_' + X.name.lower()
    recoding_varname_log_std = recoding_prefix + 'log_' + 'std_' + X.name.lower()
    
    statement_impute = "df.loc[:,'" + recoding_varname_org + "'] = df['" + varname_org + "'].fillna(" + str(impute_er) + ")"
    statement_capping = "df.loc[" + "df['" + recoding_varname_org + "'] > " + str(Xupperbound) + ",'" + recoding_varname_org + "'] = " + str(Xupperbound)
    statement_flooring = "df.loc[" + "df['" + recoding_varname_org + "'] < " + str(Xlowerbound) + ",'" + recoding_varname_org + "'] = " + str(Xlowerbound)
    statement_recoding_org = statement_impute + "\n" + statement_capping + "\n" + statement_flooring
    statement_recoding_org_std = "df.loc[:,'" + recoding_varname_org_std + "'] = (df['" + recoding_varname_org + "'] - ("  \
    + str(Xlowerbound) + "))/(" + str(Xupperbound) + " - (" + str(Xlowerbound) +"))"
  
    statement_recoding_sq = "df.loc[:,'" + recoding_varname_sq + "'] = df['" + recoding_varname_org + "'].apply(np.square)"
    statement_recoding_sq_std = "df.loc[:,'" + recoding_varname_sq_std + "'] = (df['" + recoding_varname_sq + "'] - ("  \
    + str(X2sq_min) + "))/(" + str(X2sq_max) + " - (" + str(X2sq_min) +"))"
    statement_recoding_sr = "df.loc[:,'" + recoding_varname_sr + "'] = df['" + recoding_varname_org + "'].apply(lambda x: np.sqrt(max(x,0)))"
    statement_recoding_sr_std = "df.loc[:,'" + recoding_varname_sr_std + "'] = (df['" + recoding_varname_sr + "'] - ("  \
    + str(X2sr_min) + "))/(" + str(X2sr_max) + " - (" + str(X2sr_min) +"))"
    
    statement_recoding_log = "df.loc[:,'" + recoding_varname_log + "'] = df['" + recoding_varname_org + "'].apply(lambda x: np.log(max(x,0.00001)))"
    statement_recoding_log_std = "df.loc[:,'" + recoding_varname_log_std + "'] = (df['" + recoding_varname_log + "'] - ("  \
    + str(X2log_min) + "))/(" + str(X2log_max) + " - (" + str(X2log_min) +"))"

    if Xupperbound==Xlowerbound:
        statement_recoding_org_std=""
    if X2sq_max==X2sq_min:
        statement_recoding_sq_std=""
    if X2sr_max==X2sr_min:
        statement_recoding_sr_std=""
    if X2log_max==X2log_min:
        statement_recoding_log_std=""
        
    statement_woe =  "df.loc[:,'" + recoding_varname_woe + "'] = 0"
    statement_woe_missinggroup = "df.loc[pd.isnull(df['" + varname_org + "']),'" + recoding_varname_woe + "'] = " + '0'
    for iter in range(len(d5)):
        group = d5.loc[iter,'group']
        varname_org = d5.loc[iter,'variable']
        group_woe = d5.loc[iter,'woe']
        if str(group) not in ['TOTAL','_MISSING_']:
            group_edges=str(group).replace('[','').replace(']','').replace('(','').replace(')','').split(', ')
            group_minx,group_maxx = float(group_edges[0]),float(group_edges[1])
            if iter == 1:
                statement_woe_group = "df.loc[(" + str(group_minx) + " <= df['" + \
                recoding_varname_org + "']) & (df['" + recoding_varname_org + "'] <= " + str(group_maxx) + "),'" + recoding_varname_woe + "'] = " + \
                str(group_woe)
            else:
                statement_woe_group = "df.loc[(" + str(group_minx) + " <  df['" + \
                recoding_varname_org + "']) & (df['" + recoding_varname_org + "'] <=  " + str(group_maxx) + "),'" + recoding_varname_woe + "'] = " + \
                str(group_woe)
            statement_woe = statement_woe + "\n" + statement_woe_group
        if str(group) == '_MISSING_':
            statement_woe_missinggroup = "df.loc[pd.isnull(df['" + varname_org + \
            "']),'" + recoding_varname_woe + "'] = " + str(group_woe)
    statement_woe = statement_woe + "\n" + statement_woe_missinggroup 

    if recoding_std == True:
        statement_recoding = statement_recoding_org + "\n" + statement_recoding_org_std
        if Xbest_trans == 'square':
            statement_recoding = statement_recoding + "\n" + statement_recoding_sq + "\n" + statement_recoding_sq_std
        elif  Xbest_trans == 'sqrt':
            statement_recoding = statement_recoding + "\n" + statement_recoding_sr + "\n" + statement_recoding_sr_std
        elif  Xbest_trans == 'log':
            statement_recoding = statement_recoding + "\n" + statement_recoding_log + "\n" + statement_recoding_log_std
    else:
        statement_recoding = statement_recoding_org
        if Xbest_trans == 'square':
            statement_recoding = statement_recoding + "\n" + statement_recoding_sq
        elif  Xbest_trans == 'sqrt':
            statement_recoding = statement_recoding + "\n" + statement_recoding_sr
        elif  Xbest_trans == 'log':
            statement_recoding = statement_recoding + "\n" + statement_recoding_log
        
    if recoding_woe == True:
        statement_recoding = statement_recoding + "\n" + statement_woe
    
    statement_comment = "### Continuous Recoding: " + varname_org + " ###"
    statement_recoding = statement_comment + "\n" + statement_recoding  + "\n"
    
    profile_df = d5
    return profile_df,Xstatistics,statement_recoding

### Continuous variables profiling and recoding
def cont_prof_recode(X, Y, event=1, max_missing_rate=0.99, recoding_std=True, recoding_prefix='r_', recoding_woe=True, 
                     prof_cut_group = 10, monotonic_bin=True , prof_tree_cut=True, prof_min_p= 0.05, prof_threshold_cor=0.1,class_balance=True):
    """
    profiling and recoding for continuous features based on binary target:
    :param X: 2-D numpy array explanatory features
    :param Y: 1-D numpy array target variable
    :param event: target event, default 1
    """
    iter=0
    for col in X.columns.values:   
        x = X[col]         
        x1 = x[pd.isnull(x)]
        Y1 = Y[pd.isnull(x)]                       
        x2 = x[-pd.isnull(x)]
        Y2 = Y[-pd.isnull(x)]
        
        #only do profiling for features with good quality
        if len(Y1)/len(Y)>max_missing_rate:
            print("numeric feature " + col + " : missing rate is too high")
            continue 
        if len(pd.unique(x2))<=1:
            print("numeric feature " + col + " : have only one unique value")
            continue
        
        iter = iter+1
        print(str(iter) + " numeric feature " + col + " profiling processing")
        
        try:
            x_profile,x_statistics,x_statement_recoding = univariate_cont_prof_recode(x, Y, event= event, recoding_std=recoding_std, recoding_prefix=recoding_prefix,  
                     recoding_woe=recoding_woe, prof_cut_group = prof_cut_group, monotonic_bin=monotonic_bin , prof_tree_cut=prof_tree_cut, prof_min_p= prof_min_p, 
                     prof_threshold_cor=prof_threshold_cor,class_balance=class_balance)
        except:
            x_profile,x_statistics,x_statement_recoding = pd.DataFrame(),pd.DataFrame(),''
            print("\n\t error occurs")
            
        if iter == 1:
            df_profile = x_profile
            df_statistics = x_statistics
            cont_statement_recoding = x_statement_recoding
        else:
            df_profile = pd.concat([df_profile,x_profile])
            df_statistics = pd.concat([df_statistics,x_statistics])
            cont_statement_recoding = cont_statement_recoding + "\n" + x_statement_recoding
            
    df_profile = df_profile.reset_index().sort_values(by=['iv','variable','index'],ascending=[False,True,True])
    df_profile = df_profile.reset_index(drop  = True).drop('index',axis=1)
    df_statistics = df_statistics.reset_index().sort_values(by=['iv','variable','index'],ascending=[False,True,True])
    df_statistics = df_statistics.reset_index(drop  = True).drop('index',axis=1)
    return df_profile,df_statistics,cont_statement_recoding
    
def univariate_nomi_prof_recode(X, Y, event=1, recoding_woe=True, recoding_prefix='r_',
                     prof_cut_group = 10, monotonic_bin=True, prof_tree_cut=True, prof_min_p= 0.05, prof_threshold_cor=0.1,class_balance=True):
    """
    profiling and recoding for nomial feature based on binary target:
    :param X: 1-D numpy starnds for single feature
    :param Y: 1-D numpy array target variable
    :param event: target event, default 1
    :param recoding_woe: True for woe recoding
    :param prof_cut_group: max bin groups
    :param prof_tree_cut: True for tree based (optimal) binning, False for equal cut
    :param prof_min_p: minimal sample rate in each bin, default 0.05, only available for tree_cut=True
    :param prof_threshold_cor: threshold spearman correlation of target_rank by groups, only avaible for tree_cut=False
    :output profile_df: feature profile dataframe
    :output Xstatistics: feature statistics dataframe
    :output statement_recoding: feature recoding script
    """
    if type_of_target(Y) not in ['binary']:
        Y = (Y >= Y.quantile(0.50))*1
        event = 1
        recoding_woe = False
    if event != 1:
        Y=1-Y
    
    X1 = X[pd.isnull(X)]
    Y1 = Y[pd.isnull(X)]                       
    X2 = X[-pd.isnull(X)]
    Y2 = Y[-pd.isnull(X)]
    r = 0
    min_samples_group=int(len(Y)*prof_min_p)
    
    d_notnan = pd.DataFrame({"X": X2, "Y": Y2, "Bucket": X2.values.astype('str')})
    d_nan = pd.DataFrame({"X": X1, "Y": Y1, "Bucket": ['_MISSING_' for i in Y1]})
    d_bucket = pd.concat([d_notnan,d_nan]).reset_index(drop = True)
 
    # compute WOE for each category values and IV
    woe_raw=WOE()
    woe_raw_dict, raw_iv=woe_raw.woe_single_x(d_bucket.Bucket,d_bucket.Y, event=1)
    d_bucket['Bucket_woe']=[woe_raw_dict[i] for i in d_bucket['Bucket']]
    
    # whether use balanced class weight
    if class_balance==True:
        class_weight='balanced'
    else:
        class_weight=None    
    
    # non monotonic cut
    if not monotonic_bin:
        # Tree Based Cut Point
        if prof_tree_cut == True:
            if type_of_target(Y) not in ['binary']:
                clf = tree.DecisionTreeRegressor(criterion = 'mse', max_leaf_nodes=prof_cut_group, min_samples_leaf=min_samples_group)
            else:
                clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_leaf_nodes=prof_cut_group, min_samples_leaf=min_samples_group,class_weight=class_weight)
            clf.fit(d_bucket['Bucket_woe'].values.reshape(-1, 1),d_bucket.Y)
            # threshold = clf.tree_.threshold[clf.tree_.threshold>-2]
            # threshold = np.sort(threshold)
            # threshold = threshold.tolist()
            threshold=get_tree_threshold(clf)
            threshold.append(d_bucket['Bucket_woe'].quantile(0))
            threshold.append(d_bucket['Bucket_woe'].quantile(1))
            cut_points=list(set(threshold))
            cut_points.sort()
        # Equal Cut
        else:
            cut_points=list(set(d_bucket['Bucket_woe'].quantile(i/prof_cut_group) for i in range(prof_cut_group+1)))
            cut_points.sort()
        
        d_bucket['Bucket_woe_Bucket'] = pd.cut(d_bucket['Bucket_woe'],cut_points,include_lowest=True)
        d2 = d_bucket.groupby('Bucket_woe_Bucket', as_index = True)
        group_X={}
        for dic_i,dic_v in d2:
            group_X[dic_i]=list(set(dic_v.Bucket))
            
    
    # monotonic cut
    else:
        while np.abs(r) < prof_threshold_cor and prof_cut_group>1:
            # Tree Based Cut Point
            if prof_tree_cut == True:
                if type_of_target(Y) not in ['binary']:
                    clf = tree.DecisionTreeRegressor(criterion = 'entropy', max_leaf_nodes=prof_cut_group, min_samples_leaf=min_samples_group)
                else:
                    clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_leaf_nodes=prof_cut_group, min_samples_leaf=min_samples_group,class_weight=class_weight)
                clf.fit(d_bucket['Bucket_woe'].values.reshape(-1, 1),d_bucket.Y)
                # threshold = clf.tree_.threshold[clf.tree_.threshold>-2]
                # threshold = np.sort(threshold)
                # threshold = threshold.tolist()
                threshold=get_tree_threshold(clf)
                threshold.append(d_bucket['Bucket_woe'].quantile(0))
                threshold.append(d_bucket['Bucket_woe'].quantile(1))
                cut_points=list(set(threshold))
                cut_points.sort()
            # Equal Cut
            else:
                cut_points=list(set(d_bucket['Bucket_woe'].quantile(i/prof_cut_group) for i in range(prof_cut_group+1)))
                cut_points.sort()
            d_bucket['Bucket_woe_Bucket'] = pd.cut(d_bucket['Bucket_woe'],cut_points,include_lowest=True)
            d2 = d_bucket.groupby('Bucket_woe_Bucket', as_index = True)
            group_X={}
            for dic_i,dic_v in d2:
                group_X[dic_i]=list(set(dic_v.Bucket))
            
            r, p = stats.spearmanr(d2.mean().Bucket_woe, d2.mean().Y)
            prof_cut_group = prof_cut_group - 1
    
    # compute WOE and IV
    woe=WOE()
    woe_dict, iv=woe.woe_single_x(d_bucket['Bucket_woe_Bucket'],d_bucket.Y, event=1)
    d3 = pd.DataFrame()
    d3['min'] = d2.min().Bucket_woe
    d3['max'] = d2.max().Bucket_woe
    d3['depvar_n'] = d2.sum().Y
    d3['count'] = d2.count().Y
    d3['proportion'] = d2.count().Y/len(Y)
    d3['depvar_rate'] = d2.mean().Y    
    d3['group'] = [str(i) for i in group_X.values()]
    d3['variable'] = [X.name for i in pd.Categorical(d_bucket['Bucket_woe_Bucket']).categories.values]
    d3['woe'] = list(woe_dict.values())
    d3['iv'] = iv
    d4 = (d3.sort_values(by = 'min')).reset_index(drop = True)

    d4_total=pd.DataFrame({'index':[1]})          
    d4_total['variable'] = X.name
    d4_total['group'] = "TOTAL"
    d4_total['min'] = np.nan
    d4_total['max'] = np.nan
    d4_total['depvar_n'] = Y.sum()
    d4_total['count'] = Y.count()
    d4_total['proportion'] = Y.count()/len(Y)
    d4_total['depvar_rate'] = Y.mean()
    d4_total['woe'] = np.nan
    d4_total['iv'] = iv
    d4_total = d4_total.drop('index',axis=1)    
    
    
    d5 = pd.concat([d4_total,d4])
    d5['lift'] = d5['depvar_rate']*100/(Y.mean())
    columns_pos = ['iv','variable','group','min','max','count','proportion','depvar_n','depvar_rate','lift','woe']
    d5 = d5.loc[:,columns_pos].reset_index(drop = True)
    
    Xstatistics = pd.DataFrame([{
        'variable' : X.name,
        'missingrate': len(Y1)/len(Y),
        'mean':np.nan,
        'median':np.nan,
        'min':np.nan,
        'max':np.nan,
        'upperbound':np.nan,
        'lowerbound':np.nan,
        'midrange':np.nan,
        'besttrans':np.nan,
        'erimpute':np.nan,
        'iv': iv
    }],columns = ['variable', 'missingrate', 'mean','median','min','max','upperbound','lowerbound','midrange','besttrans','erimpute','iv'])

    ### recoding output
    varname_org = X.name
    recoding_varname_org = recoding_prefix + X.name.lower()
    recoding_varname_woe = recoding_prefix + 'woe_' + X.name.lower()
    
    statement_recoding_group = "### Multinominal Recoding: " + varname_org + " ###"
    statement_woe =  "df.loc[:,'" + recoding_varname_woe + "'] = 0"
    for iter in range(len(d5)):
        group = d5.loc[iter,'group']
        varname_org = d5.loc[iter,'variable']
        group_woe = d5.loc[iter,'woe']
        group_lift = d5.loc[iter,'lift']
        if str(group) not in ['TOTAL']:
            group_classes = str(group).replace("'_MISSING_'","'nan'")
            statement_woe_group = "df.loc[[str(x) in " + group_classes + " for x in df['" + \
            varname_org + "']],'" + recoding_varname_woe + "'] = " + str(group_woe)
            statement_woe = statement_woe + "\n" + statement_woe_group
            
            if group_lift<80 or group_lift>120:
                statement_recoding_group_i = "df.loc[:,'" + recoding_varname_org + "_x" + str(iter) + "'] = 0" + "\n" + \
                "df.loc[[str(x) in " + group_classes + " for x in df['" + \
                varname_org + "']],'" + recoding_varname_org + "_x" + str(iter) + "'] = 1"
                statement_recoding_group = statement_recoding_group + "\n" + statement_recoding_group_i 
    
    statement_recoding = statement_recoding_group    
    if recoding_woe == True:
        statement_recoding = statement_recoding + "\n" + statement_woe
        

    statement_recoding = statement_recoding + "\n"
    
    profile_df = d5
    return profile_df,Xstatistics,statement_recoding

### Nominal variables profiling and recoding
def nomi_prof_recode(X, Y, event=1, max_missing_rate=0.99, recoding_std=True, recoding_woe=True, recoding_prefix='r_',
                     prof_cut_group = 10, monotonic_bin=True , prof_tree_cut=True, prof_min_p= 0.05, prof_threshold_cor=0.1,class_balance=True):
    """
    profiling and recoding  for continuous features based on binary target:
    :param X: 2-D numpy array explanatory features
    :param Y: 1-D numpy array target variable
    :param event: target event, default 1
    """
    iter=0
    for col in X.columns.values:   
        x = X[col]         
        x1 = x[pd.isnull(x)]
        Y1 = Y[pd.isnull(x)]                       
        x2 = x[-pd.isnull(x)]
        Y2 = Y[-pd.isnull(x)]
        
        #only do profiling for features with good quality
        if len(Y1)/len(Y)>max_missing_rate:
            print("character feature " + col + " : missing rate is too high")
            continue 
        if len(pd.unique(x2))>200:
            print("character feature " + col + " : have more than 200 unique value")
            continue
            
        iter = iter+1
        print(str(iter) + " character feature " + col + " profiling processing")
        
        try:
            x_profile,x_statistics,x_statement_recoding = univariate_nomi_prof_recode(x, Y, event= event, recoding_prefix=recoding_prefix,
                     recoding_woe=recoding_woe, prof_cut_group = prof_cut_group, monotonic_bin=monotonic_bin , prof_tree_cut=prof_tree_cut, 
                     prof_min_p= prof_min_p, prof_threshold_cor=prof_threshold_cor,class_balance=class_balance)
        except:
            x_profile,x_statistics,x_statement_recoding = pd.DataFrame(),pd.DataFrame(),''
            print("\n\t error occurs")
            
        if iter == 1:
            df_profile = x_profile
            df_statistics = x_statistics
            nomi_statement_recoding = x_statement_recoding
        else:
            df_profile = pd.concat([df_profile,x_profile])
            df_statistics = pd.concat([df_statistics,x_statistics])
            nomi_statement_recoding = nomi_statement_recoding + "\n" + x_statement_recoding
            
    df_profile = df_profile.reset_index().sort_values(by=['iv','variable','index'],ascending=[False,True,True])
    df_profile = df_profile.reset_index(drop  = True).drop('index',axis=1)
    df_statistics = df_statistics.reset_index().sort_values(by=['iv','variable','index'],ascending=[False,True,True])
    df_statistics = df_statistics.reset_index(drop  = True).drop('index',axis=1)
    return df_profile,df_statistics,nomi_statement_recoding

    ### Features profiling and recoding
def features_prof_recode(Xcont, Xnomi, Y, event=1, max_missing_rate=0.99, recoding_std=True, recoding_woe=True, recoding_prefix='r_', 
                     prof_cut_group = 10, monotonic_bin=True , prof_tree_cut=True, prof_min_p= 0.05, prof_threshold_cor=0.1,class_balance=True):
    """
    profiling and recoding  for features based on binary target:
    :param Xcont: 2-D numpy continous features
    :param Xnomi: 2-D numpy multinominal features
    :param Y: 1-D numpy array target variable
    :param event: target event, default 1
    :param recoding_std: True for standard recoding
    :param recoding_woe: True for woe recoding
    :param prof_cut_group: max bin groups
    :param prof_tree_cut: True for tree based (optimal) binning, False for equal cut
    :param prof_min_p: minimal sample rate in each bin, default 0.05, only available for tree_cut=True
    :param prof_threshold_cor: threshold spearman correlation of target_rank by groups, only avaible for tree_cut=False
    :output profile_df: feature profile dataframe
    :output Xstatistics: feature statistics dataframe
    :output statement_recoding: feature recoding script
    """
    ### Continuous variables profiling and recoding
    if len(Xcont)==0:
        Xcont_profile,Xcont_statistics,Xcont_statement_recoding = pd.DataFrame(),pd.DataFrame(),''
    else:
        try:
            Xcont_profile,Xcont_statistics,Xcont_statement_recoding = cont_prof_recode(Xcont, Y=Y, event=event, max_missing_rate=max_missing_rate, recoding_prefix=recoding_prefix,
                     recoding_std=recoding_std, recoding_woe=recoding_woe, prof_cut_group = prof_cut_group, monotonic_bin=monotonic_bin , 
                     prof_tree_cut=prof_tree_cut, prof_min_p= prof_min_p, prof_threshold_cor=prof_threshold_cor,class_balance=class_balance)
        except:
            Xcont_profile,Xcont_statistics,Xcont_statement_recoding = pd.DataFrame(),pd.DataFrame(),''
            print('error occurs')           
    ### Multinominal variables profiling and recoding
    if len(Xnomi)==0:
        Xnomi_profile,Xnomi_statistics,Xnomi_statement_recoding = pd.DataFrame(),pd.DataFrame(),''
    else:
        try:
            Xnomi_profile,Xnomi_statistics,Xnomi_statement_recoding = nomi_prof_recode(Xnomi, Y=Y, event=event, max_missing_rate=max_missing_rate, recoding_prefix=recoding_prefix,
                     recoding_woe=recoding_woe, prof_cut_group = prof_cut_group, monotonic_bin=monotonic_bin , prof_tree_cut=prof_tree_cut, 
                     prof_min_p= prof_min_p, prof_threshold_cor=prof_threshold_cor,class_balance=class_balance)
        except:
            Xnomi_profile,Xnomi_statistics,Xnomi_statement_recoding = pd.DataFrame(),pd.DataFrame(),''
            print('error occurs')   

    df_profile = pd.concat([Xcont_profile,Xnomi_profile])
    df_statistics = pd.concat([Xcont_statistics,Xnomi_statistics])
    statement_recoding = Xcont_statement_recoding + "\n" + Xnomi_statement_recoding
    
    return df_profile,df_statistics,statement_recoding

def write_recoding_txt(statement , file , encoding = "utf-8" ):
    # Open a file
    fo = open(file, "w", encoding=encoding)
    fo.write(statement)
    # Close opend file
    fo.close()
def exec_recoding(data, recoding_txt , encoding = "utf-8"):
    df = data.copy()
    fo = open(recoding_txt, 'r', encoding= encoding)
    recoding_text = fo.read()
    fo.close()
    exec(recoding_text)
    return df

def data_split(data, partition_time_var, partition_var = 'model_ind', 
               train_test_partition=0.7, oot_timepoint = '2017-01-01',random_seed=20170101):
    data.loc[:,partition_var] = 'O'
    randomobject=np.random.RandomState(random_seed)
    rand_split = randomobject.rand(len(data))
    time_split = pd.to_datetime(data[partition_time_var])
    data.loc[time_split>= pd.to_datetime(oot_timepoint),partition_var] = 'O'
    data.loc[(time_split< pd.to_datetime(oot_timepoint)) & (rand_split<train_test_partition),partition_var] = 'L'
    data.loc[(time_split< pd.to_datetime(oot_timepoint)) & (rand_split>=train_test_partition),partition_var] = 'V'
    
    


    
