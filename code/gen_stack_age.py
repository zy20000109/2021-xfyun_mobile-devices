# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack,vstack
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression,SGDClassifier,PassiveAggressiveClassifier,RidgeClassifier,Ridge,PassiveAggressiveRegressor
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.svm import LinearSVC,LinearSVR
import scipy.sparse

#读取数据
def load_dataset(DATA_PATH):
    df_tr=pd.read_csv(DATA_PATH+'train.csv')
    df_te=pd.read_csv(DATA_PATH+'test.csv')
    df_tr_app_events=pd.read_csv(DATA_PATH+'train_app_events.csv')
    df_te_app_events=pd.read_csv(DATA_PATH+'test_app_events.csv')
    return df_tr, df_te, df_tr_app_events, df_te_app_events

#数据预处理
def data_preprocess(DATA_PATH):
    df_tr, df_te, df_tr_app_events, df_te_app_events = load_dataset(
        DATA_PATH=DATA_PATH)
    # 数据去重&拼接
    df_tr.drop_duplicates(inplace=True, ignore_index=True)
    df_te.drop_duplicates(inplace=True, ignore_index=True)
    df_tr_te = pd.concat([df_tr, df_te], axis=0, ignore_index=True)
    #提取设备app和标签数据
    app_tag = ['device_id', 'app_id', 'tag_list']
    df_app_tag = pd.concat(
        [df_tr_app_events[app_tag], df_te_app_events[app_tag]], axis=0)
    df_app_tag.drop_duplicates(inplace=True, ignore_index=True)
    df_app_tag.sort_values(by='device_id',
                           ascending=True,
                           ignore_index=True,
                           inplace=True)
    #提取设备app列表
    df_app_list = df_app_tag.groupby([
        'device_id'
    ]).apply(lambda x: str(x['app_id'].tolist()).strip('[]')).reset_index()
    df_app_list.columns = ['device_id', 'app_list']
    #获取设备标签列表
    df_tag_list=df_app_tag.groupby(['device_id']).apply(lambda x: str(x['tag_list'].tolist()).replace('[','').replace(']','')).reset_index()
    df_tag_list.columns=['device_id','tag_list']
    #app和标签数据聚合
    df_app_tag_list=pd.merge(df_app_list,df_tag_list,on='device_id',how='left')
    df_tr_te=df_tr_te.merge(df_app_tag_list,on='device_id',how='left')

    return df_tr,df_te,df_tr_te

#生成特征
def gen_features(df):
    count_vec = CountVectorizer()
    count_vec_app = count_vec.fit_transform(df['app_list'])
    tfidf_vec = TfidfVectorizer()
    tfidf_vec_app = tfidf_vec.fit_transform(df['app_list'])
    count_vec = CountVectorizer()
    count_vec_tag = count_vec.fit_transform(df['tag_list'])
    tfidf_vec = TfidfVectorizer()
    tfidf_vec_tag = tfidf_vec.fit_transform(df['tag_list'])
    df_feature = scipy.sparse.csc_matrix(
        scipy.sparse.hstack(
            [count_vec_app, tfidf_vec_app, count_vec_tag, tfidf_vec_tag]))
    return df_feature

def mae_metric(y_true,y_pre):
    y_true=y_true
    y_pre=pd.DataFrame({'res':list(y_pre)})
    return mean_absolute_error(y_true,y_pre['res'].values)

def ridge_model(X,y):
    print("开始训练Ridge...")
    kf=StratifiedKFold(n_splits=folds,shuffle=True,random_state=2021)
    kf=kf.split(X,y)
    pred_list=[]
    mae_list=[]
    stack=np.zeros((len(y),1))
    Ridge_model = Ridge(solver='auto',
                    fit_intercept=True,
                    alpha=0.4,
                    max_iter=250,
                    normalize=False,
                    tol=0.01,
                    random_state=2021)
    for i,(tr_ind,val_ind) in enumerate(kf):
        X_tr,X_val,label_tr,label_val=X[tr_ind,:],X[val_ind,:],y[tr_ind],y[val_ind]
        Ridge_model.fit(X_tr,label_tr)
        val_re=Ridge_model.predict(X=X_val)
        stack[val_ind]=np.array(val_re).reshape(len(val_re),1)
        #print(mae_metric(label_val,val_re))
        pred_list.append(Ridge_model.predict(test))
        mae_list.append(mae_metric(label_val,val_re))

    #print("mae",np.mean(mae_list))
    s=0
    for i in pred_list:
        s=s+i
    s=s/folds
    #print(stack)
    #print(s)
    df_stack1=pd.DataFrame(stack)
    df_stack2=pd.DataFrame(s)
    df_stack=pd.concat([df_stack1,df_stack2],axis=0)
    df_stack.to_csv('../user_data/tmp/age_ridge.csv', encoding='utf8', index=None)
    
def par_model(X,y):
    print("开始训练PAR...")
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2021)
    kf = kf.split(X, y)
    pred_list = []
    mae_list = []
    stack = np.zeros((len(y), 1))

    Par_model = PassiveAggressiveRegressor(fit_intercept=True,
                                             max_iter=280,
                                             tol=0.01,
                                             random_state=2021)
    for i, (tr_ind, val_ind) in enumerate(kf):
        X_tr, X_val, label_tr, label_val = X[tr_ind, :], X[
            val_ind, :], y[tr_ind], y[val_ind]
        Par_model.fit(X_tr, label_tr)
        val_re = Par_model.predict(X=X_val)
        stack[val_ind] = np.array(val_re).reshape(len(val_re), 1)
        #print(mae_metric(label_val, val_re))
        pred_list.append(Par_model.predict(test))
        mae_list.append(mae_metric(label_val, val_re))

    #print("mae", np.mean(mae_list))
    s = 0
    for i in pred_list:
        s = s + i
    s = s / folds
    #print(stack)
    #print(s)
    df_stack1 = pd.DataFrame(stack)
    df_stack2 = pd.DataFrame(s)
    df_stack = pd.concat([df_stack1, df_stack2], axis=0)
    df_stack.to_csv('../user_data/tmp/age_par.csv', encoding='utf8', index=None)
    
def svr_model(X,y):
    print("开始训练SVR...")
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2021)
    kf = kf.split(X, y)
    pred_list = []
    mae_list = []
    stack = np.zeros((len(y), 1))

    SVR_model = LinearSVR(random_state=2021)
    for i, (tr_ind, val_ind) in enumerate(kf):
        X_tr, X_val, label_tr, label_val = X[tr_ind, :], X[
            val_ind, :], y[tr_ind], y[val_ind]
        SVR_model.fit(X_tr, label_tr)
        val_re = SVR_model.predict(X=X_val)
        stack[val_ind] = np.array(val_re).reshape(len(val_re), 1)
        #print(mae_metric(label_val, val_re))
        pred_list.append(SVR_model.predict(test))
        mae_list.append(mae_metric(label_val, val_re))

    #print("mae", np.mean(mae_list))
    s = 0
    for i in pred_list:
        s = s + i
    s = s / folds
    #print(stack)
    #print(s)
    df_stack1 = pd.DataFrame(stack)
    df_stack2 = pd.DataFrame(s)
    df_stack = pd.concat([df_stack1, df_stack2], axis=0)
    df_stack.to_csv('../user_data/tmp/age_svr.csv', encoding='utf8', index=None)
    
    
    
if __name__ == '__main__':
    DATA_PATH = '../xfdata/'
    print('读取数据...')
    df_tr, df_te, df_tr_te = data_preprocess(DATA_PATH=DATA_PATH)

    print('开始特征工程...')
    df_feature = gen_features(df_tr_te)
    
    print("开始模型训练...")
    tr_feature=df_feature[:len(df_tr)]
    label=df_tr['age']
    te_feature=df_feature[len(df_tr):]
    X=tr_feature
    test=te_feature
    y=label
    folds=5
    
    ridge_model(X,y)
    par_model(X,y)
    svr_model(X,y)