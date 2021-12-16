# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack, vstack
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier, Ridge, PassiveAggressiveRegressor
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC, LinearSVR
import scipy.sparse


#读取数据
def load_dataset(DATA_PATH):
    df_tr = pd.read_csv(DATA_PATH + 'train.csv')
    df_te = pd.read_csv(DATA_PATH + 'test.csv')
    df_tr_app_events = pd.read_csv(DATA_PATH + 'train_app_events.csv')
    df_te_app_events = pd.read_csv(DATA_PATH + 'test_app_events.csv')
    return df_tr, df_te, df_tr_app_events, df_te_app_events


#数据预处理
def data_preprocess(DATA_PATH):
    df_tr, df_te, df_tr_app_events, df_te_app_events = load_dataset(
        DATA_PATH=DATA_PATH)
    # 数据去重&拼接
    df_tr.drop_duplicates(inplace=True, ignore_index=True)
    df_te.drop_duplicates(inplace=True, ignore_index=True)
    df_tr_te = pd.concat([df_tr, df_te], axis=0, ignore_index=True)
    #提取设备激活app数据
    ac_app_tag = ['device_id', 'app_id', 'tag_list', 'is_active', 'date']
    df_ac_app_tag = pd.concat(
        [df_tr_app_events[ac_app_tag], df_te_app_events[ac_app_tag]], axis=0)
    df_ac_app_tag = df_ac_app_tag[df_ac_app_tag['is_active'] == 1]
    df_ac_app_tag = df_ac_app_tag.sort_values(by='date', ignore_index=True)
    del df_ac_app_tag['is_active']
    del df_ac_app_tag['date']
    df_ac_app_list = df_ac_app_tag.groupby([
        'device_id'
    ]).apply(lambda x: str(x['app_id'].tolist()).strip('[]')).reset_index()
    df_ac_app_list.columns = ['device_id', 'ac_app_list']
    df_tr_te = df_tr_te.merge(df_ac_app_list, on='device_id', how='left')
    df_tr_te['ac_app_list'] = df_tr_te['ac_app_list'].astype(str)

    return df_tr, df_te, df_tr_te


#生成特征
def gen_features(df):
    count_vec = CountVectorizer(ngram_range=(1, 3))
    count_csr_basic = count_vec.fit_transform(df_tr_te['ac_app_list'])
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_vec_basic = tfidf_vec.fit_transform(df_tr_te['ac_app_list'])
    df_feature = scipy.sparse.csr_matrix(
        scipy.sparse.hstack([count_csr_basic, tfidf_vec_basic]))
    return df_feature


#模型训练
def lr_model(X, y):
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2021)
    kf = kf.split(X, y)
    print("开始训练LR...")
    stack_train = np.zeros((len(df_tr), number))
    stack_test = np.zeros((len(df_te), number))
    label_va = 0
    for i, (tr, va) in enumerate(kf):
        #print('stack:%d/%d' % ((i + 1), folds))
        clf = LogisticRegression(random_state=2021, C=8)
        clf.fit(X[tr], y[tr])
        label_va = clf.predict_proba(X[va])
        label_te = clf.predict_proba(test)
        #print('得分' + str(accuracy_score(y[va], clf.predict(X[va]))))
        stack_train[va] += label_va
        stack_test += label_te
    stack_test /= folds
    stack = np.vstack([stack_train, stack_test])
    df_stack = pd.DataFrame()
    for i in range(stack.shape[1]):
        df_stack['ac_tfidf_lr_clf_{}'.format(i)] = np.around(stack[:, i], 6)
    df_stack.to_csv('../user_data/tmp/ac_tfidf_lr_gender.csv',
                    index=None,
                    encoding='utf8')


def sgd_model(X, y):
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2021)
    kf = kf.split(X, y)
    print("开始训练SGD...")
    stack_train = np.zeros((len(df_tr), number))
    stack_test = np.zeros((len(df_te), number))
    label_va = 0
    for i, (tr, va) in enumerate(kf):
        #print('stack:%d/%d' % ((i + 1), folds))
        clf = SGDClassifier(random_state=2021, loss='log')
        clf.fit(X[tr], y[tr])
        label_va = clf.predict_proba(X[va])
        label_te = clf.predict_proba(test)
        #print('得分' + str(accuracy_score(y[va], clf.predict(X[va]))))
        stack_train[va] += label_va
        stack_test += label_te
    stack_test /= folds
    stack = np.vstack([stack_train, stack_test])
    df_stack = pd.DataFrame()
    for i in range(stack.shape[1]):
        df_stack['ac_tfidf_sgd_clf_{}'.format(i)] = np.around(stack[:, i], 6)

    df_stack.to_csv('../user_data/tmp/ac_tfidf_sgd_gender.csv',
                    index=None,
                    encoding='utf8')


def pac_model(X, y):
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2021)
    kf = kf.split(X, y)
    print("开始训练PAC...")
    stack_train = np.zeros((len(df_tr), number))
    stack_test = np.zeros((len(df_te), number))
    label_va = 0
    for i, (tr, va) in enumerate(kf):
        #print('stack:%d/%d' % ((i + 1), folds))
        clf = PassiveAggressiveClassifier(random_state=2021)
        clf.fit(X[tr], y[tr])
        label_va = clf._predict_proba_lr(X[va])
        label_te = clf._predict_proba_lr(test)
        #print('得分' + str(accuracy_score(y[va], clf.predict(X[va]))))
        stack_train[va] += label_va
        stack_test += label_te
    stack_test /= folds
    stack = np.vstack([stack_train, stack_test])
    df_stack = pd.DataFrame()
    for i in range(stack.shape[1]):
        df_stack['ac_tfidf_pac_clf_{}'.format(i)] = np.around(stack[:, i], 6)

    df_stack.to_csv('../user_data/tmp/ac_tfidf_pac_gender.csv',
                    index=None,
                    encoding='utf8')

def bnb_model(X, y):
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2021)
    kf = kf.split(X, y)
    print("开始训练BNB...")
    stack_train = np.zeros((len(df_tr), number))
    stack_test = np.zeros((len(df_te), number))
    label_va = 0
    for i, (tr, va) in enumerate(kf):
        #print('stack:%d/%d' % ((i + 1), folds))
        clf = BernoulliNB()
        clf.fit(X[tr], y[tr])
        label_va = clf.predict_proba(X[va])
        label_te = clf.predict_proba(test)
        #print('得分' + str(accuracy_score(y[va], clf.predict(X[va]))))
        stack_train[va] += label_va
        stack_test += label_te
    stack_test /= folds
    stack = np.vstack([stack_train, stack_test])
    df_stack = pd.DataFrame()
    for i in range(stack.shape[1]):
        df_stack['ac_tfidf_bnb_clf_{}'.format(i)] = np.around(stack[:, i], 6)

    df_stack.to_csv('../user_data/tmp/ac_tfidf_bnb_gender.csv',
                    index=None,
                    encoding='utf8')


def mnb_model(X, y):
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2021)
    kf = kf.split(X, y)
    print("开始训练MNB...")
    stack_train = np.zeros((len(df_tr), number))
    stack_test = np.zeros((len(df_te), number))
    label_va = 0
    for i, (tr, va) in enumerate(kf):
        #print('stack:%d/%d' % ((i + 1), folds))
        clf = MultinomialNB()
        clf.fit(X[tr], y[tr])
        label_va = clf.predict_proba(X[va])
        label_te = clf.predict_proba(test)
        #print('得分' + str(accuracy_score(y[va], clf.predict(X[va]))))
        stack_train[va] += label_va
        stack_test += label_te
    stack_test /= folds
    stack = np.vstack([stack_train, stack_test])
    df_stack = pd.DataFrame()
    for i in range(stack.shape[1]):
        df_stack['ac_tfidf_mnb_clf_{}'.format(i)] = np.around(stack[:, i], 6)

    df_stack.to_csv('../user_data/tmp/ac_tfidf_mnb_gender.csv',
                    index=None,
                    encoding='utf8')


def svc_model(X, y):
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2021)
    kf = kf.split(X, y)
    print('开始训练SVC...')
    stack_train = np.zeros((len(df_tr), number))
    stack_test = np.zeros((len(df_te), number))
    label_va = 0
    for i, (tr, va) in enumerate(kf):
        #print('stack:%d/%d' % ((i + 1), folds))
        clf = LinearSVC(random_state=2021)
        clf.fit(X[tr], y[tr])
        label_va = clf._predict_proba_lr(X[va])
        label_te = clf._predict_proba_lr(test)
        #print('得分' + str(accuracy_score(y[va], clf.predict(X[va]))))
        stack_train[va] += label_va
        stack_test += label_te
    stack_test /= folds
    stack = np.vstack([stack_train, stack_test])
    df_stack = pd.DataFrame()
    for i in range(stack.shape[1]):
        df_stack['ac_tfidf_svc_clf_{}'.format(i)] = np.around(stack[:, i], 6)

    df_stack.to_csv('../user_data/tmp/ac_tfidf_svc_gender.csv',
                    index=None,
                    encoding='utf8')


#聚类模型
def get_cluster(num_clusters):
    print('开始kmeans-' + str(num_clusters))
    name = 'kmeans'
    #print(name)
    model = KMeans(n_clusters=num_clusters,
                   max_iter=300,
                   n_init=1,
                   init='k-means++',
                   n_jobs=10,
                   random_state=2021)
    result = model.fit_predict(df_feature)
    kmeans_result[name + 'ac_word_' + str(num_clusters)] = result


if __name__ == '__main__':
    DATA_PATH = '../xfdata/'
    print('读取数据...')
    df_tr, df_te, df_tr_te = data_preprocess(DATA_PATH=DATA_PATH)

    print('开始特征工程...')
    df_feature = gen_features(df_tr_te)

    print("开始聚类...")
    kmeans_result = pd.DataFrame()
    get_cluster(5)
    get_cluster(10)
    get_cluster(19)
    get_cluster(30)
    kmeans_result.to_csv('../user_data/tmp/ac_cluster_tfidf_gender.csv',
                         index=False)

    print("开始模型训练...")
    tr_feature = df_feature[:len(df_tr)]
    label = df_tr['gender']
    te_feature = df_feature[len(df_tr):]
    X = tr_feature
    test = te_feature
    y = label
    folds = 5
    number = len(np.unique(label))

    lr_model(X, y)
    sgd_model(X, y)
    pac_model(X, y)
    bnb_model(X, y)
    mnb_model(X, y)
    svc_model(X, y)