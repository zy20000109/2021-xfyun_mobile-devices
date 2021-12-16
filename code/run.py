# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgbm
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


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
    #df_tr_te = pd.concat([df_tr, df_te], axis=0, ignore_index=True)
    df_tr_te_app_events = pd.concat([df_tr_app_events, df_te_app_events],
                                    axis=0,
                                    ignore_index=True)
    #提取标签数量
    df_tr_te_app_events['tag_list_len'] = df_tr_te_app_events[
        'tag_list'].apply(lambda x: x.count(',') + 1)
    #设备聚合特征
    agg_dic = {
        "event_id": ['count', 'nunique'],
        "app_id": ['nunique'],
        "is_installed": [np.sum],
        "is_active": [np.mean, np.sum],
        "date": [np.max, np.min, 'nunique'],
        "tag_list_len": [np.mean, np.std]
    }

    df_device_features = df_tr_te_app_events.groupby('device_id').agg(
        agg_dic).reset_index()

    fea_names = ['_'.join(c) for c in df_device_features.columns]
    df_device_features.columns = fea_names
    df_device_features.rename(columns={'device_id_': 'device_id'},
                              inplace=True)
    #部分特征组合
    df_device_features['event_id_count_nunique_ratio'] = df_device_features[
        'event_id_count'] / df_device_features['event_id_nunique']
    df_device_features['tag_list_len_mean_div_std'] = df_device_features[
        'tag_list_len_mean'] / df_device_features['tag_list_len_std']
    #以上特征合并
    df_tr = df_tr.merge(df_device_features, on='device_id', how='left')
    df_te = df_te.merge(df_device_features, on='device_id', how='left')
    del df_device_features
    return df_tr, df_te, df_tr_te_app_events


def get_max_label(row):
    row_name = list(row['app_id'])
    row_diff_time = list(row['app_cnt'])
    return row_name[np.argmax(row_diff_time)]


#生成特征
def gen_features(df_tr, df_te, df_tr_te_app_events):
    tri_day_cnt = df_tr_te_app_events[(df_tr_te_app_events['date'] >= 1)
                                      & (df_tr_te_app_events['date'] <= 3)]
    tri_day_cnt = tri_day_cnt.groupby('device_id').size().reset_index()
    tri_day_cnt.columns = ['device_id', 'tri_day_event_cnt']
    df_tr = df_tr.merge(tri_day_cnt, on='device_id', how='left')
    df_te = df_te.merge(tri_day_cnt, on='device_id', how='left')
    del tri_day_cnt
    temp2 = df_tr_te_app_events.groupby(['device_id',
                                         'app_id']).size().reset_index()
    temp2.columns = ['device_id', 'app_id', 'app_cnt']
    temp2 = temp2.groupby('device_id').apply(
        lambda row: get_max_label(row)).reset_index()
    temp2.columns = ['device_id', 'max_app_id']
    df_tr = df_tr.merge(temp2, on='device_id', how='left')
    df_te = df_te.merge(temp2, on='device_id', how='left')
    del temp2
    return df_tr, df_te, df_tr_te_app_events


#target encoder
def kfold_stats_feature(train, test, feats, k):
    folds = StratifiedKFold(n_splits=k, shuffle=True,
                            random_state=2021)  # 这里最好和后面模型的K折交叉验证保持一致

    train['fold'] = None
    for fold_, (trn_idx,
                val_idx) in enumerate(folds.split(train, train[label_gender])):
        train.loc[val_idx, 'fold'] = fold_

    kfold_features = []
    for feat in feats:
        nums_columns = ['gender']
        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            kfold_features.append(colname)
            train[colname] = None
            for fold_, (trn_idx, val_idx) in enumerate(
                    folds.split(train, train[label_gender])):
                tmp_trn = train.iloc[trn_idx]
                order_label = tmp_trn.groupby([feat])[f].mean()
                tmp = train.loc[train.fold == fold_, [feat]]
                train.loc[train.fold == fold_,
                          colname] = tmp[feat].map(order_label)
                # fillna
                global_mean = train[f].mean()
                train.loc[train.fold == fold_,
                          colname] = train.loc[train.fold == fold_,
                                               colname].fillna(global_mean)
            train[colname] = train[colname].astype(float)

        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            test[colname] = None
            order_label = train.groupby([feat])[f].mean()
            test[colname] = test[feat].map(order_label)
            # fillna
            global_mean = train[f].mean()
            test[colname] = test[colname].fillna(global_mean)
            test[colname] = test[colname].astype(float)
    del train['fold']
    return train, test


#生成topN app特征
def gen_top_app_features(df_tr, df_te, df_tr_te_app_events):
    app_info_df = df_tr_te_app_events[['device_id', 'app_id']]
    app_cnt_df = app_info_df.groupby(
        'app_id')['device_id'].count().reset_index()
    app_cnt_df.columns = ['app_id', 'app_cnt']
    app_top100_set = app_cnt_df.sort_values(
        by='app_cnt', ascending=False)['app_id'][:100].tolist()
    app_list_df = app_info_df.groupby(
        ['device_id']).apply(lambda x: x['app_id'].tolist()).reset_index()
    app_list_df.columns = ['device_id', 'app_list']
    top100_app_dict = {}
    for i, value in enumerate(app_list_df['app_list']):
        index = 0
        for j in app_list_df['app_list'][i]:
            if j in app_top100_set:
                index += 1
        top100_app_dict[app_list_df['device_id'][i]] = index
    top100_app_df = pd.DataFrame.from_dict(top100_app_dict,
                                           orient='index',
                                           columns=['top100_app_cnt'])
    app_top50_set = app_cnt_df.sort_values(
        by='app_cnt', ascending=False)['app_id'][:50].tolist()
    top50_app_dict = {}
    for i, value in enumerate(app_list_df['app_list']):
        index = 0
        for j in app_list_df['app_list'][i]:
            if j in app_top50_set:
                index += 1
        top50_app_dict[app_list_df['device_id'][i]] = index
    top50_app_dict
    top50_app_df = pd.DataFrame.from_dict(top50_app_dict,
                                          orient='index',
                                          columns=['top50_app_cnt'])
    app_top10_set = app_cnt_df.sort_values(
        by='app_cnt', ascending=False)['app_id'][:10].tolist()
    top10_app_dict = {}
    for i, value in enumerate(app_list_df['app_list']):
        index = 0
        for j in app_list_df['app_list'][i]:
            if j in app_top10_set:
                index += 1
        top10_app_dict[app_list_df['device_id'][i]] = index
    top10_app_df = pd.DataFrame.from_dict(top10_app_dict,
                                          orient='index',
                                          columns=['top10_app_cnt'])
    top100_app_df = top100_app_df.reset_index()
    top100_app_df.columns = ['device_id', 'top100_app_cnt']
    top50_app_df = top50_app_df.reset_index()
    top50_app_df.columns = ['device_id', 'top50_app_cnt']
    top10_app_df = top10_app_df.reset_index()
    top10_app_df.columns = ['device_id', 'top10_app_cnt']
    df_tr = df_tr.merge(top100_app_df, on='device_id', how='left')
    df_te = df_te.merge(top100_app_df, on='device_id', how='left')
    df_tr = df_tr.merge(top50_app_df, on='device_id', how='left')
    df_te = df_te.merge(top50_app_df, on='device_id', how='left')
    df_tr = df_tr.merge(top10_app_df, on='device_id', how='left')
    df_te = df_te.merge(top10_app_df, on='device_id', how='left')
    del top100_app_dict, top100_app_df, top50_app_dict, top50_app_df, top10_app_dict, top10_app_df
    return df_tr, df_te, df_tr_te_app_events


def gen_device_tfidf_features(df, value, char):
    df[value] = df[value].astype(str)
    df[value].fillna('-1', inplace=True)
    group_df = df.groupby(['device_id'
                           ]).apply(lambda x: x[value].tolist()).reset_index()
    group_df.columns = ['device_id', 'list']
    group_df['list'] = group_df['list'].apply(lambda x: ','.join(x))
    enc_vec = TfidfVectorizer()
    tfidf_vec = enc_vec.fit_transform(group_df['list'])
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2021)
    vec_svd = svd_enc.fit_transform(tfidf_vec)
    vec_svd = pd.DataFrame(vec_svd)
    vec_svd.columns = [
        'svd_tfidf_{}_{}_{}'.format(value, i, char) for i in range(10)
    ]
    group_df = pd.concat([group_df, vec_svd], axis=1)
    del group_df['list']
    return group_df


def gen_device_countvec_features(df, value, char):
    df[value] = df[value].astype(str)
    df[value].fillna('-1', inplace=True)
    group_df = df.groupby(['device_id'
                           ]).apply(lambda x: x[value].tolist()).reset_index()
    group_df.columns = ['device_id', 'list']
    group_df['list'] = group_df['list'].apply(lambda x: ','.join(x))
    enc_vec = CountVectorizer()
    tfidf_vec = enc_vec.fit_transform(group_df['list'])
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2021)
    vec_svd = svd_enc.fit_transform(tfidf_vec)
    vec_svd = pd.DataFrame(vec_svd)
    vec_svd.columns = [
        'svd_countvec_{}_{}_{}'.format(value, i, char) for i in range(10)
    ]
    group_df = pd.concat([group_df, vec_svd], axis=1)
    del group_df['list']
    return group_df


def model_sex(train, test, label):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
    gender_probs = np.zeros(train.shape[0])
    gender_preds = 0
    offline_score = []
    gender_feature_importance_df = pd.DataFrame()
    lgb_params = {
        "objective": "binary",
        "metric": "binary_error",
        "boosting_type": "gbdt",
        'learning_rate': 0.01,
        'colsample_bytree': 0.95,
    }

    for i, (train_index, val_index) in enumerate(folds.split(train, label)):
        train_y, val_y = label[train_index], label[val_index]
        train_X, val_X = train.iloc[train_index, :], train.iloc[val_index, :]

        dtrain = lgbm.Dataset(train_X, label=train_y)
        dval = lgbm.Dataset(val_X, label=val_y)
        lgb_model = lgbm.train(params=lgb_params,
                               train_set=dtrain,
                               num_boost_round=5000,
                               valid_sets=[dval],
                               verbose_eval=100,
                               early_stopping_rounds=100)
        gender_probs[val_index] = lgb_model.predict(
            val_X, num_iteration=lgb_model.best_iteration)
        offline_score.append(lgb_model.best_score['valid_0']['binary_error'])
        gender_preds += lgb_model.predict(
            test, num_iteration=lgb_model.best_iteration) / folds.n_splits
        print(offline_score)
        gender_fold_importance_df = pd.DataFrame()
        gender_fold_importance_df["feature"] = tr_features
        gender_fold_importance_df['importance'] = lgb_model.feature_importance(
            importance_type='gain')
        gender_fold_importance_df["fold"] = i + 1
        gender_feature_importance_df = pd.concat(
            [gender_feature_importance_df, gender_fold_importance_df], axis=0)
    print('OOF-MEAN-binary_error:%.6f, OOF-STD-binary_error:%.6f' %
          (np.mean(offline_score), np.std(offline_score)))
    print('feature importance:')
    print(
        gender_feature_importance_df.groupby([
            'feature'
        ])['importance'].mean().sort_values(ascending=False).head(10))
    return gender_preds


def model_age(train_age, test_age, label_age):
    def feval_lgb_Age(preds, lgbm_train):
        labels = lgbm_train.get_label()
        return 'Age Error', round(
            1.0 / (1.0 + mean_absolute_error(y_true=labels, y_pred=preds)),
            7), True

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
    age_probs = np.zeros(train_age.shape[0])
    age_preds = 0
    offline_score = []
    age_feature_importance_df = pd.DataFrame()
    lgb_params = {
        "objective": "mae",
        "boosting_type": "gbdt",
        'learning_rate': 0.01,
        'colsample_bytree': 0.95,
    }

    for i, (train_index,
            val_index) in enumerate(folds.split(train_age, label_age)):
        train_y, val_y = label_age[train_index], label_age[val_index]
        train_X, val_X = train_age.iloc[train_index, :], train_age.iloc[
            val_index, :]

        dtrain = lgbm.Dataset(train_X, label=train_y)
        dval = lgbm.Dataset(val_X, label=val_y)
        lgb_model = lgbm.train(params=lgb_params,
                               train_set=dtrain,
                               num_boost_round=5000,
                               valid_sets=[dval],
                               verbose_eval=100,
                               feval=feval_lgb_Age,
                               early_stopping_rounds=100)
        age_probs[val_index] = lgb_model.predict(
            val_X, num_iteration=lgb_model.best_iteration)
        offline_score.append(lgb_model.best_score['valid_0']['Age Error'])
        age_preds += lgb_model.predict(
            test_age, num_iteration=lgb_model.best_iteration) / folds.n_splits
        print(offline_score)
        age_fold_importance_df = pd.DataFrame()
        age_fold_importance_df["feature"] = tr_features_age
        age_fold_importance_df['importance'] = lgb_model.feature_importance(
            importance_type='gain')
        age_fold_importance_df["fold"] = i + 1
        age_feature_importance_df = pd.concat(
            [age_feature_importance_df, age_fold_importance_df], axis=0)
    print('OOF-MEAN-mae:%.6f, OOF-STD-mae:%.6f' %
          (np.mean(offline_score), np.std(offline_score)))
    print('feature importance:')
    print(
        age_feature_importance_df.groupby([
            'feature'
        ])['importance'].mean().sort_values(ascending=False).head(10))
    return age_preds


if __name__ == '__main__':
    DATA_PATH = '../xfdata/'
    print('读取数据...')
    df_tr, df_te, df_tr_te_app_events = data_preprocess(
        DATA_PATH=DATA_PATH)

    print('开始特征工程...')
    df_tr, df_te, df_tr_te_app_events = gen_features(df_tr, df_te, df_tr_te_app_events)

    label_gender = 'gender'
    target_encode_cols = ['phone_brand', 'device_model']
    df_tr, df_te = kfold_stats_feature(df_tr, df_te, target_encode_cols, 5)

    df_tr, df_te, df_tr_te_app_events = gen_top_app_features(
        df_tr, df_te, df_tr_te_app_events)

    #合并训练集和测试集
    data = pd.concat([df_tr, df_te], ignore_index=True, axis=0)

    #读取stacking文件
    df_bnb = pd.read_csv('../user_data/tmp/tfidf_bnb_gender.csv')
    df_lr = pd.read_csv('../user_data/tmp/tfidf_lr_gender.csv')
    df_mnb = pd.read_csv('../user_data/tmp/tfidf_mnb_gender.csv')
    df_pac = pd.read_csv('../user_data/tmp/tfidf_pac_gender.csv')
    df_ridge = pd.read_csv('../user_data/tmp/tfidf_ridge_gender.csv')
    df_sgd = pd.read_csv('../user_data/tmp/tfidf_sgd_gender.csv')
    df_svc = pd.read_csv('../user_data/tmp/tfidf_svc_gender.csv')
    df_cluster = pd.read_csv('../user_data/tmp/cluster_tfidf_gender.csv')
    data = pd.concat([
        data, df_bnb, df_lr, df_mnb, df_pac, df_ridge, df_sgd, df_svc,
        df_cluster
    ],
                     axis=1)

    df_ac_cluster = pd.read_csv('../user_data/tmp/ac_cluster_tfidf_gender.csv')
    df_ac_bnb = pd.read_csv('../user_data/tmp/ac_tfidf_bnb_gender.csv')
    df_ac_lr = pd.read_csv('../user_data/tmp/ac_tfidf_lr_gender.csv')
    df_ac_mnb = pd.read_csv('../user_data/tmp/ac_tfidf_mnb_gender.csv')
    df_ac_pac = pd.read_csv('../user_data/tmp/ac_tfidf_pac_gender.csv')
    df_ac_sgd = pd.read_csv('../user_data/tmp/ac_tfidf_sgd_gender.csv')
    df_ac_svc = pd.read_csv('../user_data/tmp/ac_tfidf_svc_gender.csv')
    data = pd.concat([
        data, df_ac_bnb, df_ac_cluster, df_ac_lr, df_ac_mnb, df_ac_pac,
        df_ac_sgd, df_ac_svc
    ],
                     axis=1)

    df_nn = pd.read_csv('../user_data/tmp/stack_best_nn.csv')
    data = pd.concat([data, df_nn], axis=1)

    print("开始性别模型训练...")
    tr_features = [
        f for f in data.columns if f not in ['gender', 'age', 'device_id']
    ]
    print("model_sex特征个数：", len(tr_features))
    train = data[:len(df_tr)][tr_features]
    test = data[len(df_tr):][tr_features]
    label = data[:len(df_tr)][label_gender]

    gender_preds = model_sex(train, test, label)
    df_submit = df_te[['device_id']].copy()
    df_submit['gender'] = (gender_preds >= 0.5) + 0
    df_submit['gender'] = df_submit['gender'].astype(int)

    print("开始准备年龄训练特征...")
    df_te_copy = df_te.copy()
    df_te_copy['gender'] = df_submit['gender']
    data_age = pd.concat([df_tr, df_te_copy], ignore_index=True, axis=0)

    #读取stacking文件
    df_age_par = pd.read_csv('../user_data/tmp/age_par.csv')
    df_age_par.columns = ['age_par']
    df_age_ridge = pd.read_csv('../user_data/tmp/age_ridge.csv')
    df_age_ridge.columns = ['age_ridge']
    df_age_svf = pd.read_csv('../user_data/tmp/age_svr.csv')
    df_age_svf.columns = ['age_svf']

    data_age = pd.concat([data_age, df_age_par, df_age_ridge, df_age_svf],
                         axis=1)

    #开始提取SVD特征
    app_id_tfidf = gen_device_tfidf_features(df=df_tr_te_app_events,
                                             value='app_id',
                                             char='all')
    app_id_countvec = gen_device_countvec_features(df=df_tr_te_app_events,
                                                   value='app_id',
                                                   char='all')
    tag_list_tfidf = gen_device_tfidf_features(df=df_tr_te_app_events,
                                               value='tag_list',
                                               char='all')
    tag_list_countvec = gen_device_countvec_features(df=df_tr_te_app_events,
                                                     value='tag_list',
                                                     char='all')

    data_age = data_age.merge(app_id_tfidf, on=['device_id'], how='left')
    data_age = data_age.merge(app_id_countvec, on=['device_id'], how='left')
    data_age = data_age.merge(tag_list_tfidf, on=['device_id'], how='left')
    data_age = data_age.merge(tag_list_countvec, on=['device_id'], how='left')
    del app_id_tfidf, app_id_countvec, tag_list_tfidf, tag_list_countvec

    print("开始年龄模型训练...")
    label_of_age = 'age'
    tr_features_age = [
        f for f in data_age.columns if f not in ['age', 'device_id']
    ]
    print("model_age特征个数：", len(tr_features_age))

    train_age = data_age[:len(df_tr)][tr_features_age]
    test_age = data_age[len(df_tr):][tr_features_age]
    label_age = data_age[:len(df_tr)][label_of_age]

    age_preds = model_age(train_age, test_age, label_age)
    print("开始生成完整预测文件...")
    df_submit['age'] = age_preds
    sub_df = pd.read_csv("../xfdata/test.csv")
    sub_df = sub_df[['device_id']]
    sub_df = sub_df.merge(df_submit, on='device_id', how='left')
    sub_df.to_csv('../prediction_result/result.csv', index=None)