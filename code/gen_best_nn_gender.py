# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from gensim.models import Word2Vec
from keras.layers import *
from keras.models import *
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import *
from keras.layers.advanced_activations import LeakyReLU, PReLU
import keras.backend as K
from keras.optimizers import *
from keras import utils as np_utils


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
    df_app_list.columns = ['device_id', 'apps']
    df_app_list['app_list'] = df_app_list['apps'].apply(lambda x: x.split(','))
    #app和原数据聚合
    df_tr_te = df_tr_te.merge(df_app_list, on='device_id', how='left')
    df_tr = df_tr.merge(df_app_list, on='device_id', how='left')
    df_te = df_te.merge(df_app_list, on='device_id', how='left')
    return df_tr, df_te, df_tr_te, df_app_list


#生成特征
def gen_features(df):
    embed_size = 128
    fastmodel = Word2Vec(list(df_app_list['app_list']),
                         size=embed_size,
                         window=4,
                         min_count=3,
                         negative=2,
                         sg=1,
                         sample=0.002,
                         hs=1,
                         workers=4)
    embedding_fast = pd.DataFrame(
        [fastmodel[word] for word in (fastmodel.wv.vocab)])
    embedding_fast['app'] = list(fastmodel.wv.vocab)
    embedding_fast.columns = ["fdim_%s" % str(i)
                              for i in range(embed_size)] + ["app"]
    tokenizer = Tokenizer(lower=False, char_level=False, split=',')
    tokenizer.fit_on_texts(list(df_app_list['apps']))
    X_seq = tokenizer.texts_to_sequences(df_tr['apps'])
    X_test_seq = tokenizer.texts_to_sequences(df_te['apps'])
    maxlen = 50
    X = pad_sequences(X_seq, maxlen=maxlen, value=0)
    X_test = pad_sequences(X_test_seq, maxlen=maxlen, value=0)
    Y_sex = df_tr['gender']

    max_feaures = 35001
    embedding_matrix = np.zeros((max_feaures, embed_size))
    for word in tokenizer.word_index:
        if word not in fastmodel.wv.vocab:
            continue
        embedding_matrix[tokenizer.word_index[word]] = fastmodel[word]
    return X, X_test, Y_sex, embedding_matrix


def model_conv1D(embedding_matrix):
    maxlen = 50
    K.clear_session()
    emb_layer = Embedding(input_dim=embedding_matrix.shape[0],
                          output_dim=embedding_matrix.shape[1],
                          weights=[embedding_matrix],
                          input_length=maxlen,
                          trainable=False)
    lstm_layer = Bidirectional(
        GRU(128, recurrent_dropout=0.15, dropout=0.15, return_sequences=True))

    conv1 = Conv1D(
        filters=128,
        kernel_size=1,
        padding='same',
        activation='relu',
    )
    conv2 = Conv1D(
        filters=64,
        kernel_size=2,
        padding='same',
        activation='relu',
    )
    conv3 = Conv1D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu',
    )
    conv5 = Conv1D(
        filters=32,
        kernel_size=5,
        padding='same',
        activation='relu',
    )

    seq = Input(shape=(maxlen, ))
    emb = emb_layer(seq)
    lstm = lstm_layer(emb)
    conv1a = conv1(lstm)
    gap1a = GlobalAveragePooling1D()(conv1a)
    gmp1a = GlobalMaxPool1D()(conv1a)

    conv2a = conv2(lstm)
    gap2a = GlobalAveragePooling1D()(conv2a)
    gmp2a = GlobalMaxPool1D()(conv2a)

    conv3a = conv3(lstm)
    gap3a = GlobalAveragePooling1D()(conv3a)
    gmp3a = GlobalMaxPooling1D()(conv3a)

    conv5a = conv5(lstm)
    gap5a = GlobalAveragePooling1D()(conv5a)
    gmp5a = GlobalMaxPooling1D()(conv5a)
    merge1 = concatenate([gmp1a, gmp2a, gmp3a, gmp5a])

    x = Dropout(0.3)(merge1)
    x = BatchNormalization()(x)
    x = Dense(
        200,
        activation='relu',
    )(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    x = Dense(
        200,
        activation='relu',
    )(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    x = Dense(
        200,
        activation='relu',
    )(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization()(x)
    pred = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[seq], outputs=pred)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

    return model


def model_train(X, Y_sex, X_test, embedding_matrix):
    kfold = StratifiedKFold(n_splits=5, random_state=2021, shuffle=True)
    sub = np.zeros((X_test.shape[0], ))
    oof_pred = np.zeros((X.shape[0], 1))
    score = []
    count = 0
    for i, (train_index, test_index) in enumerate(kfold.split(X, Y_sex)):
        print("FOLD | ", count + 1)
        filepath = "../user_data//tmp/sex_weights_best_%d.h5" % count
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.8,
                                      patience=2,
                                      min_lr=0.0001,
                                      verbose=1)
        earlystopping = EarlyStopping(monitor='val_loss',
                                      min_delta=0.0001,
                                      patience=6,
                                      verbose=1,
                                      mode='auto')
        callbacks = [checkpoint, reduce_lr, earlystopping]

        model_sex = model_conv1D(embedding_matrix)
        X_tr, X_vl, y_tr, y_vl = X[train_index], X[test_index], Y_sex[
            train_index], Y_sex[test_index]
        hist = model_sex.fit(X_tr,
                             y_tr,
                             batch_size=256,
                             epochs=10,
                             validation_data=(X_vl, y_vl),
                             callbacks=callbacks,
                             verbose=1,
                             shuffle=True)
        model_sex.load_weights(filepath)
        sub += np.squeeze(model_sex.predict(X_test)) / kfold.n_splits
        oof_pred[test_index] = model_sex.predict(X_vl)
        score.append(np.min(hist.history['val_loss']))
        count += 1
    return oof_pred, sub


if __name__ == '__main__':
    DATA_PATH = '../xfdata/'
    print('读取数据...')
    df_tr, df_te, df_tr_te, df_app_list = data_preprocess(DATA_PATH=DATA_PATH)

    print('开始特征工程...')
    X, X_test, Y_sex, embedding_matrix = gen_features(df_tr_te)

    print("开始模型训练...")
    oof_pred, sub = model_train(X, Y_sex, X_test, embedding_matrix)
    oof_pred = pd.DataFrame(oof_pred, columns=['nn_sex_1'])
    sub = pd.DataFrame(sub, columns=['nn_sex_1'])
    res = pd.concat([oof_pred, sub])
    res['nn_sex_0'] = 1 - res['nn_sex_1']
    res.to_csv('../user_data/tmp/stack_best_nn.csv', index=None, encoding='utf8')