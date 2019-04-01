import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import matplotlib as plt
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from scipy import sparse
from scipy import stats
import warnings
import re
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns',None)
pd.set_option('max_colwidth',100)
train= pd.read_csv('./data/jinnan_round1_train_20181227.csv', encoding = 'gb18030')
trainA=pd.read_csv('./data/jinnan_round1_testA_20181227.csv', encoding = 'gb18030')
trainB=pd.read_csv('./data/jinnan_round1_testB_20190121.csv', encoding = 'gb18030')
testA=pd.read_csv('./data/jinnan_round1_ansA_20190125.csv',names=['样本id', '收率'], encoding = 'gb18030')
testB=pd.read_csv('./data/jinnan_round1_ansB_20190125.csv', names=['样本id', '收率'],encoding = 'gb18030')
testC = pd.read_csv('./data/jinnan_round1_test_20190201.csv ',encoding = 'gb18030')
ansc=pd.read_csv('./data/jinnan_round1_ans_20190201.csv', names=['样本id', '收率'],encoding = 'gb18030')
opt = pd.read_csv('./data/optimize.csv',encoding = 'gb18030')
FuSai = pd.read_csv('./data/FuSai.csv', encoding = 'gb18030')
trainA=pd.concat([trainA,testA['收率']],axis=1)
trainB=pd.concat([trainB,testB['收率']],axis=1)
trainC=pd.concat([testC,ansc['收率']],axis=1)
def timeTranSecond(t):
    try:
        t, m, s = t.split(":")
    except:
        if t == '1900/1/9 7:00':
            return 7 * 3600 / 3600
        elif t == '1900/1/1 2:30':
            return (2 * 3600 + 30 * 60) / 3600
        elif t == -1:
            return -1
        else:
            return 0

    try:
        tm = (int(t) * 3600 + int(m) * 60 + int(s)) / 3600
    except:
        return (30 * 60) / 3600

    return tm
def getDuration(se):
    try:
        sh, sm, eh, em = re.findall(r"\d+\.?\d*", se)
    except:
        if se == -1:
            return -1
        else:
            return 0

    try:
        if int(sh) > int(eh):
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600 + 24
        else:
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600
    except:
        if se == '19:-20:05':
            return 1
        elif se == '15:00-1600':
            return 1

    return tm
train = pd.concat([train,trainA,trainC],axis=0,ignore_index=True)
train.drop(['B3', 'B13', 'A13', 'A18'], axis=1, inplace=True)
def main(train,test):
    test_ori = test
    test.drop(['B3', 'B13', 'A13', 'A18'], axis=1, inplace=True)
    good_cols = list(train.columns)
    for col in train.columns:
        rate = train[col].value_counts(normalize=True, dropna=False).values[0]
        if col not in['A23']:
         if rate > 0.9:
            good_cols.remove(col)
            print(col, rate)
    good_cols.append('A1')
    good_cols.append('A3')
    good_cols.append('A4')
    train = train[train['收率'] > 0.87]
    train = train[train['收率'] <=1]
    train = train[train['B14'] >= 350]
    train = train[train['B14'] <= 460]
    train = train[good_cols]
    good_cols.remove('收率')
    test = test[good_cols]
    target = train['收率']
    del train['收率']
    data = pd.concat([train,test],axis=0,ignore_index=True)
    data.loc[data['A25']=='1900/3/10 0:00', 'A25'] =70
    for f in data.columns:
        if f!='样本id':
         if f in ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7','A20', 'A28', 'B4', 'B9', 'B10', 'B11']:
            data[f]=data[f].fillna(0)
         else:
            counts = stats.mode(data[f].astype(float))[0][0]
            data[f] =data[f].fillna(counts)

    for f in ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']:
        try:
            data[f] = data[f].apply(timeTranSecond)
        except:
            print(f, '应该在前面被删除了！')

    for f in ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']:
        data[f] = data.apply(lambda df: getDuration(df[f]), axis=1)
    for f in ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']:
        data.loc[data[f] == 0, f] = stats.mode(data[f].astype(float))[0][0]
    # data['样本id'] = data['样本id'].apply(lambda x: int(x.split('_')[1]))
    data.drop(['样本id'], axis=1, inplace=True)
    categorical_columns = [f for f in data.columns if f not in ['样本id']]
    numerical_columns = [f for f in data.columns if f not in categorical_columns]
    data['A25']=pd.DataFrame(data['A25'],dtype=np.float)
    data['b14/a1_a3_a4_a19_b1_b12'] = data['B14']/(data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])
    data['A1_A3_A4/a1_a3_a4_a19_b1_b12'] = (data['A1']+data['A3']+data['A4'])/(data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])
    data['B10_B11']=(data['B12'])/(data['B10']+data['B11'])
    data['A11_A5']=data['A11']-data['A5']
    for f in range(len(data['A11_A5'])):
        if data['A11_A5'][f]<0:
            data['A11_A5'][f]=data['A11_A5'][f]+24
    data['A16_A11']=data['A16']-data['A11']
    for f in range(len(data['A16_A11'])):
        if data['A16_A11'][f]<0:
            data['A16_A11'][f]=data['A16_A11'][f]+24
    data['A26_A24'] = (data['A26'] - data['A24'])
    for f in range(len(data['A26_A24'])):
        if data['A26_A24'][f]<0:
            data['A26_A24'][f]=data['A26_A24'][f]+24
    data['A26_A24_A28']=data['A26_A24']/data['A28']
    data['A21_A22_shijian']=(data['A21']+data['A22'])/data['A26_A24']
    data['B7_B5']=(data['B7']-data['B5'])
    for f in range(len(data['B7_B5'])):
        if data['B7_B5'][f]<0:
            data['B7_B5'][f]=data['B7_B5'][f]+24
    data['B14/B7_B5']=data['B14']/data['B7_B5']
    # data['B11*B14'] = data['B11'] * data['B14']
    # numerical_columns.append('B11*B14')
    numerical_columns.append('b14/a1_a3_a4_a19_b1_b12')
    numerical_columns.append('A21_A22_shijian')
    for l in ['A1','A3','A4','A7','A5','A11','A9','A14','A16',
              'A21','A22','A20','A26','A24','A28','A23','B8','B6','A17']:
        data.drop([l],axis = 1, inplace = True)
    categorical_columns.append('B14/B7_B5')
    categorical_columns.append('A26_A24_A28')
    categorical_columns.append('A16_A11')
    categorical_columns.append('B10_B11')
    categorical_columns.append('A11_A5')
    for l in ['A1','A3','A4','A7','A5','A11','A9','A14','A16',
              'A21','A22','A20','A26','A24','A28','A23','B8','B6','A17']:
        categorical_columns.remove(l)
    for f in categorical_columns:
        data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
    train = data[:train.shape[0]]
    test  = data[train.shape[0]:]
    print(train.shape)
    print(test.shape)
    train['target'] = target
    train['intTarget'] = pd.cut(train['target'], 5, labels=False)
    train = pd.get_dummies(train, columns=['intTarget'])
    li = ['intTarget_0.0', 'intTarget_1.0', 'intTarget_2.0', 'intTarget_3.0', 'intTarget_4.0']
    mean_columns = []
    for f1 in categorical_columns:
        cate_rate = train[f1].value_counts(normalize=True, dropna=False).values[0]
        if cate_rate < 0.90:
            for f2 in li:
                col_name = 'B14_to_' + f1 + "_" + f2 + '_mean'
                mean_columns.append(col_name)
                order_label = train.groupby([f1])[f2].mean()
                train[col_name] = train['B14'].map(order_label)
                miss_rate = train[col_name].isnull().sum() * 100 / train[col_name].shape[0]
                if miss_rate > 0:
                    train = train.drop([col_name], axis=1)
                    mean_columns.remove(col_name)
                else:
                    test[col_name] = test['B14'].map(order_label)
    train.drop(li + ['target'], axis=1, inplace=True)
    print(train.shape)
    print(test.shape)
    X_train = train[mean_columns+numerical_columns].values
    X_test = test[mean_columns+numerical_columns].values
    enc = OneHotEncoder()
    for f in categorical_columns:
        enc.fit(data[f].values.reshape(-1, 1))
        X_train = sparse.hstack((X_train, enc.transform(train[f].values.reshape(-1, 1))), 'csr')
        X_test = sparse.hstack((X_test, enc.transform(test[f].values.reshape(-1, 1))), 'csr')
    print(X_train.shape)
    print(X_test.shape)
    y_train = target.values
    param = {'num_leaves': 120,
         'min_data_in_leaf': 30,
         'objective': 'regression',
         'max_depth': -1,
         'learning_rate': 0.05,
         "min_child_samples": 30,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l1": 0.1,
         "verbosity": -1}
    folds = KFold(n_splits=8, shuffle=True, random_state=2018)
    oof_lgb = np.zeros(len(train))
    predictions_lgb = np.zeros(len(test))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train,  y_train)):
        print("fold n°{}".format(fold_ + 1))
        trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
        val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])
        num_round = 10000
        clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=200,
                    early_stopping_rounds=100)
        oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)

        predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

    print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, target)))
    xgb_params = {'eta': 0.005, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
              'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4}

    folds = KFold(n_splits=5, shuffle=True, random_state=2018)
    oof_xgb = np.zeros(len(train))
    predictions_xgb = np.zeros(len(test))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("fold n°{}".format(fold_ + 1))
        trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
        val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

        watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
        clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                    verbose_eval=100, params=xgb_params)
        oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
        predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

    print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, target)))
    # # stacking
    train_stack = np.vstack([oof_lgb, oof_xgb]).transpose()
    test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()

    folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
    oof_stack = np.zeros(train_stack.shape[0])
    predictions = np.zeros(test_stack.shape[0])

    for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, y_train)):
        print("fold {}".format(fold_))
        trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
        val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values

        clf_3 = BayesianRidge()
        clf_3.fit(trn_data, trn_y)

        oof_stack[val_idx] = clf_3.predict(val_data)
        predictions += clf_3.predict(test_stack) / 10
    print("CV score: {:<8.8f}".format(mean_squared_error(target.values, oof_stack)))
    # print("CV score: {:<8.8f}".format(mean_squared_error(ansc['收率'], predictions_xgb)))
    # print(predictions_xgb)
    sub_df = pd.DataFrame({'a':test_ori['样本id'],'b':predictions})
    # sub_df['b']=  sub_df['b'].apply(lambda x: round(x, 3))
    return sub_df
ansOpt=main(train,opt)
ansFu=main(train,FuSai)
ansOpt.to_csv("submit_optimize.csv", index=False, header=None)
ansFu.to_csv("submit_FuSai.csv", index=False, header=None)