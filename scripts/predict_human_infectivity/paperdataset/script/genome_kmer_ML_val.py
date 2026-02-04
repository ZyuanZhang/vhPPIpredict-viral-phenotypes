import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, roc_curve
import os
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def kmer_rf_val_paperdataset():
    # 1. 读取数据
    for k in [3,4,5]:
        dt_mapping_0 = pd.read_csv("../data/AllDataIdMapping.csv", sep="\t")
        dt_mapping_0 = dt_mapping_0[dt_mapping_0["belong"]=="861paper"].reset_index(drop=True)
        target_taxid = list(dt_mapping_0["Taxid"])

        df_mapping = pd.read_csv('../data/AllDataIdMapping.csv', sep='\t')
        df_kmer = pd.read_csv(f'../data/allvirusGenome{k}Kmer.csv').rename(columns={'Unnamed: 0': 'Taxid'})
        df = pd.merge(df_mapping[['Taxid', 'Label']], df_kmer, on='Taxid')
        df = df[df["Taxid"].isin(target_taxid)].reset_index(drop=True)
        df.set_index('Taxid', inplace=True)
        print(df.shape)
    
        # 2. 准备特征和标签
        X = df.drop(columns=['Label']).values
        y = df['Label'].values
    
        # 3. 定义评价指标记录器
        metrics_list = {
          'accuracy': [],
          'precision': [],
          'recall': [],
          'f1': [],
          'auroc': [],
          'auprc':[]
        }
    
        # 4. 定义划分器
        sss = StratifiedShuffleSplit(n_splits=100, test_size=0.15, random_state=42)
    
        # 5. 网格搜索参数 - RandomForest
        param_grid = {
          'n_estimators': [50, 100, 200, 300, 500],
          'max_depth': [None, 10, 15, 20, 25],
        }
    
        # 6. 进行100次划分 + 训练 + 评估
        for train_val_index, test_index in sss.split(X, y):
            # 15% 测试集
            X_train_val, X_test = X[train_val_index], X[test_index]
            y_train_val, y_test = y[train_val_index], y[test_index]
    
            # 从 85% 的训练+验证集中划出 17.65% ≈ 15% 的验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=0.1765,  # 15 / 85
                stratify=y_train_val,
                random_state=42
            )
    
            # 网格搜索
            grid = GridSearchCV(RandomForestClassifier(
                random_state=train_val_index[0]),
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=32)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
    
            # 预测
            y_pred = best_model.predict(X_val)
            y_prob = best_model.predict_proba(X_val)[:, 1]
    
            # 记录评价指标
            metrics_list['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics_list['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            metrics_list['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            metrics_list['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            #metrics_list['auroc'].append(roc_auc_score(y_val, y_prob))
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            metrics_list['auroc'].append(auc(fpr, tpr))
            precision, recall, _ = precision_recall_curve(y_val, y_prob, pos_label=1)
            metrics_list['auprc'].append(auc(recall, precision))
    
        # 7. 保存结果
        result_df = pd.DataFrame(metrics_list)
        summary = result_df.describe().loc[['mean', 'std']]
    
        print(summary)
    
        # 保存文件
        os.makedirs("../result/val/genome", exist_ok=True)
        top_num = X.shape[1]  # 特征数
        result_df.to_csv(f"../result/val/genome/rf_result_{k}mer.csv", index=False)
        summary.to_csv(f"../result/val/genome/rf_summary_{k}mer.csv")



def kmer_svm_val_paperdataset():
    # 1. 读取数据
    for k in [3, 4, 5]:
        dt_mapping_0 = pd.read_csv("../data/AllDataIdMapping.csv", sep="\t")
        dt_mapping_0 = dt_mapping_0[dt_mapping_0["belong"]=="861paper"].reset_index(drop=True)
        target_taxid = list(dt_mapping_0["Taxid"])

        df_mapping = pd.read_csv('../data/AllDataIdMapping.csv', sep='\t')
        df_kmer = pd.read_csv(f'../data/allvirusGenome{k}Kmer.csv').rename(columns={'Unnamed: 0': 'Taxid'})
        df = pd.merge(df_mapping[['Taxid', 'Label']], df_kmer, on='Taxid')
        df = df[df["Taxid"].isin(target_taxid)].reset_index(drop=True)
        df.set_index('Taxid', inplace=True)
        print(df.shape)
    
        # 2. 准备特征和标签
        X = df.drop(columns=['Label']).values
        y = df['Label'].values
    
        # 3. 定义评价指标记录器
        metrics_list = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auroc': [],
            'auprc': []
        }
    
        # 4. 定义划分器
        sss = StratifiedShuffleSplit(n_splits=100, test_size=0.15, random_state=42)
    
        # 5. 网格搜索参数 - SVM
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    
        # 6. 进行100次划分 + 训练 + 评估
        for train_val_index, test_index in sss.split(X, y):
            # 15% 测试集
            X_train_val, X_test = X[train_val_index], X[test_index]
            y_train_val, y_test = y[train_val_index], y[test_index]
    
            # 从 85% 的训练+验证集中划出 17.65% ≈ 15% 的验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=0.1765,  # 15 / 85
                stratify=y_train_val,
                random_state=42
            )
    
            # SVM对特征尺度敏感，需要进行标准化
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
    
            # 网格搜索
            grid = GridSearchCV(SVC(probability=True, random_state=train_val_index[0]),  # probability=True用于支持predict_proba
                                param_grid,
                                cv=5,
                                scoring='roc_auc',
                                n_jobs=32)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
    
            # 预测
            y_pred = best_model.predict(X_val)
            y_prob = best_model.predict_proba(X_val)[:, 1]  # 获取正类的概率
    
            # 记录评价指标
            metrics_list['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics_list['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            metrics_list['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            metrics_list['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            #metrics_list['auc'].append(roc_auc_score(y_val, y_prob))
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            metrics_list['auroc'].append(auc(fpr, tpr))
            precision, recall, _ = precision_recall_curve(y_val, y_prob, pos_label=1)
            metrics_list['auprc'].append(auc(recall, precision))
    
        # 7. 保存结果
        result_df = pd.DataFrame(metrics_list)
        summary = result_df.describe().loc[['mean', 'std']]
    
        print(summary)
    
        # 保存文件
        os.makedirs("../result/val/genome", exist_ok=True)
        result_df.to_csv(f"../result/val/genome/svm_result_{k}mer.csv", index=False)
        summary.to_csv(f"../result/val/genome/svm_summary_{k}mer.csv")



def kmer_xgboost_val_paperdataset():
    for k in [3,4,5]:
        dt_mapping_0 = pd.read_csv("../data/AllDataIdMapping.csv", sep="\t")
        dt_mapping_0 = dt_mapping_0[dt_mapping_0["belong"]=="861paper"].reset_index(drop=True)
        target_taxid = list(dt_mapping_0["Taxid"])

        df_mapping = pd.read_csv('../data/AllDataIdMapping.csv', sep='\t')
        df_kmer = pd.read_csv(f'../data/allvirusGenome{k}Kmer.csv').rename(columns={'Unnamed: 0': 'Taxid'})
        df = pd.merge(df_mapping[['Taxid', 'Label']], df_kmer, on='Taxid')
        df = df[df["Taxid"].isin(target_taxid)].reset_index(drop=True)
        df.set_index('Taxid', inplace=True)
        print(df.shape)
    
        # 2. 准备特征和标签
        X = df.drop(columns=['Label']).values
        y = df['Label'].values
    
        # 3. 定义评价指标记录器
        metrics_list = {
          'accuracy': [],
          'precision': [],
          'recall': [],
          'f1': [],
          'auroc': [],
          'auprc': []
        }
    
        # 4. 定义划分器
        sss = StratifiedShuffleSplit(n_splits=100, test_size=0.15, random_state=42)
    
        # 5. 网格搜索参数 - XGBoost
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1],
            'colsample_bytree': [0.8, 1]
        }
    
        # 6. 进行100次划分 + 训练 + 评估
        for train_val_index, test_index in sss.split(X, y):
            # 15% 测试集
            X_train_val, X_test = X[train_val_index], X[test_index]
            y_train_val, y_test = y[train_val_index], y[test_index]
    
            # 从 85% 的训练+验证集中划出 17.65% ≈ 15% 的验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=0.1765,  # 15 / 85
                stratify=y_train_val,
                random_state=42
            )
    
            # 网格搜索
            grid = GridSearchCV(XGBClassifier(random_state=train_val_index[0],
                                              eval_metric='logloss',
                                              use_label_encoder=False),
                                param_grid,
                                cv=5,
                                scoring='roc_auc',
                                n_jobs=64)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
    
            # 预测
            y_pred = best_model.predict(X_val)
            y_prob = best_model.predict_proba(X_val)[:, 1]
    
            # 记录评价指标
            metrics_list['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics_list['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            metrics_list['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            metrics_list['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            #metrics_list['auc'].append(roc_auc_score(y_val, y_prob))
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            metrics_list['auroc'].append(auc(fpr, tpr))
            precision, recall, _ = precision_recall_curve(y_val, y_prob, pos_label=1)
            metrics_list['auprc'].append(auc(recall, precision))
    
        # 7. 保存结果
        result_df = pd.DataFrame(metrics_list)
        summary = result_df.describe().loc[['mean', 'std']]
    
        print(summary)
    
        # 保存文件
        os.makedirs("../result/val/genome", exist_ok=True)
        result_df.to_csv(f"../result/val/genome/xgboost_result_{k}mer.csv", index=False)
        summary.to_csv(f"../result/val/genome/xgboost_summary_{k}mer.csv")


def kmer_knn_val_paperdataset():
    # 1. 读取数据
    for k in [3,4,5]:
        dt_mapping_0 = pd.read_csv("../data/AllDataIdMapping.csv", sep="\t")
        dt_mapping_0 = dt_mapping_0[dt_mapping_0["belong"]=="861paper"].reset_index(drop=True)
        target_taxid = list(dt_mapping_0["Taxid"])

        df_mapping = pd.read_csv('../data/AllDataIdMapping.csv', sep='\t')
        df_kmer = pd.read_csv(f'../data/allvirusGenome{k}Kmer.csv').rename(columns={'Unnamed: 0': 'Taxid'})
        df = pd.merge(df_mapping[['Taxid', 'Label']], df_kmer, on='Taxid')
        df = df[df["Taxid"].isin(target_taxid)].reset_index(drop=True)
        df.set_index('Taxid', inplace=True)
        print(df.shape)
    
        # 2. 准备特征和标签
        X = df.drop(columns=['Label']).values
        y = df['Label'].values
    
        # 3. 定义评价指标记录器
        metrics_list = {
          'accuracy': [],
          'precision': [],
          'recall': [],
          'f1': [],
          'auroc': [],
          'auprc': []
        }
    
        # 4. 定义划分器
        sss = StratifiedShuffleSplit(n_splits=100, test_size=0.15, random_state=42)
    
        # 5. 网格搜索参数 - KNN
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    
        # 6. 进行100次划分 + 训练 + 评估
        for train_val_index, test_index in sss.split(X, y):
            # 15% 测试集
            X_train_val, X_test = X[train_val_index], X[test_index]
            y_train_val, y_test = y[train_val_index], y[test_index]
    
            # 从 85% 的训练+验证集中划出 17.65% ≈ 15% 的验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=0.1765,  # 15 / 85
                stratify=y_train_val,
                random_state=42
            )
            # 网格搜索
            grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='roc_auc', n_jobs=32)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
    
            # 预测
            y_pred = best_model.predict(X_val)
            y_prob = best_model.predict_proba(X_val)[:, 1]
    
            # 记录评价指标
            metrics_list['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics_list['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            metrics_list['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            metrics_list['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            #metrics_list['auc'].append(roc_auc_score(y_val, y_prob))
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            metrics_list['auroc'].append(auc(fpr, tpr))
            precision, recall, _ = precision_recall_curve(y_val, y_prob, pos_label=1)
            metrics_list['auprc'].append(auc(recall, precision))
    
        # 7. 保存结果
        result_df = pd.DataFrame(metrics_list)
        summary = result_df.describe().loc[['mean', 'std']]
    
        print(summary)
    
        # 保存文件
        os.makedirs("../result/val/genome", exist_ok=True)
        top_num = X.shape[1]  # 特征数
        result_df.to_csv(f"../result/val/genome/knn_result_{k}mer.csv", index=False)
        summary.to_csv(f"../result/val/genome/knn_summary_{k}mer.csv")


if __name__ == "__main__":
    print("START: ", time.ctime(), flush=True)
    kmer_xgboost_val_paperdataset()
    print("END: ", time.ctime(), flush=True)
