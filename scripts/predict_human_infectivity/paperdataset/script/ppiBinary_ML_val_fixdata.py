import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split
import os
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def ppiBinary_rf_val_paperdataset():
    dt_mapping_0 = pd.read_csv("../data/AllDataIdMapping.csv", sep="\t")
    dt_mapping_0 = dt_mapping_0[dt_mapping_0["belong"]=="861paper"].reset_index(drop=True)
    target_taxid = list(dt_mapping_0["Taxid"])

    # 获得每个病毒不同阈值下的ppi
    df = pd.read_csv('../data/all_virus_ppi_matrix_threshold999.csv',index_col=0)
    df = df.groupby('taxid', as_index=False).max()
    #print(df.shape)
    
    # 合并标签
    df_label = pd.read_csv('../data/AllDataIdMapping.csv',sep='\t')
    df_label.rename(columns={'Taxid':'taxid'}, inplace=True)
    df_label.rename(columns={'Label':'label'}, inplace=True)
    df = pd.merge(df, df_label[['label','taxid']], on='taxid')
    df = df[df["taxid"].isin(target_taxid)].reset_index(drop=True)
    df.set_index('taxid', inplace=True)
    print(df.shape)
    
    # 读取100次划分下的特征重要性
    feature_importances_df = pd.read_csv("../result/rf_feature_importances_100.csv")
    mean_importance = feature_importances_df.mean()
    
    for top_num in range(50, 2050, 50):
    #for top_num in [300]:
        #print(f"Top {top_num} features:")
        top_100_features = mean_importance.sort_values(ascending=False).head(top_num).index.tolist()
        #print(top_100_features)
    
        # 进行随机85：15的随机划分，进行预测。
        X = df[top_100_features].values
        y = np.array(df['label'])
        #print(X.shape) # (861, 12039)
    
        # 初始化评估指标容器
        metrics_list = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auroc': [],
            'auprc': []
        }
    
        ## 读取事先划分好的数据集  
        splits = pd.read_csv("../data/fixed_100_splits_iteration_taxid_species.csv")

        for i in range(1, 101):
            split_i = splits[splits["iteration"] == i]
        
            test_taxids = split_i[split_i["set"] == "test"]["taxid"].values
            train_taxids = split_i[split_i["set"] == "train"]["taxid"].values
        
            train_df = df.loc[train_taxids]
            test_df  = df.loc[test_taxids]
        
            X_train_val = train_df[top_100_features].values
            y_train_val = train_df["label"].values
        
            X_test  = test_df[top_100_features].values
            y_test  = test_df["label"].values
    
            # 从 85% 的训练+验证集中划出 17.65% ≈ 15% 的验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=0.1765,  # 15 / 85
                stratify=y_train_val,
                random_state=42
            )
    
            # 定义参数搜索空间（可根据需要调整）
            param_grid = {
                'n_estimators': [50 ,100, 200, 300, 500],
                'max_depth': [None, 10, 15, 20, 25],
            }
    
            # 使用 3 折交叉验证进行网格搜索
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=5,
                scoring='roc_auc',  # 以AUC作为选择指标
                n_jobs=64
            )
            grid_search.fit(X_train, y_train)
    
            # 使用最优参数训练模型
            best_clf = grid_search.best_estimator_
            best_clf.fit(X_train, y_train)
    
            # 预测与评估
            y_pred = best_clf.predict(X_val)
            y_prob = best_clf.predict_proba(X_val)[:, 1]
    
            metrics_list['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics_list['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            metrics_list['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            metrics_list['f1'].append(f1_score(y_val, y_pred, zero_division=0))
    
            #try:
            #    auc = roc_auc_score(y_val, y_prob)
            #except ValueError:
            #    auc = np.nan
            #metrics_list['auc'].append(auc)
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            metrics_list['auroc'].append(auc(fpr, tpr))
            precision, recall, _ = precision_recall_curve(y_val, y_prob, pos_label=1)
            metrics_list['auprc'].append(auc(recall, precision))
    
        # 结果汇总
        result_df = pd.DataFrame(metrics_list)
        summary = result_df.describe().loc[['mean', 'std']]
    
        # 保存结果
        result_df.to_csv(f"../result/ppi_fix/rf_top{top_num}_result_val.csv", index=False)
        summary.to_csv(f"../result/ppi_fix/rf_top{top_num}_summary_val.csv")
        print("RF Over: ", time.ctime(), flush=True)



def ppiBinary_knn_val_paperdataset():
    dt_mapping_0 = pd.read_csv("../data/AllDataIdMapping.csv", sep="\t")
    dt_mapping_0 = dt_mapping_0[dt_mapping_0["belong"]=="861paper"].reset_index(drop=True)
    target_taxid = list(dt_mapping_0["Taxid"])

    # 获得每个病毒不同阈值下的ppi
    df = pd.read_csv('../data/all_virus_ppi_matrix_threshold999.csv',index_col=0)
    df = df.groupby('taxid', as_index=False).max()
    #print(df)
    
    # 合并标签
    df_label = pd.read_csv('../data/AllDataIdMapping.csv',sep='\t')
    df_label.rename(columns={'Taxid':'taxid'}, inplace=True)
    df_label.rename(columns={'Label':'label'}, inplace=True)
    df = pd.merge(df, df_label[['label','taxid']], on='taxid')
    df = df[df["taxid"].isin(target_taxid)].reset_index(drop=True)
    df.set_index('taxid', inplace=True)
    print(df.shape)
    
    # 读取100次划分下的特征重要性
    feature_importances_df = pd.read_csv("../result/rf_feature_importances_100.csv")
    mean_importance = feature_importances_df.mean()
    
    for top_num in range(50, 2050, 50):
    #for top_num in [250]:
        #print(f"Top {top_num} features:")
        top_100_features = mean_importance.sort_values(ascending=False).head(top_num).index.tolist()
        #print(top_100_features)
    
        X = df[top_100_features].values
        y = np.array(df['label'])
        #print(X.shape)
    
        metrics_list = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auroc': [],
            'auprc': []
        }
    
        splits = pd.read_csv("../data/fixed_100_splits_iteration_taxid_species.csv")

        for i in range(1, 101):
            split_i = splits[splits["iteration"] == i]
        
            test_taxids = split_i[split_i["set"] == "test"]["taxid"].values
            train_taxids = split_i[split_i["set"] == "train"]["taxid"].values
        
            train_df = df.loc[train_taxids]
            test_df  = df.loc[test_taxids]
        
            X_train_val = train_df[top_100_features].values
            y_train_val = train_df["label"].values
        
            X_test  = test_df[top_100_features].values
            y_test  = test_df["label"].values
    
            # 从 85% 的训练+验证集中划出 17.65% ≈ 15% 的验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=0.1765,  # 15 / 85
                stratify=y_train_val,
                random_state=42
            )
    
            # KNN参数搜索空间
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
    
            grid_search = GridSearchCV(
                KNeighborsClassifier(),
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=64
            )
            grid_search.fit(X_train, y_train)
    
            best_clf = grid_search.best_estimator_
            best_clf.fit(X_train, y_train)
    
            y_pred = best_clf.predict(X_val)
            y_prob = best_clf.predict_proba(X_val)[:, 1]
    
            metrics_list['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics_list['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            metrics_list['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            metrics_list['f1'].append(f1_score(y_val, y_pred, zero_division=0))
    
            #try:
            #    auc = roc_auc_score(y_val, y_prob)
            #except ValueError:
            #    auc = np.nan
            #metrics_list['auc'].append(auc)
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            metrics_list['auroc'].append(auc(fpr, tpr))
            precision, recall, _ = precision_recall_curve(y_val, y_prob, pos_label=1)
            metrics_list['auprc'].append(auc(recall, precision))
    
        result_df = pd.DataFrame(metrics_list)
        summary = result_df.describe().loc[['mean', 'std']]
        result_df.to_csv(f"../result/ppi_fix/knn_top{top_num}_result_val.csv", index=False)
        summary.to_csv(f"../result/ppi_fix/knn_top{top_num}_summary_val.csv")
        print("KNN Over: ", time.ctime(), flush=True)



def ppiBinary_svm_val_paperdataset():
    dt_mapping_0 = pd.read_csv("../data/AllDataIdMapping.csv", sep="\t")
    dt_mapping_0 = dt_mapping_0[dt_mapping_0["belong"]=="861paper"].reset_index(drop=True)
    target_taxid = list(dt_mapping_0["Taxid"])

    # 获得每个病毒不同阈值下的ppi
    df = pd.read_csv('../data/all_virus_ppi_matrix_threshold999.csv',index_col=0)
    df = df.groupby('taxid', as_index=False).max()
    #print(df)
    
    # 合并标签
    df_label = pd.read_csv('../data/AllDataIdMapping.csv',sep='\t')
    df_label.rename(columns={'Taxid':'taxid'}, inplace=True)
    df_label.rename(columns={'Label':'label'}, inplace=True)
    df = pd.merge(df, df_label[['label','taxid']], on='taxid')
    df = df[df["taxid"].isin(target_taxid)].reset_index(drop=True)
    df.set_index('taxid', inplace=True)
    print(df.shape)
    
    # 读取100次划分下的特征重要性
    feature_importances_df = pd.read_csv("../result/rf_feature_importances_100.csv")
    mean_importance = feature_importances_df.mean()
    
    for top_num in range(50, 2050, 50):
    #for top_num in [550]:
        #print(f"Top {top_num} features:")
        top_100_features = mean_importance.sort_values(ascending=False).head(top_num).index.tolist()
        #print(top_100_features)
    
        X = df[top_100_features].values
        y = np.array(df['label'])
        #print(X.shape)
    
        metrics_list = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auroc': [],
            'auprc': []
        }
    
        splits = pd.read_csv("../data/fixed_100_splits_iteration_taxid_species.csv")

        for i in range(1, 101):
            split_i = splits[splits["iteration"] == i]
        
            test_taxids = split_i[split_i["set"] == "test"]["taxid"].values
            train_taxids = split_i[split_i["set"] == "train"]["taxid"].values
        
            train_df = df.loc[train_taxids]
            test_df  = df.loc[test_taxids]
        
            X_train_val = train_df[top_100_features].values
            y_train_val = train_df["label"].values
        
            X_test  = test_df[top_100_features].values
            y_test  = test_df["label"].values
    
            # 从 85% 的训练+验证集中划出 17.65% ≈ 15% 的验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=0.1765,  # 15 / 85
                stratify=y_train_val,
                random_state=42
            )
            # SVM参数搜索空间
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
    
            grid_search = GridSearchCV(
                SVC(probability=True, random_state=42),
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=64
            )
            grid_search.fit(X_train, y_train)
    
            best_clf = grid_search.best_estimator_
            best_clf.fit(X_train, y_train)
    
            y_pred = best_clf.predict(X_val)
            y_prob = best_clf.predict_proba(X_val)[:, 1]
    
            metrics_list['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics_list['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            metrics_list['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            metrics_list['f1'].append(f1_score(y_val, y_pred, zero_division=0))
    
            #try:
            #    auc = roc_auc_score(y_val, y_prob)
            #except ValueError:
            #    auc = np.nan
            #metrics_list['auc'].append(auc)
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            metrics_list['auroc'].append(auc(fpr, tpr))
            precision, recall, _ = precision_recall_curve(y_val, y_prob, pos_label=1)
            metrics_list['auprc'].append(auc(recall, precision))
    
        result_df = pd.DataFrame(metrics_list)
        summary = result_df.describe().loc[['mean', 'std']]
        result_df.to_csv(f"../result/ppi_fix/svm_top{top_num}_result_val.csv", index=False)
        summary.to_csv(f"../result/ppi_fix/svm_top{top_num}_summary_val.csv")
        print("SVM Over: ", time.ctime(), flush=True)



def ppiBinary_xgboost_val_paperdataset():
    dt_mapping_0 = pd.read_csv("../data/AllDataIdMapping.csv", sep="\t")
    dt_mapping_0 = dt_mapping_0[dt_mapping_0["belong"]=="861paper"].reset_index(drop=True)
    target_taxid = list(dt_mapping_0["Taxid"])

    # 获得每个病毒不同阈值下的ppi
    df = pd.read_csv('../data/all_virus_ppi_matrix_threshold999.csv',index_col=0)
    df = df.groupby('taxid', as_index=False).max()
    #print(df)
    
    # 合并标签
    df_label = pd.read_csv('../data/AllDataIdMapping.csv',sep='\t')
    df_label.rename(columns={'Taxid':'taxid'}, inplace=True)
    df_label.rename(columns={'Label':'label'}, inplace=True)
    df = pd.merge(df, df_label[['label','taxid']], on='taxid')
    df = df[df["taxid"].isin(target_taxid)].reset_index(drop=True)
    df.set_index('taxid', inplace=True)
    print(df.shape)
    
    # 读取100次划分下的特征重要性
    feature_importances_df = pd.read_csv("../result/rf_feature_importances_100.csv")
    mean_importance = feature_importances_df.mean()
    
    for top_num in range(50, 2050, 50):
    #for top_num in [200]:
        #print(f"Top {top_num} features:")
        top_100_features = mean_importance.sort_values(ascending=False).head(top_num).index.tolist()
        #print(top_features)
    
        X = df[top_100_features].values
        y = np.array(df['label'])
        #print(X.shape)
    
        metrics_list = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auroc': [],
            'auprc': []
        }
    
        splits = pd.read_csv("../data/fixed_100_splits_iteration_taxid_species.csv")

        for i in range(1, 101):
            split_i = splits[splits["iteration"] == i]
        
            test_taxids = split_i[split_i["set"] == "test"]["taxid"].values
            train_taxids = split_i[split_i["set"] == "train"]["taxid"].values
        
            train_df = df.loc[train_taxids]
            test_df  = df.loc[test_taxids]
        
            X_train_val = train_df[top_100_features].values
            y_train_val = train_df["label"].values
        
            X_test  = test_df[top_100_features].values
            y_test  = test_df["label"].values
    
            # 从 85% 的训练+验证集中划出 17.65% ≈ 15% 的验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=0.1765,  # 15 / 85
                stratify=y_train_val,
                random_state=42
            )
            # XGBoost 参数搜索空间
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1],
                'colsample_bytree': [0.8, 1]
            }
    
            grid_search = GridSearchCV(
                XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, tree_method="hist", predictor="cpu_predictor"),
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=16
            )
            grid_search.fit(X_train, y_train)
    
            best_clf = grid_search.best_estimator_
            best_clf.fit(X_train, y_train)
    
            y_pred = best_clf.predict(X_val)
            y_prob = best_clf.predict_proba(X_val)[:, 1]
    
            metrics_list['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics_list['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            metrics_list['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            metrics_list['f1'].append(f1_score(y_val, y_pred, zero_division=0))
    
            #try:
            #    auc = roc_auc_score(y_val, y_prob)
            #except ValueError:
            #    auc = np.nan
            #metrics_list['auc'].append(auc)
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            metrics_list['auroc'].append(auc(fpr, tpr))
            precision, recall, _ = precision_recall_curve(y_val, y_prob, pos_label=1)
            metrics_list['auprc'].append(auc(recall, precision))
    
        result_df = pd.DataFrame(metrics_list)
        summary = result_df.describe().loc[['mean', 'std']]
        result_df.to_csv(f"../result/ppi_fix/xgboost_top{top_num}_result_val.csv", index=False)
        summary.to_csv(f"../result/ppi_fix/xgboost_top{top_num}_summary_val.csv")
        print("XGB Over: ", time.ctime(), flush=True)



if __name__ == "__main__":
    print("START: ",time.ctime(),flush=True)

    ppiBinary_rf_val_paperdataset()
    ppiBinary_knn_val_paperdataset()
    ppiBinary_svm_val_paperdataset()
    ppiBinary_xgboost_val_paperdataset()

    print("END: ",time.ctime(),flush=True)
