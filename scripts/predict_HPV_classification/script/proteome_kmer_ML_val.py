import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, roc_curve
import os
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def proteome_kmer_rf_val_hpv():
    for k in [1, 2, 3]:
        # 1. 读取数据
        df = pd.read_csv(f'../data/hpv_proteome_{k}mer.csv').rename(
            columns={'Unnamed: 0': 'Taxid', 'risk_label': 'Label'}
        )
        df.set_index('Taxid', inplace=True)

        # 2. 准备特征和标签
        X = df.drop(columns=['Label']).values
        y = df['Label'].values

        # 3. 定义参数网格
        param_grid = {
            'n_estimators': [5,10,15],
        }

        # 4. 记录指标
        metrics_list = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auroc': [],
            'auprc': []
        }

        # 5. 重复 100 次
        for repeat in range(100):
            # 定义 5 折交叉验证划分（每次 random_state 不同，保证随机性）
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat)

            # 网格搜索 + 5折交叉验证
            grid = GridSearchCV(
                #RandomForestClassifier(random_state=42, max_depth=2, min_samples_split=5, min_samples_leaf=3),
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=32
            )
            grid.fit(X, y)

            best_model = grid.best_estimator_

            y_pred = cross_val_predict(best_model, X, y, cv=cv, n_jobs=32, method="predict")
            y_prob = cross_val_predict(best_model, X, y, cv=cv, n_jobs=32, method="predict_proba")[:, 1]

            metrics_list['accuracy'].append(accuracy_score(y, y_pred))
            metrics_list['precision'].append(precision_score(y, y_pred, zero_division=0))
            metrics_list['recall'].append(recall_score(y, y_pred, zero_division=0))
            metrics_list['f1'].append(f1_score(y, y_pred, zero_division=0))
            metrics_list['auroc'].append(roc_auc_score(y, y_prob))
            precision, recall, _ = precision_recall_curve(y, y_prob, pos_label=1)
            metrics_list['auprc'].append(auc(recall, precision))

        # 6. 保存结果
        result_df = pd.DataFrame(metrics_list)
        summary = result_df.describe().loc[['mean', 'std']]

        os.makedirs("../result/val/proteome", exist_ok=True)
        result_df.to_csv(f"../result/val/proteome/rf_result_{k}mer.csv", index=False)
        summary.to_csv(f"../result/val/proteome/rf_summary_{k}mer.csv")

        print(f"===== {k}-mer Summary =====")
        print(summary)



def proteome_kmer_svm_val_hpv():
    # 1. 读取数据
    for k in [1, 2, 3]:
        df = pd.read_csv(f'../data/hpv_proteome_{k}mer.csv').rename(
            columns={'Unnamed: 0': 'Taxid', 'risk_label': 'Label'}
        )
        df.set_index('Taxid', inplace=True)
        
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

        param_grid = {
            'C': [0.1, 1]
        }
        
        for repeat in range(100):
            # 定义 5 折交叉验证划分（每次 random_state 不同，保证随机性）
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat)

            # 网格搜索 + 5折交叉验证
            grid = GridSearchCV(
                SVC(kernel="linear", probability=True, random_state=42),  # probability=True用于支持predict_proba
                param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=32)
            
            grid.fit(X, y)

            best_model = grid.best_estimator_

            y_pred = cross_val_predict(best_model, X, y, cv=cv, n_jobs=32, method="predict")
            y_prob = cross_val_predict(best_model, X, y, cv=cv, n_jobs=32, method="predict_proba")[:, 1]

            metrics_list['accuracy'].append(accuracy_score(y, y_pred))
            metrics_list['precision'].append(precision_score(y, y_pred, zero_division=0))
            metrics_list['recall'].append(recall_score(y, y_pred, zero_division=0))
            metrics_list['f1'].append(f1_score(y, y_pred, zero_division=0))
            metrics_list['auroc'].append(roc_auc_score(y, y_prob))
            precision, recall, _ = precision_recall_curve(y, y_prob, pos_label=1)
            metrics_list['auprc'].append(auc(recall, precision))

        # 6. 保存结果
        result_df = pd.DataFrame(metrics_list)
        summary = result_df.describe().loc[['mean', 'std']]

        os.makedirs("../result/val/proteome", exist_ok=True)
        result_df.to_csv(f"../result/val/proteome/svm_result_{k}mer.csv", index=False)
        summary.to_csv(f"../result/val/proteome/svm_summary_{k}mer.csv")

        print(f"===== {k}-mer Summary =====")
        print(summary)



def proteome_kmer_xgboost_val_hpv():
    for k in [1, 2, 3]:
        df = pd.read_csv(f'../data/hpv_proteome_{k}mer.csv').rename(
            columns={'Unnamed: 0': 'Taxid', 'risk_label': 'Label'}
        )
        df.set_index('Taxid', inplace=True)
        
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
    
        # 5. 网格搜索参数 - XGBoost
        param_grid = {
            'n_estimators': [5,10,15],
            'max_depth': [2],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.7],
            'colsample_bytree': [0.7]
        }


        for repeat in range(100):
            # 定义 5 折交叉验证划分（每次 random_state 不同，保证随机性）
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat)

            # 网格搜索 + 5折交叉验证
            grid = GridSearchCV(
                XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=32)
            
            grid.fit(X, y)

            best_model = grid.best_estimator_

            y_pred = cross_val_predict(best_model, X, y, cv=cv, n_jobs=32, method="predict")
            y_prob = cross_val_predict(best_model, X, y, cv=cv, n_jobs=32, method="predict_proba")[:, 1]

            metrics_list['accuracy'].append(accuracy_score(y, y_pred))
            metrics_list['precision'].append(precision_score(y, y_pred, zero_division=0))
            metrics_list['recall'].append(recall_score(y, y_pred, zero_division=0))
            metrics_list['f1'].append(f1_score(y, y_pred, zero_division=0))
            metrics_list['auroc'].append(roc_auc_score(y, y_prob))
            precision, recall, _ = precision_recall_curve(y, y_prob, pos_label=1)
            metrics_list['auprc'].append(auc(recall, precision))

        # 6. 保存结果
        result_df = pd.DataFrame(metrics_list)
        summary = result_df.describe().loc[['mean', 'std']]

        os.makedirs("../result/val/proteome", exist_ok=True)
        result_df.to_csv(f"../result/val/proteome/xgb_result_{k}mer.csv", index=False)
        summary.to_csv(f"../result/val/proteome/xgb_summary_{k}mer.csv")

        print(f"===== {k}-mer Summary =====")
        print(summary)

        
        
def proteome_kmer_knn_val_hpv():
    # 1. 读取数据
    for k in [1, 2, 3]:
        df = pd.read_csv(f'../data/hpv_proteome_{k}mer.csv').rename(
            columns={'Unnamed: 0': 'Taxid', 'risk_label': 'Label'}
        )
        df.set_index('Taxid', inplace=True)
    
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
        
        # 5. 网格搜索参数 - XGBoost
        param_grid = {
            'n_neighbors': [3,5],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        
        for repeat in range(100):
            # 定义 5 折交叉验证划分（每次 random_state 不同，保证随机性）
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat)
            grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='roc_auc', n_jobs=32)
            grid.fit(X, y)

            best_model = grid.best_estimator_

            y_pred = cross_val_predict(best_model, X, y, cv=cv, n_jobs=32, method="predict")
            y_prob = cross_val_predict(best_model, X, y, cv=cv, n_jobs=32, method="predict_proba")[:, 1]

            metrics_list['accuracy'].append(accuracy_score(y, y_pred))
            metrics_list['precision'].append(precision_score(y, y_pred, zero_division=0))
            metrics_list['recall'].append(recall_score(y, y_pred, zero_division=0))
            metrics_list['f1'].append(f1_score(y, y_pred, zero_division=0))
            metrics_list['auroc'].append(roc_auc_score(y, y_prob))
            precision, recall, _ = precision_recall_curve(y, y_prob, pos_label=1)
            metrics_list['auprc'].append(auc(recall, precision))

        # 6. 保存结果
        result_df = pd.DataFrame(metrics_list)
        summary = result_df.describe().loc[['mean', 'std']]

        os.makedirs("../result/val/proteome", exist_ok=True)
        result_df.to_csv(f"../result/val/proteome/knn_result_{k}mer.csv", index=False)
        summary.to_csv(f"../result/val/proteome/knn_summary_{k}mer.csv")

        print(f"===== {k}-mer Summary =====")
        print(summary)

    
        # 保存文件
        


if __name__ == "__main__":
    print("START: ", time.ctime(), flush=True)
    proteome_kmer_knn_val_hpv()
    print("END: ", time.ctime(), flush=True)
