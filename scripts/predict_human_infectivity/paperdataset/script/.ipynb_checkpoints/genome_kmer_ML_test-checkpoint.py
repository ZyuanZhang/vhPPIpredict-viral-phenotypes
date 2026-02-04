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


def kmer_svm_test_paperdataset():
    # 1. 读取数据
    for k in [4]:
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

        best_params_list = []
        # 6. 进行100次划分 + 训练 + 评估
        for train_val_index, test_index in sss.split(X, y):
            # 15% 测试集
            X_train_val, X_test = X[train_val_index], X[test_index]
            y_train_val, y_test = y[train_val_index], y[test_index]
            
    
            # SVM对特征尺度敏感，需要进行标准化
            scaler = StandardScaler()
            X_train_val = scaler.fit_transform(X_train_val)
            X_test = scaler.transform(X_test)
    
            # 网格搜索
            grid = GridSearchCV(SVC(probability=True, random_state=train_val_index[0]),  # probability=True用于支持predict_proba
                                param_grid,
                                cv=5,
                                scoring='roc_auc',
                                n_jobs=64)
            grid.fit(X_train_val, y_train_val)
            best_model = grid.best_estimator_
            best_params_list.append(grid.best_estimator_)
    
            # 预测
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]  # 获取正类的概率
    
            # 记录评价指标
            metrics_list['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics_list['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            metrics_list['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            metrics_list['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            #metrics_list['auc'].append(roc_auc_score(y_val, y_prob))
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            metrics_list['auroc'].append(auc(fpr, tpr))
            precision, recall, _ = precision_recall_curve(y_test, y_prob, pos_label=1)
            metrics_list['auprc'].append(auc(recall, precision))
    
        # 7. 保存结果
        result_df = pd.DataFrame(metrics_list)
        summary = result_df.describe().loc[['mean', 'std']]
    
        print(summary)
    
        # 保存文件
        os.makedirs("../result/test/genome", exist_ok=True)
        result_df.to_csv(f"../result/test/genome/svm_result_{k}mer.csv", index=False)
        summary.to_csv(f"../result/test/genome/svm_summary_{k}mer.csv")

        # 保存最佳参数
        best_params_df = pd.DataFrame(best_params_list)
        best_params_df.to_csv(f"../result/test/genome/svm_best_params_{k}mer.csv", index=False)



if __name__ == "__main__":
    print("START: ", time.ctime(), flush=True)
    kmer_svm_test_paperdataset()
    print("END: ", time.ctime(), flush=True)
