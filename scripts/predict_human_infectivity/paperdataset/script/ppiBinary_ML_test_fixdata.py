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

def ppiBinary_rf_test_paperdataset():
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
    feature_importances_df = pd.read_csv("../data/rf_feature_importances_100.csv")
    mean_importance = feature_importances_df.mean()
    
    #for top_num in range(50, 2050, 50):
    for top_num in [500]:
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
    
        best_params_list = []
        
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
            grid_search.fit(X_train_val, y_train_val)
    
            # 使用最优参数训练模型
            best_clf = grid_search.best_estimator_
            best_params_list.append(grid_search.best_estimator_)
            
            #best_clf.fit(X_train_val, y_train_val)
    
            # 预测与评估
            y_pred = best_clf.predict(X_test)
            y_prob = best_clf.predict_proba(X_test)[:, 1]
    
            metrics_list['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics_list['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            metrics_list['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            metrics_list['f1'].append(f1_score(y_test, y_pred, zero_division=0))
    
            #try:
            #    auc = roc_auc_score(y_val, y_prob)
            #except ValueError:
            #    auc = np.nan
            #metrics_list['auc'].append(auc)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            metrics_list['auroc'].append(auc(fpr, tpr))
            precision, recall, _ = precision_recall_curve(y_test, y_prob, pos_label=1)
            metrics_list['auprc'].append(auc(recall, precision))
    
        # 结果汇总
        result_df = pd.DataFrame(metrics_list)
        summary = result_df.describe().loc[['mean', 'std']]
        print(summary)
    
        # 保存结果
        os.makedirs("../result/test/ppi_fix", exist_ok=True)
        result_df.to_csv(f"../result/test/ppi_fix/rf_top{top_num}_result.csv", index=False)
        summary.to_csv(f"../result/test/ppi_fix/rf_top{top_num}_summary.csv")

        best_params_df = pd.DataFrame(best_params_list)
        best_params_df.to_csv(f"../result/test/ppi_fix/rf_best_params_top{top_num}.csv", index=False)







if __name__ == "__main__":
    print("START: ",time.ctime(),flush=True)

    ppiBinary_rf_test_paperdataset()

    print("END: ",time.ctime(),flush=True)
