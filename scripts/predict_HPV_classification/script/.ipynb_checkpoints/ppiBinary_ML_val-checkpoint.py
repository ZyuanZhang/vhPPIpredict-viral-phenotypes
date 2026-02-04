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



def generate_binary_matrix():
    dt_hpv_ppi = pd.DataFrame(data=None, columns=["virus_unid","human_unid","pred_score","pred_label"])
    fList = ["/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase1/result_pred_interactions/"+f for f in os.listdir("/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase1/result_pred_interactions/") if f.endswith("_predscore.txt")]
    for f in fList:
        dt_tmp = pd.read_csv(f, sep="\t", header=0)
        dt_tmp_pos = dt_tmp[dt_tmp["pred_label"]==1.0].reset_index(drop=True)
        dt_hpv_ppi = pd.concat([dt_hpv_ppi, dt_tmp_pos], ignore_index=True)
    
    dt_hpv_info = pd.read_excel("/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase1/originalDT/HPV_subspecies_info_from_2019_cell.xlsx", sheet_name="Sheet1", header=0)
    dict_hpv_subspecies_label = {}
    for i in range(dt_hpv_info.shape[0]):
        if dt_hpv_info["Label"][i]=="HR":
            dict_hpv_subspecies_label[dt_hpv_info["Abbreviation"][i].strip()]=1
        else:
            dict_hpv_subspecies_label[dt_hpv_info["Abbreviation"][i].strip()]=0
    
    dt_hpv_subspecies = []
    for i in range(dt_hpv_ppi.shape[0]):
        vname, vid, hid = dt_hpv_ppi["virus_unid"][i].split("_")[0], dt_hpv_ppi["virus_unid"][i], dt_hpv_ppi["human_unid"][i]
        dt_hpv_subspecies.append([vname, vid, hid, dict_hpv_subspecies_label[vname]])
    dt_hpv_subspecies = pd.DataFrame(dt_hpv_subspecies, columns=["virus_name","virus_unid","human_unid","label"])
    
    hpv_subsp = list(set(dt_hpv_subspecies["virus_name"]))
    hpv_subsp.sort()
    hid_list = list(set(dt_hpv_subspecies["human_unid"]))
    hid_list.sort()
    
    feature_matrix = np.zeros((len(hpv_subsp), len(hid_list)+1))
    feature_matrix = pd.DataFrame(feature_matrix, columns=hid_list+["risk_label"])
    feature_matrix.index = hpv_subsp
    
    for i in range(dt_hpv_subspecies.shape[0]):
        feature_matrix.loc[dt_hpv_subspecies["virus_name"][i], dt_hpv_subspecies["human_unid"][i]] = 1
        feature_matrix.loc[dt_hpv_subspecies["virus_name"][i], "risk_label"] = dt_hpv_subspecies["label"][i]
    
    feature_matrix.to_csv("../data/ppi_binary_matrix.csv")



def ppiBinary_rf_val_hpv():
    # 获得每个病毒不同阈值下的ppi
    df = pd.read_csv('../data/ppi_binary_matrix.csv').rename(
        columns={'Unnamed: 0': 'taxid', 'risk_label': 'label'}
    )
    df.set_index('taxid', inplace=True)
    print(df.shape)
    
    # 读取100次划分下的特征重要性
    feature_importances_df = pd.read_csv("../feature_importances/rf_feature_importances_ppibinary.csv")
    mean_importance = feature_importances_df.mean()
    
    #for top_num in range(50, 2050, 50):
    for top_num in [30]:
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
        
        # 随机划分 100 次
        sss = StratifiedShuffleSplit(n_splits=100, test_size=0.15, random_state=42)
        
        param_grid = {
                'n_estimators': [5,10,15],
                'max_depth': [2],
            }

        
        best_params_list = []
        for train_val_index, test_index in sss.split(X, y):
            # 15% 测试集
            X_train_val, X_test = X[train_val_index], X[test_index]
            y_train_val, y_test = y[train_val_index], y[test_index]
    
            # # 从 85% 的训练+验证集中划出 17.65% ≈ 15% 的验证集
            # X_train, X_val, y_train, y_val = train_test_split(
            #     X_train_val, y_train_val,
            #     test_size=0.1765,  # 15 / 85
            #     stratify=y_train_val,
            #     random_state=42
            # )
    
            # 定义参数搜索空间（可根据需要调整）
            


            grid = GridSearchCV(RandomForestClassifier(
                random_state=train_val_index[0]),
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=64)
            grid.fit(X_train_val, y_train_val)
            best_model = grid.best_estimator_
            best_params_list.append(grid.best_estimator_)
            # 预测
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]
    
            # 记录评价指标
            metrics_list['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics_list['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            metrics_list['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            metrics_list['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            #metrics_list['auc'].append(roc_auc_score(y_test, y_prob))
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            metrics_list['auroc'].append(auc(fpr, tpr))
            precision, recall, _ = precision_recall_curve(y_test, y_prob, pos_label=1)
            metrics_list['auprc'].append(auc(recall, precision))
            
        
        # 结果汇总
        result_df = pd.DataFrame(metrics_list)
        summary = result_df.describe().loc[['mean', 'std']]
        print(summary)
    
        # 保存结果
        os.makedirs("../result/test/ppi", exist_ok=True)
        result_df.to_csv(f"../result/test/ppi/rf_top{top_num}_result_val.csv", index=False)
        summary.to_csv(f"../result/test/ppi/rf_top{top_num}_summary_val.csv")

        best_params_df = pd.DataFrame(best_params_list)
        best_params_df.to_csv(f"../result/test/ppi/rf_best_params_ppi_top{top_num}.csv", index=False)




def kmer_rf_val_hpv():
    for k in [3, 4, 5]:
        # 1. 读取数据
        df = pd.read_csv(f'../data/hpv_genome_{k}mer.csv').rename(
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

        os.makedirs("../result/val/genome", exist_ok=True)
        result_df.to_csv(f"../result/val/genome/rf_result_{k}mer.csv", index=False)
        summary.to_csv(f"../result/val/genome/rf_summary_{k}mer.csv")

        print(f"===== {k}-mer Summary =====")
        print(summary)



if __name__ == "__main__":
    print("START: ",time.ctime())
    #generate_binary_matrix()
    ppiBinary_rf_val_hpv()
    print("END: ",time.ctime())
