import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.preprocessing import label_binarize
import time


def Transmissibility_LowConfidence_Pred():
    ## 加载待预测文件
    df_test = pd.read_csv("../../data_updated/feat_ppi_binary_matrix_low_confidence.csv", header=0)
    df_test.set_index("Taxid", inplace=True)
    dt_test_pred = df_test

    # 加载 PPI 特征
    df = pd.read_csv('../Virulence/all_virus_ppi_matrix_threshold999_214.csv', index_col=0)
    df = df.groupby('taxid', as_index=False).max()

    # 加载标签
    df_label = pd.read_csv('../../scripts_pred_phenotype/predict_transmissibility/data/final_214_genome_proteome_label.csv')
    df_label.rename(columns={'Taxid': 'taxid'}, inplace=True)
    df_label.dropna(subset=['Tr.level'], inplace=True)

    # 合并标签
    df_xy = pd.merge(df, df_label[['taxid', 'Tr.level']], on='taxid')
    df_xy.dropna(subset=['Tr.level'], inplace=True)
    df_xy.set_index('taxid', inplace=True)
    classes = np.unique(df_xy['Tr.level'])

    # 特征重要性
    top_num = 200
    feature_importances_df = pd.read_csv(f"../../scripts_pred_phenotype/predict_transmissibility/data/rf_feature_importances_100split_Trlevel.csv")
    mean_importance = feature_importances_df.mean()
    top_features = mean_importance.sort_values(ascending=False).head(top_num).index.tolist()

    X = df_xy[top_features].values
    y = np.array(df_xy['Tr.level'])
    taxids = df_xy.index.values

    # GridSearch CV 找最佳参数
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid,
        cv=5,
        scoring='roc_auc_ovr_weighted',
        n_jobs=12
    )

    grid_search.fit(X, y)
    best_params = grid_search.best_params_

    final_model = SVC(**best_params, random_state=42, probability=True)
    final_model.fit(X,y)

    xxx = 0
    for feat in top_features:
        if feat not in dt_test_pred.columns:
            dt_test_pred[feat] = 0
            xxx += 1

    # 丢弃预测集里多余的列，只保留训练时的 top_features，并严格按顺序排列
    df_test_pred_final = dt_test_pred[top_features]
    X_test_pred = df_test_pred_final.values

    y_test_pred = final_model.predict(X_test_pred)
    y_test_prob = final_model.predict_proba(X_test_pred)

    pred_df = []
    vtaxid_test_pred = dt_test_pred.index.tolist()
    for i in range(len(vtaxid_test_pred)):
        pred_df.append([vtaxid_test_pred[i], y_test_prob[i][0], y_test_prob[i][1], y_test_prob[i][2], y_test_pred[i]])
    pred_df = pd.DataFrame(pred_df, columns=["vtaxid","no human-to-human","limited human-to-human","sustained human-to-human","pred_label"])
    dict_tmp = {2:"no human-to-human", 3:"limited human-to-human", 4:"sustained human-to-human"}
    pred_df["pred_label"] = [dict_tmp[pred_df["pred_label"][i]] for i in range(pred_df.shape[0])]

    #pred_df.to_csv("./result/infectivity_v451_rna_viruses.csv", index=False, float_format="%.4f")
    output_dir = "../../mid_result_updated/low_confidience_predict/transmissibility/"
    os.makedirs(output_dir, exist_ok=True)

    # 保存预测结果
    pred_df.to_csv(f"{output_dir}/svm_top{top_num}_low_confidence_prediction_score_transmissibility.csv", index=False)

if __name__ == "__main__":
    print("START: ",time.ctime(), flush=True)

    Transmissibility_LowConfidence_Pred()

    print("END: ",time.ctime(), flush=True)
