import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.preprocessing import label_binarize, LabelEncoder
import time


def main_pred_tissue_tropism():
    ## 加载待预测文件
    dt_test_pred = pd.read_csv("../../low_evidence_virus_dataset/original_dataset/v451_rna_binary_ppi_matrix.txt", sep="\t", index_col=0)

    # 加载 PPI 特征
    df = pd.read_csv('./data/all_virus_ppi_matrix_threshold999_214.csv', index_col=0)
    df = df.groupby('taxid', as_index=False).max()

    # 加载标签
    df_label = pd.read_csv('./data/final_214_genome_proteome_label.csv')
    df_label.rename(columns={'Taxid': 'taxid'}, inplace=True)
    df_label.dropna(subset=['Tp.primary'], inplace=True)

    # 合并标签
    df_xy = pd.merge(df, df_label[['taxid', 'Tp.primary']], on='taxid')
    df_xy.dropna(subset=['Tp.primary'], inplace=True)
    df_xy.set_index('taxid', inplace=True)
    # 合并数量较少的标签vascular和hepatic为other   10:[vascular,hepatic]  20:[vascular,circulatory,hepatic]
    df_xy['Tp.primary'] = df_xy['Tp.primary'].replace({'vascular': 'other', 'hepatic': 'other','circulatory': 'other'})
    # 统计一下各个标签的数量
    print(df_xy['Tp.primary'].value_counts())

    # 特征重要性
    top_num = 1250
    feature_importances_df = pd.read_csv(f"./data/rf_feature_importances_100split_Tpprimary_20other.csv")
    mean_importance = feature_importances_df.mean()
    top_features = mean_importance.sort_values(ascending=False).head(top_num).index.tolist()

    X = df_xy[top_features].values
    y = np.array(df_xy['Tp.primary'])
    taxids = df_xy.index.values

    ## GridSearch CV找最佳参数
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid,
        cv=3,
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
        pred_df.append([vtaxid_test_pred[i], y_test_prob[i][0], y_test_prob[i][1], y_test_prob[i][2], y_test_prob[i][3], y_test_prob[i][4], y_test_prob[i][5], y_test_pred[i]])
    pred_df = pd.DataFrame(pred_df, columns=["vtaxid","prob_gastrointestinal","prob_neural","prob_other","prob_respiratory","prob_systemic","prob_viraemic","pred_label"])
    
    pred_df.to_csv("./result/tissue_tropism_v451_rna_viruses.csv", index=False, float_format="%.4f")



if __name__ == "__main__":
    print("START: ",time.ctime(), flush=True)

    main_pred_tissue_tropism()

    print("END: ",time.ctime(), flush=True)

    """
    # 标签编码
    le = LabelEncoder()
    df_xy['Tp.primary'] = le.fit_transform(df_xy['Tp.primary'])
    class_names = le.classes_
    n_classes = len(class_names)
    classes = np.unique(df_xy['Tp.primary'])
    outpath = f"./result_macro_20other/svm_test/Tpprimary"
    os.makedirs(outpath, exist_ok=True)

    for top_num in [1250]:
        print(f"Top {top_num} features start!!!")
        top_features = mean_importance.sort_values(ascending=False).head(top_num).index.tolist()

        X = df_xy[top_features].values
        y = np.array(df_xy['Tp.primary'])
        taxids = df_xy.index.values

        # 初始化评估指标
        metrics_list = {
            'accuracy': [],
            'precision_macro': [],
            'recall_macro': [],
            'f1_macro': [],
            'auc_macro': []
        }

        sss = StratifiedShuffleSplit(n_splits=100, test_size=0.15, random_state=42)
        for i, (train_val_index, test_index) in enumerate(sss.split(X, y)):
            if i % 10 == 0:
                print(f"Processing split {i + 1}/100", flush=True)

            X_train_val, X_test = X[train_val_index], X[test_index]
            y_train_val, y_test = y[train_val_index], y[test_index]
            taxid_test = taxids[test_index]

            # X_train, X_val, y_train, y_val, taxid_train, taxid_val = train_test_split(
            #     X_train_val, y_train_val, taxids[train_val_index],
            #     test_size=0.1765,
            #     stratify=y_train_val,
            #     random_state=42
            # )

            # SVM参数网格，kernel, C 和 gamma 是常用参数
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }

            grid_search = GridSearchCV(
                SVC(probability=True, random_state=42),
                param_grid,
                cv=3,
                scoring='roc_auc_ovr_weighted',
                n_jobs=12
            )
            grid_search.fit(X_train_val, y_train_val)

            best_clf = grid_search.best_estimator_
            best_clf.fit(X_train_val, y_train_val)

            # 预测与评估
            y_pred = best_clf.predict(X_test)
            y_prob = best_clf.predict_proba(X_test)

            metrics_list['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics_list['precision_macro'].append(precision_score(y_test, y_pred, average='macro', zero_division=0))
            metrics_list['recall_macro'].append(recall_score(y_test, y_pred, average='macro', zero_division=0))
            metrics_list['f1_macro'].append(f1_score(y_test, y_pred, average='macro', zero_division=0))

            def safe_roc_auc(y_true, y_prob, n_classes):
                # 由于某些验证集可能缺少某一类，导致macro指标无法计算ROC。将这列忽略计算。
                # one-hot 编码 (注意这里用整数标签范围)
                y_true = label_binarize(y_true, classes=range(n_classes))
                auc_list = []
                n_classes = y_true.shape[1]
                for i in range(n_classes):
                    if len(np.unique(y_true[:, i])) < 2:  # 如果该类没有正负样本
                        auc_list.append(np.nan)  # 用 NaN 占位
                    else:
                        auc_list.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
                return np.nanmean(auc_list)  # 忽略 NaN，计算平均


            auc_macro = safe_roc_auc(y_test, y_prob, n_classes)
            metrics_list['auc_macro'].append(auc_macro)

            # 保存每个样本的概率预测
            prob_df = pd.DataFrame(y_prob, columns=[f'prob_{c}' for c in class_names])
            prob_df['taxid'] = taxids[test_index]
            prob_df['true'] = y_test
            prob_df['pred'] = y_pred
            prob_df.to_csv(f"{outpath}/top{top_num}_probs_split{i}.csv", index=False)

        # 结果保存
        result_df = pd.DataFrame(metrics_list)
        summary = result_df.describe().loc[['mean', 'std']]
        result_df.to_csv(f"{outpath}/top{top_num}_result_test.csv", index=False)
        summary.to_csv(f"{outpath}/top{top_num}_summary_test.csv")
    """



