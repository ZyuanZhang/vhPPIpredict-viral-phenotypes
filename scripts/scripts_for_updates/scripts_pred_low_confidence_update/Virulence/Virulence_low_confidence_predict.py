import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import time


def Virulence_LowConfidence_Pred():

    # 加载待预测文件
    df_test = pd.read_csv("../../data_updated/feat_ppi_binary_matrix_low_confidence.csv", header=0)
    df_test.set_index("Taxid", inplace=True)

    # 加载 PPI 特征
    df = pd.read_csv("./all_virus_ppi_matrix_threshold999_214.csv", index_col=0)
    df = df.groupby("taxid", as_index=False).max()

    # 加载标签
    df_label = pd.read_excel("../../data/Supplementary Tables.xlsx", sheet_name="TableS5", header=2)
    df_label = df_label.iloc[:, 0:4]
    df_label.rename(columns={"Virus Taxid": "taxid"}, inplace=True)

    # 仅保留具有明确毒力标签的样本
    df_label = df_label[df_label["Virulence"].isin(["severe", "nonsevere"])].copy()
    df_label["label"] = df_label["Virulence"].map({"severe": 1, "nonsevere": 0})

    # 合并 PPI 和标签
    df_xy = pd.merge(df, df_label[["taxid", "label"]], on="taxid")
    df_xy.set_index("taxid", inplace=True)

    print("Training samples:", df_xy.shape[0])
    print("Test samples:", df_test.shape[0])
    print("Training label distribution:")
    print(df_xy["label"].value_counts())

    # 读取特征重要性并选择 Top 250
    top_num = 250
    feature_importances_df = pd.read_csv("../../scripts_pred_phenotype/predict_virulence/data/rf_feature_importances_100.csv", index_col=0)
    mean_importance = feature_importances_df.mean()
    top_features = mean_importance.sort_values(ascending=False).head(top_num).index.tolist()

    # 检查训练集特征
    missing_train = [feat for feat in top_features if feat not in df_xy.columns]
    if len(missing_train) > 0:
        raise ValueError(f"{len(missing_train)} top features are missing from training data: {missing_train[:10]}")

    # 构建训练集
    X = df_xy[top_features].values
    y = df_xy["label"].values

    # 测试集中不存在的训练特征补0
    missing_test = [feat for feat in top_features if feat not in df_test.columns]
    print(f"Missing features in test data: {len(missing_test)}/{top_num}")
    if len(missing_test) > 0:
        print("Missing features will be filled with 0.")
        print("Examples:", missing_test[:10])

    # 严格按照训练特征顺序构建预测集，缺失特征自动补0
    df_test_pred = df_test.reindex(columns=top_features, fill_value=0)
    X_test_pred = df_test_pred.values

    print("X train shape:", X.shape)
    print("X test shape:", X_test_pred.shape)

    # GridSearch CV
    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(SVC(probability=True, random_state=42, class_weight="balanced"), param_grid, cv=cv, scoring="roc_auc", n_jobs=12, refit=True)
    grid_search.fit(X, y)

    print("Best parameters:", grid_search.best_params_)
    print("Best CV AUROC:", grid_search.best_score_)

    # GridSearchCV已经使用最佳参数在全部训练数据上重新拟合
    final_model = grid_search.best_estimator_

    # 对独立无标签数据进行预测
    y_test_prob = final_model.predict_proba(X_test_pred)[:, 1]

    # 保存结果
    pred_df = pd.DataFrame({"vtaxid": df_test.index, "pred": y_test_prob})
    pred_df["pred_label"] = np.where(pred_df["pred"] > 0.5, "severe", "nonsevere")

    pred_df_severe = pred_df[pred_df["pred_label"] == "severe"].reset_index(drop=True)

    print("Predicted severe:", pred_df_severe.shape[0])
    print(pred_df_severe.head())

    output_dir = "../../mid_result_updated/low_confidience_predict/virulence/"
    os.makedirs(output_dir, exist_ok=True)

    # 保存预测结果
    pred_df.to_csv(f"{output_dir}/svm_top{top_num}_low_confidence_prediction_score_virulence.csv", index=False)


if __name__ == "__main__":
    print("START:", time.ctime(), flush=True)
    Virulence_LowConfidence_Pred()
    print("END:", time.ctime(), flush=True)