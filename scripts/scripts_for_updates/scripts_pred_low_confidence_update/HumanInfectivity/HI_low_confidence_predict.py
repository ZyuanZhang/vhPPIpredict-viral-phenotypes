import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import os
import time


def HI_LowConfidence_Pred():

    # 读取训练数据
    df = pd.read_csv("../../data_updated/human_infecting_data_updated/feat_ppis_binary_matrix.csv", sep="\t", header=0)
    df.set_index("Taxid", inplace=True)

    # 读取100次划分下的特征重要性，并取平均值
    mean_importance = pd.read_csv("../../mid_result_updated/rf_feature_importances_100_for_ppi.csv", header=0).mean()

    # 选取Top 1000重要特征
    top_num = 1000
    top_features = mean_importance.sort_values(ascending=False).head(top_num).index.tolist()

    # 读取无标签独立测试数据
    df_test = pd.read_csv("../../data_updated/feat_ppi_binary_matrix_low_confidence.csv", header=0)

    print("Training dataset shape:", df.shape)
    print("Test dataset shape:", df_test.shape)
    print("Number of selected features:", len(top_features))

    # 检查Top特征是否都存在于训练集和测试集中
    # 检查训练集Top 1000特征
    missing_train = [feature for feature in top_features if feature not in df.columns]
    if len(missing_train) > 0:
        raise ValueError(f"{len(missing_train)} features are missing from training data: {missing_train[:10]}")

    # 检查测试集缺失特征
    missing_test = [feature for feature in top_features if feature not in df_test.columns]
    print(f"Missing features in test data: {len(missing_test)}/{len(top_features)}")

    if len(missing_test) > 0:
        print("Missing test features will be filled with 0.")
        print("Examples:", missing_test[:10])

    # 构建训练集
    X_train = df[top_features].values
    y_train = df["Label"].values

    # 构建测试集：
    # 1. 只保留Top 1000特征
    # 2. test缺少的特征补0
    # 3. 自动按照top_features的顺序排列
    X_test = df_test.reindex(columns=top_features, fill_value=0).values

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)

    # RF参数搜索
    param_grid = {
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [None, 10, 15, 20, 25]
    }

    # 5-fold CV选择最佳参数，并在全部训练数据上重新拟合最佳模型
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="roc_auc", n_jobs=64, refit=True)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("Best parameters:", grid.best_params_)
    print("Best CV AUROC:", grid.best_score_)

    # 对无标签测试集进行预测
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # 保存预测结果
    if "Taxid" in df_test.columns:
        prediction_df = pd.DataFrame({"Taxid": df_test["Taxid"], "predicted_label": y_pred, "prediction_score": y_prob})
    else:
        prediction_df = pd.DataFrame({"sample_index": df_test.index, "predicted_label": y_pred, "prediction_score": y_prob})

    # 输出目录
    output_dir = "../../mid_result_updated/low_confidience_predict/human_infectivity/"
    os.makedirs(output_dir, exist_ok=True)

    # 保存预测结果
    prediction_df.to_csv(f"{output_dir}/rf_top{top_num}_low_confidence_prediction_score.csv", index=False)

    # 保存最佳参数
    best_params_df = pd.DataFrame([grid.best_params_])
    best_params_df["cv_auroc"] = grid.best_score_
    best_params_df.to_csv(f"{output_dir}/rf_top{top_num}_low_confidence_best_params.csv", index=False)

    # 输出基本结果
    print("\nPrediction completed.")
    print("Number of predicted samples:", len(y_pred))
    print("Predicted positive samples:", np.sum(y_pred == 1))
    print("Predicted negative samples:", np.sum(y_pred == 0))

    print("\nPrediction score summary:")
    print(prediction_df["prediction_score"].describe())

    print("\nTop 20 predicted samples:")
    print(prediction_df.sort_values("prediction_score", ascending=False).head(20))


if __name__ == "__main__":
    print("START:", time.ctime(), flush=True)
    HI_LowConfidence_Pred()
    print("END:", time.ctime(), flush=True)