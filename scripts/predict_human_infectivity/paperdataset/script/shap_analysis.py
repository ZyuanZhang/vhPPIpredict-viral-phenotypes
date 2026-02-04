import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import shap
import matplotlib.pyplot as plt
import os
import time


def ppiBinary_rf_test_paperdataset_shap_analysis():
    # ===================== 数据准备 =====================
    dt_mapping_0 = pd.read_csv("../data/AllDataIdMapping.csv", sep="\t")
    dt_mapping_0 = dt_mapping_0[dt_mapping_0["belong"]=="861paper"].reset_index(drop=True)
    target_taxid = list(dt_mapping_0["Taxid"])

    # 获得每个病毒不同阈值下的ppi
    df = pd.read_csv('../data/all_virus_ppi_matrix_threshold999.csv', index_col=0)
    df = df.groupby('taxid', as_index=False).max()
    
    # 合并标签
    df_label = pd.read_csv('../data/AllDataIdMapping.csv', sep='\t')
    df_label.rename(columns={'Taxid':'taxid', 'Label':'label'}, inplace=True)
    df = pd.merge(df, df_label[['label','taxid']], on='taxid')
    df = df[df["taxid"].isin(target_taxid)].reset_index(drop=True)
    df.set_index('taxid', inplace=True)
    print("Final dataset shape:", df.shape)
    
    # ===================== 取 Top300 特征 =====================
    feature_importances_df = pd.read_csv("../data/rf_feature_importances_100.csv")
    mean_importance = feature_importances_df.mean()
    top_features = mean_importance.sort_values(ascending=False).head(300).index.tolist()

    X = df[top_features].values
    y = np.array(df['label'])

    # ===================== GridSearchCV 找全局最优参数 =====================
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 15, 20, 25],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=64
    )
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    print("Best Params:", best_params)

    # ===================== 用最优参数训练全量模型 =====================
    final_model = RandomForestClassifier(**best_params, random_state=42)
    final_model.fit(X, y)

    # ===================== SHAP 分析 =====================
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)   # list[y_class]，取正类
    shap_values_class1 = shap_values[1]

    # 计算全局 SHAP 重要性
    shap_importance = pd.DataFrame({
        "feature": top_features,
        "shap_importance": np.abs(shap_values_class1).mean(axis=0)
    }).sort_values("shap_importance", ascending=False)

    # 保存
    os.makedirs("../result/test/ppi", exist_ok=True)
    shap_importance.to_csv("../result/test/ppi/rf_top300_shap_importance.csv", index=False)

    # 可视化 SHAP summary (点图)
    shap.summary_plot(shap_values_class1, X, feature_names=top_features, show=False)
    plt.tight_layout()
    plt.savefig("../result/test/ppi/rf_top300_shap_summary.png", dpi=300)
    plt.close()

    # 可视化 SHAP bar 图
    shap.summary_plot(shap_values_class1, X, feature_names=top_features, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("../result/test/ppi/rf_top300_shap_bar.png", dpi=300)
    plt.close()



def ppiBinary_rf_test_paperdataset_shap_analysis_new():
    # ===================== 数据准备 =====================
    dt_mapping_0 = pd.read_csv("../data/AllDataIdMapping.csv", sep="\t")
    dt_mapping_0 = dt_mapping_0[dt_mapping_0["belong"]=="861paper"].reset_index(drop=True)
    target_taxid = list(dt_mapping_0["Taxid"])

    # 获得每个病毒不同阈值下的ppi
    df = pd.read_csv('../data/all_virus_ppi_matrix_threshold999.csv', index_col=0)
    df = df.groupby('taxid', as_index=False).max()
    
    # 合并标签
    df_label = pd.read_csv('../data/AllDataIdMapping.csv', sep='\t')
    df_label.rename(columns={'Taxid':'taxid', 'Label':'label'}, inplace=True)
    df = pd.merge(df, df_label[['label','taxid']], on='taxid')
    df = df[df["taxid"].isin(target_taxid)].reset_index(drop=True)
    df.set_index('taxid', inplace=True)
    print("Final dataset shape:", df.shape)
    
    # ===================== 取 Top300 特征 =====================
    feature_importances_df = pd.read_csv("../data/rf_feature_importances_100.csv")
    mean_importance = feature_importances_df.mean()
    top_features = mean_importance.sort_values(ascending=False).head(300).index.tolist()

    X = df[top_features].values
    y = np.array(df['label'])

    # ===================== GridSearchCV 找全局最优参数 =====================
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 15, 20, 25],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=64
    )
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    print("Best Params:", best_params)

    # ===================== 用最优参数训练全量模型 =====================
    final_model = RandomForestClassifier(**best_params, random_state=42)
    final_model.fit(X, y)

    # ===================== SHAP 分析 =====================
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)   # list[y_class]
    shap_values_class0 = shap_values[0]      # 对应 label=0
    shap_values_class1 = shap_values[1]      # 对应 label=1

    # 计算全局 SHAP 重要性
    shap_importance = pd.DataFrame({
        "feature": top_features,
        "shap_importance": np.abs(shap_values_class1).mean(axis=0)
    }).sort_values("shap_importance", ascending=False)

    # 保存
    os.makedirs("../result/test/ppi", exist_ok=True)
    shap_importance.to_csv("../result/test/ppi/rf_top300_shap_importance.csv", index=False)

    # =============== 保存散点图用的结果 ===============
    # 把每个样本 × 特征的 shap 值保存下来，便于绘制蓝红点图
    shap_df_class0 = pd.DataFrame(shap_values_class0, columns=top_features, index=df.index)
    shap_df_class1 = pd.DataFrame(shap_values_class1, columns=top_features, index=df.index)

    shap_df_class0.to_csv("../result/test/ppi/rf_top300_shap_values_class0.csv")
    shap_df_class1.to_csv("../result/test/ppi/rf_top300_shap_values_class1.csv")

    # =============== 可视化 ===============
    # SHAP summary (散点图, 蓝=class0, 红=class1)
    plt.figure(figsize=(10,6))
    cmap = plt.get_cmap("RdBu")
    shap.summary_plot(shap_values_class1, X, feature_names=top_features, show=False, color=cmap)
    plt.tight_layout()
    plt.savefig("../result/test/ppi/rf_top300_shap_summary_scatter.png", dpi=300)
    plt.close()

    # SHAP bar 图 (平均重要性)
    cmap = plt.get_cmap("RdBu")
    shap.summary_plot(shap_values_class1, X, feature_names=top_features, show=False, color=cmap)
    plt.tight_layout()
    plt.savefig("../result/test/ppi/rf_top300_shap_bar.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    print("START:", time.ctime(), flush=True)
    ppiBinary_rf_test_paperdataset_shap_analysis_new()
    print("END:", time.ctime(), flush=True)
