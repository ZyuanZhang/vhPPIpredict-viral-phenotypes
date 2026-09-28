import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import shap
import matplotlib.pyplot as plt
import os
import time

def ppiBinary_rf_test_mydataset_shap_analysis():
    # ===================== 数据准备 =====================
    df = pd.read_csv('../data/all_virus_ppi_matrix_threshold999.csv', index_col=0)
    df = df.groupby('taxid', as_index=False).max()
    
    df_label = pd.read_csv('../data/AllDataIdMapping.csv', sep='\t')
    df_label.rename(columns={'Taxid':'taxid', 'Label':'label'}, inplace=True)
    df = pd.merge(df, df_label[['label','taxid']], on='taxid')
    df.set_index('taxid', inplace=True)
    print("Final dataset shape:", df.shape)
    
    # ===================== 取 Top450 特征 =====================
    feature_importances_df = pd.read_csv("../data/rf_feature_importances_100.csv")
    mean_importance = feature_importances_df.mean()
    top_features = mean_importance.sort_values(ascending=False).head(450).index.tolist()

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

    # ===================== RF 自带的 feature_importances_ =====================
    rf_importance = pd.DataFrame({
        "feature": top_features,
        "rf_importance": final_model.feature_importances_
    }).sort_values("rf_importance", ascending=False)

    # ===================== SHAP 分析 =====================
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)   # list[y_class]，取正类
    shap_values_class1 = shap_values[1]

    shap_importance = pd.DataFrame({
        "feature": top_features,
        "shap_importance": np.abs(shap_values_class1).mean(axis=0)
    }).sort_values("shap_importance", ascending=False)

    # ===================== 合并对比 RF vs SHAP =====================
    importance_compare = pd.merge(rf_importance, shap_importance, on="feature", how="outer")

    # ===================== 保存结果 =====================
    os.makedirs("../result/test/ppi", exist_ok=True)
    rf_importance.to_csv("../result/test/ppi/rf_top450_importance.csv", index=False)
    shap_importance.to_csv("../result/test/ppi/rf_top450_shap_importance.csv", index=False)
    importance_compare.to_csv("../result/test/ppi/rf_top450_importance_compare.csv", index=False)

    # SHAP summary plot (点图)
    shap.summary_plot(shap_values_class1, X, feature_names=top_features, show=False)
    plt.tight_layout()
    plt.savefig("../result/test/ppi/rf_top450_shap_summary.png", dpi=300)
    plt.close()

    # SHAP bar 图
    shap.summary_plot(shap_values_class1, X, feature_names=top_features, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("../result/test/ppi/rf_top450_shap_bar.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    print("START:", time.ctime(), flush=True)
    ppiBinary_rf_test_mydataset_shap_analysis()
    print("END:", time.ctime(), flush=True)
