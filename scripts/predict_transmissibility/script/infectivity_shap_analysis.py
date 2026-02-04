import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import shap
import time
import matplotlib.pyplot as plt


def shap_analysis_infectivity():
    # 加载 PPI 特征
    df = pd.read_csv('./data/all_virus_ppi_matrix_threshold999_214.csv', index_col=0)
    df = df.groupby('taxid', as_index=False).max()

    # 加载标签
    df_label = pd.read_csv('./data/final_214_genome_proteome_label.csv')
    df_label.rename(columns={'Taxid': 'taxid'}, inplace=True)
    df_label.dropna(subset=['Tr.level'], inplace=True)

    # 合并标签
    df_xy = pd.merge(df, df_label[['taxid', 'Tr.level']], on='taxid')
    df_xy.dropna(subset=['Tr.level'], inplace=True)
    df_xy.set_index('taxid', inplace=True)
    classes = np.unique(df_xy['Tr.level'])

    # 特征重要性
    top_num = 200
    feature_importances_df = pd.read_csv(f"./data/rf_feature_importances_100split_Trlevel.csv")
    mean_importance = feature_importances_df.mean()
    top_features = mean_importance.sort_values(ascending=False).head(top_num).index.tolist()

    X = df_xy[top_features].values
    y = np.array(df_xy['Tr.level'])

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

    # ========= 4. SHAP 分析 (全量 200+ 样本) =========
    # 用所有样本做 background
    explainer = shap.KernelExplainer(final_model.predict_proba, X)

    # 对全量样本计算 SHAP 值
    shap_values = explainer.shap_values(X)  # list[num_classes]

    # ========= 5. 计算 feature importance =========
    importance_list = []
    for class_idx, class_name in enumerate(classes):
        class_shap = np.abs(shap_values[class_idx]).mean(axis=0)
        importance_list.append(pd.DataFrame({
            "feature": top_features,
            "class": class_name,
            "shap_importance": class_shap
        }))

    shap_importance = pd.concat(importance_list, axis=0)
    shap_importance.sort_values("shap_importance", ascending=False, inplace=True)

    shap_importance.to_csv("./result/infectivity_shap_importance_svm.csv", index=False)

    # ========= 6. 可选: 可视化 top 特征 =========
    shap.summary_plot(shap_values, X, feature_names=top_features, class_names=classes, show=False)
    plt.tight_layout()
    plt.savefig("./result/infectivity_route_shap_analysis.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    print("START: ",time.ctime(), flush=True)

    shap_analysis_infectivity()

    print("END: ",time.ctime(), flush=True)
