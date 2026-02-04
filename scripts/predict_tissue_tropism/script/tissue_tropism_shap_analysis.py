import shap
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
import matplotlib.pyplot as plt


def shap_analysis_tissue_tropism():
    # ========= 1. 加载 PPI 特征 =========
    df = pd.read_csv('./data/all_virus_ppi_matrix_threshold999_214.csv', index_col=0)
    df = df.groupby('taxid', as_index=False).max()

    # ========= 2. 加载标签 =========
    df_label = pd.read_csv('./data/final_214_genome_proteome_label.csv')
    df_label.rename(columns={'Taxid': 'taxid'}, inplace=True)
    df_label.dropna(subset=['Tp.primary'], inplace=True)

    # 合并标签
    df_xy = pd.merge(df, df_label[['taxid', 'Tp.primary']], on='taxid')
    df_xy.dropna(subset=['Tp.primary'], inplace=True)
    df_xy.set_index('taxid', inplace=True)

    # 合并小类
    df_xy['Tp.primary'] = df_xy['Tp.primary'].replace({
        'vascular': 'other',
        'hepatic': 'other',
        'circulatory': 'other'
    })
    classes = np.unique(df_xy['Tp.primary'])
    print("标签分布：")
    print(df_xy['Tp.primary'].value_counts())

    # ========= 3. 特征重要性 =========
    top_num = 1250
    feature_importances_df = pd.read_csv("./data/rf_feature_importances_100split_Tpprimary_20other.csv")
    mean_importance = feature_importances_df.mean()
    top_features = mean_importance.sort_values(ascending=False).head(top_num).index.tolist()

    X = df_xy[top_features].values
    y = np.array(df_xy['Tp.primary'])
    taxids = df_xy.index.values

    # ========= 4. 训练 SVM (GridSearch) =========
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
    final_model.fit(X, y)

    # ========= 5. SHAP 分析 =========
    # 用部分样本作为 background，加速计算
    background = shap.sample(X, 50, random_state=42)
    explainer = shap.KernelExplainer(final_model.predict_proba, background)

    shap_values = explainer.shap_values(X)  # list[num_classes]

    # ========= 6. 计算特征重要性 =========
    importance_list = []
    for class_idx, class_name in enumerate(classes):
        class_shap = np.abs(shap_values[class_idx]).mean(axis=0)
        importance_list.append(pd.DataFrame({
            "feature": top_features,
            "class": class_name,
            "shap_importance": class_shap
        }))

    shap_importance = pd.concat(importance_list, axis=0)
    shap_importance = shap_importance.sort_values("shap_importance", ascending=False)

    # 只保存前100个特征
    shap_importance.head(100).to_csv(
        "./result/tissue_tropism_shap_importance_svm.csv", index=False
    )

    # ========= 7. 可视化 =========
    shap.summary_plot(shap_values, X, feature_names=top_features, show=False)
    plt.tight_layout()
    plt.savefig("./result/tissue_tropism_shap_analysis.png", dpi=300)
    plt.close()



if __name__ == "__main__":
    print("START: ", time.ctime(), flush=True)
    shap_analysis_tissue_tropism()
    print("END: ", time.ctime(), flush=True)
