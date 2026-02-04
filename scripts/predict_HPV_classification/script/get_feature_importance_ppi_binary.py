import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import time
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split

def feature_importance_binary_ppi():
    # 获得每个病毒不同阈值下的ppi
    df = pd.read_csv('../data/ppi_binary_matrix.csv').rename(columns={'Unnamed: 0': 'taxid', 'risk_label': 'label'})
    #print(list(df.columns)[0:10])
    df.set_index('taxid', inplace=True)
    print(df)
    
    # 进行随机85：15的随机划分，进行预测。
    X = np.array(df.drop(columns=['label'])) # 筛选特征重要性使用
    y = np.array(df['label'])
    print(X.shape) # (861, 12039)
    # # PCA降维
    # pca = PCA(n_components=256, random_state=42)
    # X = pca.fit_transform(X)
    # print(X.shape)
    
    # 初始化评估指标容器
    metrics_list = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    
    # 特征重要性
    feature_importances_list = []
    
    # 随机划分 100 次  70:15:15
    # 先划分 15% 测试集
    sss = StratifiedShuffleSplit(n_splits=100, test_size=0.15, random_state=42)
    
    for train_val_index, test_index in sss.split(X, y):
        # 15% 测试集
        X_train_val, X_test = X[train_val_index], X[test_index]
        y_train_val, y_test = y[train_val_index], y[test_index]
    
        # 从 85% 的训练+验证集中划出 17.65% ≈ 15% 的验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=0.1765,  # 15 / 85
            stratify=y_train_val,
            random_state=42
        )
    
        # 训练随机森林
        clf = RandomForestClassifier(random_state=train_val_index[0])
        clf.fit(X_train, y_train)
    
        # 保存特征重要性
        feature_importances_list.append(clf.feature_importances_)
    
    # 保存特征重要性
    feature_importances_df = pd.DataFrame(feature_importances_list, columns=df.drop(columns=['label']).columns)
    feature_importances_df.to_csv("../feature_importances/rf_feature_importances_ppibinary.csv", index=False)







if __name__ == "__main__":
    print("START: ",time.ctime(), flush=True)
    feature_importance_binary_ppi()
    print("END: ",time.ctime(), flush=True)