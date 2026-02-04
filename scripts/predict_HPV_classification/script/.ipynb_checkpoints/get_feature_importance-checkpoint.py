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


def feature_importance_genome_kmer():
    for k in [3,4,5]:
        # 基于 Genome-kmer 为特征
        df = pd.read_csv(f'../data/hpv_genome_{k}mer.csv').rename(columns={'Unnamed: 0': 'Taxid', 'risk_label':'Label'})
        df.set_index('Taxid', inplace=True)
    
        # 2. 准备特征和标签
        X = df.drop(columns=['Label']).values
        y = df['Label'].values
        
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
        
        # 随机划分 100 次
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
            #break
            
        #print(feature_importances_list)
        # 保存特征重要性
        feature_importances_df = pd.DataFrame(feature_importances_list, columns=df.drop(columns=['Label']).columns)
        feature_importances_df.to_csv(f"../feature_importances/rf_feature_importances_genome_{k}mer.csv", index=False)



def feature_importance_proteome_kmer():
    for k in [1,2,3]:
        # 基于 Genome-kmer 为特征
        df = pd.read_csv(f'../data/hpv_proteome_{k}mer.csv').rename(columns={'Unnamed: 0': 'Taxid', 'risk_label':'Label'})
        df.set_index('Taxid', inplace=True)
    
        # 2. 准备特征和标签
        X = df.drop(columns=['Label']).values
        y = df['Label'].values
        
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
        
        # 随机划分 100 次
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
            #break
            
        #print(feature_importances_list)
        # 保存特征重要性
        feature_importances_df = pd.DataFrame(feature_importances_list, columns=df.drop(columns=['Label']).columns)
        feature_importances_df.to_csv(f"../feature_importances/rf_feature_importances_proteome_{k}mer.csv", index=False)




def get_topN_feat_ppi_binary_RF():
    # 获得每个病毒不同阈值下的ppi
    df = pd.read_csv('../data/ppi_binary_matrix.csv').rename(columns={'Unnamed: 0': 'taxid', 'risk_label': 'label'})
    df.set_index('taxid', inplace=True)
    
    # 读取100次划分下的特征重要性
    feature_importances_df = pd.read_csv("../feature_importances/rf_feature_importances_ppibinary.csv")
    mean_importance = feature_importances_df.mean()
    
    for top_num in range(50, 2050, 50):
        print(f"Top {top_num} features:")
        top_100_features = mean_importance.sort_values(ascending=False).head(top_num).index.tolist()
    
        # 进行随机85：15的随机划分，进行预测。
        X = df[top_100_features].values
        y = np.array(df['label'])
    
        # 初始化评估指标容器
        metrics_list = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
    
    
        # 随机划分 100 次
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
    
            # 定义参数搜索空间（可根据需要调整）
            param_grid = {
                'n_estimators': [5,10,15]
            }
    
            # 使用 3 折交叉验证进行网格搜索
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=train_val_index[0]),
                param_grid,
                cv=5,
                scoring='roc_auc',  # 以AUC作为选择指标
                n_jobs=32
            )
            grid_search.fit(X_train, y_train)
    
            # 使用最优参数训练模型
            best_clf = grid_search.best_estimator_
            best_clf.fit(X_train, y_train)
    
            # 预测与评估
            y_pred = best_clf.predict(X_val)
            y_prob = best_clf.predict_proba(X_val)[:, 1]
    
            metrics_list['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics_list['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            metrics_list['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            metrics_list['f1'].append(f1_score(y_val, y_pred, zero_division=0))
    
            try:
                auc = roc_auc_score(y_val, y_prob)
            except ValueError:
                auc = np.nan
            metrics_list['auc'].append(auc)
    
        # 结果汇总
        result_df = pd.DataFrame(metrics_list)
        summary = result_df.describe().loc[['mean', 'std']]
        print(summary)
    
        # 保存结果
        os.makedirs("../feature_importances/feat_res/", exist_ok=True)
        result_df.to_csv(f"../feature_importances/feat_res/rf_ppi_top{top_num}_result_val.csv", index=False)
        summary.to_csv(f"../feature_importances/feat_res/rf_ppi_top{top_num}_summary_val.csv")




if __name__ == "__main__":
    print("START: ",time.ctime(), flush=True)
    #feature_importance_binary_ppi()
    #feature_importance_genome_kmer()
    #feature_importance_proteome_kmer()
    get_topN_feat_ppi_binary_RF()
    print("END: ",time.ctime(), flush=True)