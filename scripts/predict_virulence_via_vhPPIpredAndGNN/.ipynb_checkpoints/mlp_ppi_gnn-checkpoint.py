import os
import random
from concurrent.futures import ProcessPoolExecutor
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import time


def seed_torch(seed=20):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def get_PLMfeatures_dict():
    dict_plm_embedding = {}
    vpath, hpath = "/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedFengYang/virusProtPLM/",\
                    "/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedFengYang/humanProtPLM/"
    
    for vf in os.listdir(vpath):
        vid = vf[0:-3]
        vembed = torch.load(vpath + vf, weights_only=True).tolist()
        dict_plm_embedding[vid] = vembed
    
    for hf in os.listdir(hpath):
        hid = hf[0:-3]
        hembed = torch.load(hpath + hf, weights_only=True).tolist()
        dict_plm_embedding[hid] = hembed

    vpath_test = "/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedFY_low_evidence_virus/virusProtPLM/"
    for vf in os.listdir(vpath_test):
        vid = vf[0:-3]
        vembed = torch.load(vpath_test + vf, weights_only=True).tolist()
        dict_plm_embedding[vid] = vembed
        
    return dict_plm_embedding

"""Model"""


def aggregate_features_by_virus(node_features, virus_to_index):
    aggregated_features = []
    # 对于每个病毒，聚合其所有蛋白特征
    for virus in virus_list:
        virus_features = node_features[virus_to_index[virus]]
        aggregated_features.append(virus_features.mean(dim=0))

    return torch.stack(aggregated_features)


class GCN_Model(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim1=512, hidden_dim2=256, output_dim=1):
        super(GCN_Model, self).__init__()

        # 图卷积层
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)

        # 多层感知机（MLP）
        self.fc1 = nn.Linear(hidden_dim2, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x, edge_index):
        # GCN层
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)  # Dropout 防止过拟合
        x = self.conv2(x, edge_index)
        aggregated_features = aggregate_features_by_virus(x, dict_taxid4viruspro)
        # MLP层
        x = F.relu(self.fc1(aggregated_features))
        x = self.fc2(x)

        return x


def test_model_on_independent_set(repeats=5, epoch_num=45, lr=0.0005, device='cuda'):
    criterion = BCEWithLogitsLoss()
    sss = StratifiedShuffleSplit(n_splits=repeats, test_size=0.2, random_state=76)
    test_metrics = []

    virus_train = virus_list
    #table_list = table_list
    labels_train = [dict_taxid4table[v] for v in virus_train]
    
    virus_test = virus_list_test
    #table_list_test = table_list_test

    train_labels_tensor = torch.tensor(labels_train, dtype=torch.float).view(-1, 1).to(device)

    model = GCN_Model(input_dim=node_features.shape[1], hidden_dim1=512, hidden_dim2=256, output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch_num):
        model.train()
        optimizer.zero_grad()
        out = model(node_features.to(device), edge_index.to(device))
        out_train = out[[virus_list.index(v) for v in virus_train]]
        loss_train = criterion(out_train, train_labels_tensor)
        loss_train.backward()
        optimizer.step()

    # 测试
    model.eval()
    with torch.no_grad():
        out = model(node_features.to(device), edge_index.to(device))
        out_test = out[[virus_list_test.index(v) for v in virus_test]]

        pred_probs = torch.sigmoid(out_test).cpu().numpy().flatten().tolist()
        dt_output = [[virus_test[i], pred_probs[i]] for i in range(len(virus_test))]
        dt_output = pd.DataFrame(dt_output, columns = ["virus", "virulence"])

        dt_output.to_csv("./low_evidence_virus_Virulence.csv", sep="\t", header=0)

        
    
if __name__ == "__main__":
    print("START: ", time.ctime(), flush=True)
    
    seed_torch()

    df = pd.read_csv("/data/150T/databases/help_zhangzhiyuan/vhPPIpredCasesDT/case3_pred_virulence_v214/pred_ppi_norm_score_v214.txt", sep="\t")
    df = df.sort_values(by="vtaxid")
    
    # 节点特征 边 标签 # data = Data(x=node_features, edge_index=edge_index, y=label)
    # 得到 nodes 列表
    node_viruspro = list(set(df["virus_unid"]))  # 927
    node_humanpro = list(set(df["human_unid"]))  # 6110
    nodes = node_viruspro + node_humanpro
    nodes.sort()  # 规定了节点的顺序

    dict_plm_embedding = get_PLMfeatures_dict()
    node_features = torch.tensor([dict_plm_embedding[v] for v in nodes], dtype=torch.float)

    # 得到 edge_index 列表
    vhid_to_index = {v: i for i, v in enumerate(nodes)}
    edge_index = torch.tensor([
        [vhid_to_index[df["virus_unid"][i]] for i in range(df.shape[0])],
        [vhid_to_index[df["human_unid"][i]] for i in range(df.shape[0])]
    ], dtype=torch.long)

    # 得到 label 列表
    prelabel = "label"
    df_grouped = df.groupby("vtaxid")
    dict_taxid4viruspro = {}  # 用于得到病毒的嵌入
    dict_taxid4table = {}
    for taxid, gp in df_grouped:
        gp.reset_index(drop=True, inplace=True)
        dict_taxid4table[taxid] = gp[prelabel][0]
        dict_taxid4viruspro[taxid] = [nodes.index(i) for i in list(set(gp["virus_unid"]))]
    virus_list = []  # 后期计算损失时， 按照此顺序取出相应的嵌入进行概率的计算。
    tables_list = []
    for taxid, table in dict_taxid4table.items():
        virus_list.append(taxid)
    virus_list = sorted(virus_list)
    #for virus in virus_list:
    #    tables_list.append(dict_taxid4table[virus])

    ## Test load
    df_test = pd.read_csv("/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedFY_low_evidence_virus/low_evidence_vhppi_pred_result.txt", sep="\t")
    df_test = df_test.sort_values(by="vtaxid")
    
    # 节点特征 边 标签 # data = Data(x=node_features, edge_index=edge_index, y=label)
    # 得到 nodes 列表
    node_viruspro_test = list(set(df_test["virus_unid"]))  # 927
    node_humanpro_test = list(set(df_test["human_unid"]))  # 6110
    nodes_test = node_viruspro_test + node_humanpro_test
    nodes_test.sort()  # 规定了节点的顺序

    node_features_test = torch.tensor([dict_plm_embedding[v] for v in nodes_test], dtype=torch.float)

    # 得到 edge_index 列表
    vhid_to_index_test = {v: i for i, v in enumerate(nodes_test)}
    edge_index_test = torch.tensor([
        [vhid_to_index_test[df_test["virus_unid"][i]] for i in range(df_test.shape[0])],
        [vhid_to_index_test[df_test["human_unid"][i]] for i in range(df_test.shape[0])]
    ], dtype=torch.long)

    # 得到 label 列表
    df_grouped_test = df_test.groupby("vtaxid")
    virus_list_test = []
    #dict_taxid4viruspro_test = {}  # 用于得到病毒的嵌入
    for taxid, gp in df_grouped_test:
        gp.reset_index(drop=True, inplace=True)
        virus_list_test.append(taxid)
        dict_taxid4viruspro[taxid] = [nodes_test.index(i) for i in list(set(gp["virus_unid"]))]
    
    virus_list_test = sorted(virus_list_test)
    
    ## RUNNING
    test_model_on_independent_set()

    print("END: ", time.ctime(), flush=True)