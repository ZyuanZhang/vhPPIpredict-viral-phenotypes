import os
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.decomposition import PCA


# ========== 1. 固定随机种子 ==========
def seed_torch(seed=20):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ========== 2. 载入病毒和宿主蛋白 PLM embedding ==========
def get_PLMfeatures_dict():
    dict_plm_embedding = {}

    vpath = "/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedFengYang/virusProtPLM/"
    hpath = "/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedFengYang/humanProtPLM/"
    vpath_test = "/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedFY_low_evidence_virus/virusProtPLM/"

    # 载入训练病毒蛋白
    for vf in os.listdir(vpath):
        vid = vf[:-3]
        vembed = torch.load(vpath + vf, weights_only=True).tolist()
        dict_plm_embedding[vid] = vembed

    # 载入宿主蛋白
    for hf in os.listdir(hpath):
        hid = hf[:-3]
        hembed = torch.load(hpath + hf, weights_only=True).tolist()
        dict_plm_embedding[hid] = hembed

    # 载入低证据病毒
    for vf in os.listdir(vpath_test):
        vid = vf[:-3]
        vembed = torch.load(vpath_test + vf, weights_only=True).tolist()
        dict_plm_embedding[vid] = vembed

    return dict_plm_embedding


# ========== 3. 病毒层级特征聚合 ==========
def aggregate_features_by_virus(node_features, virus_to_index, virus_list):
    aggregated_features = []
    for virus in virus_list:
        virus_features = node_features[virus_to_index[virus]]
        aggregated_features.append(virus_features.mean(dim=0))
    return torch.stack(aggregated_features)


# ========== 4. 定义GCN模型 ==========
class GCN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim,
                 virus_list, dict_taxid4viruspro):
        super(GCN_Model, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.fc1 = nn.Linear(hidden_dim2, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, output_dim)
        self.virus_list = virus_list
        self.dict_taxid4viruspro = dict_taxid4viruspro

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        aggregated_features = aggregate_features_by_virus(
            x, self.dict_taxid4viruspro, self.virus_list
        )
        x = F.relu(self.fc1(aggregated_features))
        x = self.fc2(x)
        return x


# ========== 5. 模型训练 + 独立集预测 ==========
def test_model_on_independent_set(node_features, edge_index,
                                  node_features_test, edge_index_test,
                                  virus_list, dict_taxid4viruspro,
                                  virus_list_test, dict_taxid4viruspro_test,
                                  dict_taxid4table,
                                  repeats=5, epoch_num=45, lr=0.0005, device='cuda'):

    criterion = BCEWithLogitsLoss()
    train_labels_tensor = torch.tensor(
        [dict_taxid4table[v] for v in virus_list],
        dtype=torch.float
    ).view(-1, 1).to(device)

    model = GCN_Model(
        input_dim=node_features.shape[1],
        hidden_dim1=512,
        hidden_dim2=256,
        output_dim=1,
        virus_list=virus_list,
        dict_taxid4viruspro=dict_taxid4viruspro
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- Train ----
    for epoch in range(epoch_num):
        model.train()
        optimizer.zero_grad()
        out = model(node_features.to(device), edge_index.to(device))
        loss_train = criterion(out, train_labels_tensor)
        loss_train.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epoch_num}]  Loss: {loss_train.item():.4f}", flush=True)

    # ---- Test ----
    model.eval()
    model.dict_taxid4viruspro = dict_taxid4viruspro_test
    model.virus_list = virus_list_test

    with torch.no_grad():
        out_test = model(node_features_test.to(device), edge_index_test.to(device))
        pred_probs = torch.sigmoid(out_test).cpu().numpy().flatten()

        dt_output = pd.DataFrame({
            "virus": virus_list_test,
            "virulence": pred_probs
        })
        dt_output.to_csv("./low_evidence_virus_Virulence.csv", sep="\t", index=False)

    print("Independent test prediction saved to ./low_evidence_virus_Virulence.csv", flush=True)


# ========== 6. 主程序 ==========
if __name__ == "__main__":
    print("START: ", time.ctime(), flush=True)
    seed_torch()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- 加载训练数据 ----
    df = pd.read_csv("/data/150T/databases/help_zhangzhiyuan/vhPPIpredCasesDT/case3_pred_virulence_v214/pred_ppi_norm_score_v214.txt", sep="\t")
    df = df.sort_values(by="vtaxid")

    dict_plm_embedding = get_PLMfeatures_dict()

    # 构建节点
    node_viruspro = list(set(df["virus_unid"]))
    node_humanpro = list(set(df["human_unid"]))
    nodes = sorted(node_viruspro + node_humanpro)

    node_features = torch.tensor([dict_plm_embedding[v] for v in nodes], dtype=torch.float)
    print(f"Training node feature matrix: {node_features.shape}", flush=True)

    # edge_index
    vhid_to_index = {v: i for i, v in enumerate(nodes)}
    edge_index = torch.tensor([
        [vhid_to_index[df["virus_unid"].iloc[i]] for i in range(df.shape[0])],
        [vhid_to_index[df["human_unid"].iloc[i]] for i in range(df.shape[0])]
    ], dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # 双向边

    # label
    df_grouped = df.groupby("vtaxid")
    dict_taxid4viruspro, dict_taxid4table = {}, {}
    for taxid, gp in df_grouped:
        dict_taxid4table[taxid] = int(gp["label"].iloc[0])
        dict_taxid4viruspro[taxid] = [nodes.index(i) for i in list(set(gp["virus_unid"]))]
    virus_list = sorted(dict_taxid4viruspro.keys())

    # ---- 加载测试数据 ----
    df_test = pd.read_csv("/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedFY_low_evidence_virus/low_evidence_vhppi_pred_result.txt", sep="\t")
    df_test = df_test.sort_values(by="vtaxid")

    node_viruspro_test = list(set(df_test["virus_unid"]))
    node_humanpro_test = list(set(df_test["human_unid"]))
    nodes_test = sorted(node_viruspro_test + node_humanpro_test)

    node_features_test = torch.tensor([dict_plm_embedding[v] for v in nodes_test], dtype=torch.float)
    print(f"Test node feature matrix: {node_features_test.shape}", flush=True)

    vhid_to_index_test = {v: i for i, v in enumerate(nodes_test)}
    edge_index_test = torch.tensor([
        [vhid_to_index_test[df_test["virus_unid"].iloc[i]] for i in range(df_test.shape[0])],
        [vhid_to_index_test[df_test["human_unid"].iloc[i]] for i in range(df_test.shape[0])]
    ], dtype=torch.long)
    edge_index_test = torch.cat([edge_index_test, edge_index_test.flip(0)], dim=1)

    df_grouped_test = df_test.groupby("vtaxid")
    dict_taxid4viruspro_test, virus_list_test = {}, []
    for taxid, gp in df_grouped_test:
        virus_list_test.append(taxid)
        dict_taxid4viruspro_test[taxid] = [nodes_test.index(i) for i in list(set(gp["virus_unid"]))]
    virus_list_test = sorted(virus_list_test)

    # ---- 运行 ----
    test_model_on_independent_set(
        node_features, edge_index,
        node_features_test, edge_index_test,
        virus_list, dict_taxid4viruspro,
        virus_list_test, dict_taxid4viruspro_test,
        dict_taxid4table,
        repeats=5, epoch_num=45, lr=0.0005, device=device
    )

    print("END: ", time.ctime(), flush=True)

