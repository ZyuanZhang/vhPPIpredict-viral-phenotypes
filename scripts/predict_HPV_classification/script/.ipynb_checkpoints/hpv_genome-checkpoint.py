import pandas as pd
import os
import numpy as np
from collections import defaultdict, Counter
from itertools import product
from Bio import SeqIO
import time
from multiprocessing import Pool, cpu_count
from collections import defaultdict


def splitKmer(geneseq, k=3):
    """计算k-mer组成"""
    return Counter(geneseq[i:i+k] for i in range(len(geneseq) - k + 1))


def process_sequence(args):
    """处理单个序列的 k-mer 计算"""
    seq_id, base_path, k = args
    file_path = os.path.join(base_path, f"{seq_id}.fasta")
    if not os.path.exists(file_path):
        return seq_id, {}
    with open(file_path, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            return seq_id, splitKmer(str(record.seq), k)
    return seq_id, {}


def load_sequences_parallel(ids, base_path, k=3, n_processes=None):
    if n_processes is None:  # 如果未指定，使用最大可用CPU数
        n_processes = cpu_count()
    with Pool(processes=n_processes) as pool:
        args = [(seq_id, base_path, k) for seq_id in ids]
        results = pool.map(process_sequence, args)
    return {seq_id: kmer for seq_id, kmer in results}


def getMatrixKmer(k=3):
    dt_virus_cluster = pd.read_csv("../data/hpv_virus_cluster_cluster.tsv", sep="\t", names=["cluster","representative"])
    dt_human_cluster = pd.read_csv("../data/hpv_human_cluster_cluster.tsv", sep="\t", names=["cluster","representative"])
    dict_vid2clu = {dt_virus_cluster["representative"][i]:dt_virus_cluster["cluster"][i] for i in range(dt_virus_cluster.shape[0])}
    dict_hid2clu = {dt_human_cluster["representative"][i]:dt_human_cluster["cluster"][i] for i in range(dt_human_cluster.shape[0])}
    
    dt_hpv_ppi = pd.DataFrame(data=None, columns=["virus_unid","human_unid","pred_score","pred_label"])
    fList = ["/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase1/result_pred_interactions/"+f for f in os.listdir("/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase1/result_pred_interactions/") if f.endswith("_predscore.txt")]
    for f in fList:
        dt_tmp = pd.read_csv(f, sep="\t", header=0)
        dt_tmp_pos = dt_tmp[dt_tmp["pred_label"]==1.0].reset_index(drop=True)
        dt_hpv_ppi = pd.concat([dt_hpv_ppi, dt_tmp_pos], ignore_index=True)
    dt_hpv_ppi["virus_clu"] = [dict_vid2clu[v] for v in dt_hpv_ppi["virus_unid"]]
    dt_hpv_ppi["human_clu"] = [dict_hid2clu[h] for h in dt_hpv_ppi["human_unid"]]
    
    dt_hpv_info = pd.read_excel("../data/HPV_subspecies_info_from_2019_cell.xlsx", sheet_name="Sheet1", header=0)
    dict_hpv_subspecies_label = {}
    for i in range(dt_hpv_info.shape[0]):
        if dt_hpv_info["Label"][i]=="HR":
            dict_hpv_subspecies_label[dt_hpv_info["Abbreviation"][i].strip()]=1
        else:
            dict_hpv_subspecies_label[dt_hpv_info["Abbreviation"][i].strip()]=0
    
    dt_hpv_subspecies = []
    for i in range(dt_hpv_ppi.shape[0]):
        vname, vid, hid = dt_hpv_ppi["virus_unid"][i].split("_")[0], dt_hpv_ppi["virus_unid"][i], dt_hpv_ppi["human_unid"][i]
        dt_hpv_subspecies.append([vname, vid, hid, dict_hpv_subspecies_label[vname]])
    dt_hpv_subspecies = pd.DataFrame(dt_hpv_subspecies, columns=["virus_name","virus_unid","human_unid","label"])
    
    """构建k-mer特征矩阵"""
    dt_genome = dt_hpv_subspecies.drop_duplicates(subset=["virus_name","label"], keep="first", ignore_index=True)
    virus_ids = set(dt_genome["virus_name"])

    # 多进程加载序列并计算k-mer
    virus_kmer = load_sequences_parallel(virus_ids, "/data/150T/databases/help_zhangzhiyuan/PredictAllHumanVirusPpiDatasetEmbedCase1/originalDT/HPV_Genomes/", k, n_processes=20)

    # 所有k-mer组合
    gene_acids = "ACGT"
    kmer_combinations = [''.join(p) for p in product(gene_acids, repeat=k)]
    
    # 初始化特征矩阵
    row_ids = sorted(dt_genome["virus_name"].unique())
    feat_matrix = pd.DataFrame(0, index=row_ids, columns=kmer_combinations + ["risk_label"])

    # 填充特征矩阵
    for _, row in dt_genome.iterrows():
        vname = row["virus_name"]
        if row["virus_name"] in virus_kmer:
            for kmer, count in virus_kmer[row["virus_name"]].items():
                feat_matrix.at[vname, kmer] += count
        feat_matrix.loc[vname, "risk_label"] = row["label"]
    
    print(feat_matrix.shape)
    feat_matrix.to_csv(f"../data/hpv_genome_{k}mer.csv")


if __name__ == "__main__":
    print("START: ", time.ctime())
    getMatrixKmer(k=5)
    print("END: ", time.ctime())

