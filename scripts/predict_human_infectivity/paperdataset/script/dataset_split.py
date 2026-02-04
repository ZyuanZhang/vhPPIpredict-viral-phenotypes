import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# 读取标签数据
df = pd.read_csv("../data/AllDataIdMapping.csv", sep="\t")
df = df[df["belong"] == "861paper"].reset_index(drop=True)

y = df["Label"].values
taxids = df["Taxid"].values

sss = StratifiedShuffleSplit(
    n_splits=100,
    test_size=0.15,
    random_state=2025
)

splits = []

for i, (train_idx, test_idx) in enumerate(sss.split(taxids, y)):
    split_df = pd.DataFrame({
        "iteration": i + 1,
        "taxid": taxids,
        "set": ["train"] * len(taxids)
    })
    split_df.loc[test_idx, "set"] = "test"
    splits.append(split_df)

all_splits = pd.concat(splits, ignore_index=True)
all_splits.to_csv("../data/fixed_100_splits_taxid.csv", index=False)

