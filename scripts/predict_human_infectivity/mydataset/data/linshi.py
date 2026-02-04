import pandas as pd

dt = pd.read_csv("./all_virus_ppi_matrix_threshold999.csv", sep=",", header=0)
print(dt.shape)
print(dt.columns.tolist()[0:3])
