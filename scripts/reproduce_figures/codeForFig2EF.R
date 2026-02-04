library(enrichplot)
library(DOSE)
library(clusterProfiler)
library(org.Hs.eg.db)
library(ggplot2)
library(patchwork)
library(ggtree)

setwd("/Users/zhiyuanzhang/Documents/GitHub/vhPPIpredict-viral-phenotypes/")
df = read.csv("../data/rf_top450_shap_importance_with_gene.csv", sep = ",", header=TRUE)
df$gene = as.character(df$gene)
gene_list = df$gene

# GO 富集
edo = enrichGO(gene=gene_list,
               OrgDb='org.Hs.eg.db', 
               ont="BP", 
               pvalueCutoff=0.5)

# 转换成可读形式
edox <- setReadable(edo, 'org.Hs.eg.db', 'ENTREZID')

# ==== 1. 筛选 virus/viral ====
res_virus <- edox@result[grep("viral|virus", edox@result$Description, ignore.case = TRUE), ]
edox_virus <- edox
edox_virus@result <- res_virus
edox2_virus <- pairwise_termsim(edox_virus)

p1 <- treeplot(edox2_virus,
               nCluster=4,   # 固定为 4 类
               showCategory=15,
               nWords=15,
               label_format_cladelab=30,
               group_color=c("#E41A1C","#377EB8","#4DAF4A","#984EA3"),
               label_format=20) +
  geom_tree(size=0.8) +
  ggtitle("GO-BP terms related to Virus")+
  theme(plot.title = element_text(hjust = 0.5, size=16, face="bold"))



# ==== 2. 筛选 immune ====
res_immune <- edox@result[grep("immune|inflamma|lymphocyte", edox@result$Description, ignore.case = TRUE), ]
edox_immune <- edox
edox_immune@result <- res_immune
edox2_immune <- pairwise_termsim(edox_immune)

p2 <- treeplot(edox2_immune,
               nCluster=4,   # 固定为 4 类
               showCategory=15,
               nWords=15,
               label_format_cladelab=30,
               label_format=20,
               group_color=c("#FF7F00","#66C2A5","#999999", "#A65628")) +
  geom_tree(size=0.8) +
  ggtitle("GO-BP terms related to Immune Response")+
  theme(plot.title = element_text(hjust = 0.5, size=16, face="bold"))


