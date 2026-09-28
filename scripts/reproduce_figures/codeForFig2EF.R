library(enrichplot)
library(DOSE)
library(clusterProfiler)
library(org.Hs.eg.db)
library(ggplot2)
library(patchwork)
library(ggtree)

setwd("/Users/zhiyuanzhang/Documents/GitHub/vhPPIpredict-viral-phenotypes/")
df = read.csv("./data_updated/top1000_genes_for_enrichment.csv", sep = ",", header=TRUE)
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









library(enrichplot)
library(DOSE)
library(clusterProfiler)
library(org.Hs.eg.db)
library(ggplot2)
library(patchwork)
library(ggtree)

setwd("/Users/zhiyuanzhang/Documents/GitHub/vhPPIpredict-viral-phenotypes/")

# ============================================================
# 1. Read input data
# ============================================================

df <- read.csv(
  "./data_updated/top1000_genes_for_enrichment.csv",
  sep = ",",
  header = TRUE
)

# UniProt accession
df$unid <- as.character(df$unid)

print(head(df))


# ============================================================
# 2. Convert UniProt ID to ENTREZID
# ============================================================

gene_mapping <- bitr(
  df$unid,
  fromType = "UNIPROT",
  toType = "ENTREZID",
  OrgDb = org.Hs.eg.db
)

print(head(gene_mapping))

# Remove duplicated ENTREZ IDs
gene_mapping <- gene_mapping[
  !duplicated(gene_mapping$ENTREZID),
]

gene_list <- gene_mapping$ENTREZID

cat("Input UniProt IDs:", length(unique(df$unid)), "\n")
cat("Mapped Entrez IDs:", length(unique(gene_list)), "\n")


# ============================================================
# 3. GO Biological Process enrichment
# ============================================================

edo <- enrichGO(
  gene = gene_list,
  OrgDb = org.Hs.eg.db,
  keyType = "ENTREZID",
  ont = "BP",
  pvalueCutoff = 0.5,
  pAdjustMethod = "BH",
  readable = FALSE
)


# ============================================================
# 4. Convert Entrez IDs to gene symbols
# ============================================================

edox <- setReadable(
  edo,
  OrgDb = org.Hs.eg.db,
  keyType = "ENTREZID"
)


# ============================================================
# 5. Virus-related GO terms
# ============================================================

res_virus <- edox@result[
  grep(
    "viral|virus",
    edox@result$Description,
    ignore.case = TRUE
  ),
]

cat("Virus-related GO terms:", nrow(res_virus), "\n")

edox_virus <- edox
edox_virus@result <- res_virus

edox2_virus <- pairwise_termsim(edox_virus)


p1 <- treeplot(
  edox2_virus,
  nCluster = 4,
  showCategory = 10,
  nWords = 3,
  
  group_color = c(
    "#E41A1C",
    "#377EB8",
    "#4DAF4A",
    "#984EA3"
  ),
  label_format = 18
) +
  geom_tree(size = 0.8) +
  ggtitle("GO-BP terms related to Virus") +
  theme(
    plot.title = element_text(
      hjust = 0.5,
      size = 14,
      face = "bold"
    )
  )


# ============================================================
# 6. Immune-related GO terms
# ============================================================

res_immune <- edox@result[
  grep(
    "immune|inflamma|lymphocyte",
    edox@result$Description,
    ignore.case = TRUE
  ),
]

cat("Immune-related GO terms:", nrow(res_immune), "\n")

edox_immune <- edox
edox_immune@result <- res_immune

edox2_immune <- pairwise_termsim(edox_immune)


# 按完整单词换行
wrap_by_word <- function(x, width = 18) {
  vapply(
    x,
    function(s) paste(
      strwrap(s, width = width),
      collapse = "\n"
    ),
    character(1)
  )
}

# 对 immune GO term 的 Description 进行换行
edox_immune@result$Description <- wrap_by_word(
  edox_immune@result$Description,
  width=46
  #width = 64
)

# 重新计算 similarity
edox2_immune <- pairwise_termsim(edox_immune)

# 绘图
p2 <- treeplot(
  edox2_immune,
  nCluster = 4,
  showCategory = 10,
  nWords = 0,
  
  # 已经手动换行，不让 treeplot 再处理
  label_format = 100,
  
  group_color = c(
    "#FF7F00",
    "#66C2A5",
    "#999999",
    "#A65628"
  )
) +
  geom_tree(size = 0.8) +
  ggtitle("GO-BP terms related to Immune Response") +
  theme(
    plot.title = element_text(
      hjust = 0.5,
      size = 14,
      face = "bold"
    )
  )




# ============================================================
# 7. Plot
# ============================================================

p1
p2