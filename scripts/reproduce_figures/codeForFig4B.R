library(ggsankey)
library(ggplot2)
library(dplyr)

setwd("/Users/zhiyuanzhang/Documents/GitHub/vhPPIpredict-viral-phenotypes/")
dt_v451_rna = read.csv("./mid_result_updated/low_confidience_predict/low_pred_res.txt", sep=",", header=TRUE)


df_v451 <- dt_v451_rna %>%
  #make_long(infect_human_label, transmission_route_label, tissue_tropism_pred_label, virulence_label, infectivity_label) 
  make_long(Genome.Type, Transmission.Route, Tissue.Tropism, Virulence, Transmissibility) 


p<-ggplot(df_v451, aes(x = x, 
               next_x = next_x, 
               node = node, 
               next_node = next_node,
               fill = factor(node),
               label = node)) +
  geom_sankey(flow.alpha = 0.5, node.color = 1) +
  geom_sankey_label(size = 3.5, color = 1, fill = "white") +
  scale_fill_viridis_d(alpha = 0.95) +
  theme_sankey(base_size = 12) +
  theme(legend.position = "none",
        axis.title.x = element_blank(),
        axis.text.x = element_text(size=12, color="black"),
        plot.margin = margin(5, 5, 5, 5)
  )

ggsave(
  filename = "./tables_and_figs_updated/fig4b_update.pdf",
  plot = p,
  width = 12,
  height = 5,
  units = "in",
  device = cairo_pdf
)









library(ggsankey)
library(ggplot2)
library(dplyr)

setwd("/Users/zhiyuanzhang/Documents/GitHub/vhPPIpredict-viral-phenotypes/")

# 读取数据
dt_v451_rna <- read.csv("./mid_result_updated/low_confidience_predict/low_pred_res.txt", sep = ",", header = TRUE)
dt_v451_rna$Virulence[dt_v451_rna$Virulence == "nonsevere"] <- "low"
# 只保留 Genome.Type 字符串中包含 RNA 的病毒
dt_v451_rna <- dt_v451_rna %>%
  filter(grepl("RNA", Genome.Type, ignore.case = TRUE))

write.csv(
  dt_v451_rna,
  "./mid_result_updated/low_confidience_predict/low_pred_res_RNA.csv",
  row.names = FALSE
)

cat("Number of RNA viruses:", nrow(dt_v451_rna), "\n")
print(table(dt_v451_rna$Genome.Type))

# 转换为 Sankey 所需的 long 格式
df_v451 <- dt_v451_rna %>%
  make_long(Genome.Type, Transmission.Route, Tissue.Tropism, Virulence, Transmissibility)

# 绘图
p <- ggplot(
  df_v451,
  aes(
    x = x,
    next_x = next_x,
    node = node,
    next_node = next_node,
    fill = factor(node),
    label = node
  )
) +
  geom_sankey(flow.alpha = 0.5, node.color = 1) +
  geom_sankey_label(size = 3.2, color = 1, fill = "white") +
  scale_fill_viridis_d(alpha = 0.95) +
  theme_sankey(base_size = 12) +
  theme(
    legend.position = "none",
    axis.title.x = element_blank(),
    axis.text.x = element_text(size = 12, color = "black"),
    plot.margin = margin(5, 5, 5, 5)
  )

# 保存PDF
ggsave(
  filename = "./tables_and_figs_updated/fig4b_update.pdf",
  plot = p,
  #width = 12,
  #height = 5,
  width = 8,
  height = 5,
  units = "in",
  device = cairo_pdf
)