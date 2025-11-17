library(ggsankey)
library(ggplot2)
library(dplyr)

setwd("/Users/zhiyuanzhang/Documents/GitHub/vhPPIpredict-viral-phenotypes/")
dt_v451_rna = read.csv("./data/low_evidence_virus/v403_rna_for_sankey_new.csv", sep=",", header=TRUE)


df_v451 <- dt_v451_rna %>%
  #make_long(infect_human_label, transmission_route_label, tissue_tropism_pred_label, virulence_label, infectivity_label) 
  make_long(Genome.Type, Transmission.Route, Tissue.Tropism, Virulence, Transmissibility) 


ggplot(df_v451, aes(x = x, 
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
