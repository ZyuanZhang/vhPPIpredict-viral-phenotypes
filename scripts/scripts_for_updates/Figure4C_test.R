library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)

setwd("/Users/zhiyuanzhang/Documents/GitHub/vhPPIpredict-viral-phenotypes/scripts/scripts_for_updates/")
# ========= 1. 读入数据 =========
dt_value <- read_csv("../../data/low_evidence_virus/403_rna_phenotype_value.csv")

# 去掉括号内容
dt_value$`virus name` <- sapply(strsplit(dt_value$`virus name`, "\\("), `[`, 1)
dt_value$`virus name` <- trimws(dt_value$`virus name`)

# 计算 risk_score
dt_value$risk_score <- dt_value$virulence +
  dt_value$`infecting-human` +
  dt_value$`sustained human-to-human`

# 按 risk_score 排序，取前10
dt_sorted <- dt_value %>% arrange(desc(risk_score))
top10 <- head(dt_sorted$`virus name`, 10)

# 只保留需要的列
heatmap_df <- dt_sorted %>%
  filter(`virus name` %in% top10) %>%
  select(`virus name`, risk_score,
         `infecting-human`, virulence,
         `sustained human-to-human`, `limited human-to-human`, `no human-to-human`,
         gastrointestinal, neural, `respiratory (tissue tropism)`, systemic, viraemic, other,
         `direct contact`, `faecal-oral`, `respiratory (transmission route)`, vector)

# 保持 top10 顺序
heatmap_df$`virus name` <- factor(heatmap_df$`virus name`, levels = rev(top10))

# ========= 2. 转成长格式 =========
plot_long <- heatmap_df %>%
  pivot_longer(
    cols = -`virus name`,
    names_to = "indicator",
    values_to = "value"
  )

# 设定指标顺序（按你的分组）
indicator_order <- c(
  "risk_score",
  "infecting-human", "virulence",
  "sustained human-to-human", "limited human-to-human", "no human-to-human",
  "gastrointestinal", "neural", "respiratory (tissue tropism)", "systemic", "viraemic", "other",
  "direct contact", "faecal-oral", "respiratory (transmission route)", "vector"
)
plot_long$indicator <- factor(plot_long$indicator, levels = indicator_order)

# ========= 3. 分组信息（用于顶部色条和分面） =========
group_info <- data.frame(
  indicator = indicator_order,
  group = c(
    "Risk",
    rep("Transmissibility", 5),
    rep("Tissue Tropism", 6),
    rep("Transmission Route", 4)
  )
)
group_info$indicator <- factor(group_info$indicator, levels = indicator_order)

# 把 group 合并到 plot_long
plot_long <- plot_long %>%
  left_join(group_info, by = "indicator")

# ========= 4. 画气泡矩阵图 =========
ggplot(plot_long, aes(x = value, y = `virus name`, fill = group)) +
  geom_col(width = 0.7) +
  facet_wrap(~indicator, ncol = 5, scales = "free_x") +
  scale_fill_manual(values = c("Risk"="black",
                               "Transmissibility"="orange",
                               "Tissue Tropism"="cyan",
                               "Transmission Route"="purple")) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 7))