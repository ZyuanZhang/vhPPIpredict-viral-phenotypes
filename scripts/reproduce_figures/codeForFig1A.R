## ========== 加载包 ==========
library(dplyr)
library(ggplot2)
library(geomtextpath)
library(stringr)

setwd("/Users/zhiyuanzhang/Documents/GitHub/vhPPIpredict-viral-phenotypes")

## ========== 读入数据 ==========
dt <- read.csv("../data/virus_human_physical_ppi.txt", sep="\t", header=TRUE)

## ========== 按病毒家族统计 PPI 数目 ==========
family_counts <- dt %>%
  group_by(FamilyName) %>%
  summarise(PPI_Count = n()) %>%
  arrange(desc(PPI_Count))

## ========== 取前5个，其余合并为“Other families” ==========
top5 <- family_counts[1:5, ]
other_sum <- sum(family_counts$PPI_Count[6:nrow(family_counts)])
family_counts_top <- bind_rows(top5, data.frame(FamilyName = "Other families", PPI_Count = other_sum))

## 计算百分比
family_counts_top <- family_counts_top %>%
  mutate(Percent = PPI_Count / sum(PPI_Count) * 100)

## 设置颜色序列（从高到低对应）
colors_hex <- c("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#A9A9A9")

## 设置顺序
family_counts_top$FamilyName <- factor(family_counts_top$FamilyName, levels = family_counts_top$FamilyName)

## 标签：PPI数 + 百分比
family_counts_top <- family_counts_top %>%
  mutate(label = paste0(PPI_Count, "\n(", sprintf("%.1f", Percent), "%)"))

## ========== 绘图 ==========
ggplot(family_counts_top) +
  ## ===== 仅添加最外圈的圆环线 =====
geom_hline(
  aes(yintercept = y),
  data = data.frame(y = max(family_counts_top$PPI_Count) * 1.28),
  color = "gray60",
  linewidth = 0.8
) +
  
  ## ===== 添加径向虚线 =====
geom_segment(
  aes(
    x = FamilyName,
    y = 0,
    xend = FamilyName,
    yend = max(family_counts_top$PPI_Count) * 1.2
  ),
  linetype = "dashed",
  color = "gray70",
  linewidth = 0.5
) +
  
  ## 柱子（环形）
  geom_col(
    aes(x = FamilyName, y = PPI_Count, fill = FamilyName),
    width = 1,
    color = "black",
    show.legend = FALSE
  ) +
  
  ## 使用自定义颜色
  scale_fill_manual(values = colors_hex) +
  
  ## 转换为环形
  coord_polar() +
  
  ## 柱顶标注 PPI 数 + 百分比
  geom_textpath(
    aes(x = FamilyName, y = PPI_Count + max(PPI_Count) * 0.145, label = label),
    size = 4.5,
    colour = "black",
    upright = TRUE
  ) +
  
  ## 外圈标注家族名称
  geom_textpath(
    aes(x = FamilyName, y = max(PPI_Count) * 1.4, label = FamilyName),
    size = 5,
    colour = "black",
    upright = TRUE
  ) +
  
  ## 控制环厚度
  scale_y_continuous(
    limits = c(-max(family_counts_top$PPI_Count)*0.15, max(family_counts_top$PPI_Count)*1.4),
    expand = c(0, 0)
  ) +
  theme_minimal()+
  ## 样式设置
  theme(
    panel.border = element_blank(),
    panel.grid = element_blank(),
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    plot.background = element_rect(fill = "white", color = NA)
  )
