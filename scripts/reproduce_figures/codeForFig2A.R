library(UpSetR)

setwd("/Users/zhiyuanzhang/Documents/GitHub/vhPPIpredict-viral-phenotypes/")

data <- read.csv("./data/dt_for_fig2a_upset_plot.csv", sep="\t", header = TRUE, row.names = 1)

upset(data, nset=6, mb.ratio = c(0.7, 0.3), text.scale=1.2, mainbar.y.label="Count of virus", nintersects=NA)

## figsize = 6x5
