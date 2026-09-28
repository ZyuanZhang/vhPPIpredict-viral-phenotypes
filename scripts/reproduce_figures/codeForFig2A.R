library(UpSetR)

setwd("/Users/zhiyuanzhang/Documents/GitHub/vhPPIpredict-viral-phenotypes/")

data <- read.csv("./data/dt_for_fig2a_upset_plot.csv", sep="\t", header = TRUE, row.names = 1)


pdf("./tables_and_figs_updated/fig2a_update.pdf", width = 9, height = 6.5)
upset(data, nset=7, mb.ratio = c(0.6, 0.4), text.scale=2, 
      mainbar.y.label="Intersection of viruses", sets.x.label="Number of viruses", 
      nintersects=10, order.by = "freq", set_size.show=TRUE,
      set_size.scale_max = 1700)


dev.off()
## figsize = 6x5 
## 800*650



library(UpSetR)

setwd("/Users/zhiyuanzhang/Documents/GitHub/vhPPIpredict-viral-phenotypes/")

data <- read.csv(
  "./data/dt_for_fig2a_upset_plot.csv",
  sep = "\t",
  header = TRUE,
  row.names = 1
)

colnames(data)[colnames(data) == "Virus.HostDB"] <- "Virus-Host DB"

# 关闭所有已有绘图设备
graphics.off()

# 打开PDF
pdf(
  "./tables_and_figs_updated/fig2a_update.pdf",
  width = 9,
  height = 6.5,
  onefile = FALSE
)

# 直接绘制
upset(
  data,
  nset = 7,
  mb.ratio = c(0.6, 0.4),
  text.scale = 2,
  mainbar.y.label = "Intersection of viruses",
  sets.x.label = "Number of viruses",
  nintersects = 10,
  order.by = "freq",
  set_size.show = TRUE,
  set_size.scale_max = 1800
)

dev.off()