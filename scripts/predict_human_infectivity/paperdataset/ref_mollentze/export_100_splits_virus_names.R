library(dplyr)

# ---------------------------
# 1. Read fixed taxid splits
# ---------------------------
fixed_taxid <- read.csv(
  "/home/zhangzhiyuan/Desktop/vhppipred_cases/PPIpredictHumanVirus/zzyscript/paperdataset/data/fixed_100_splits_taxid.csv",
  stringsAsFactors = FALSE
)

# 强制列名标准化（防大小写）
colnames(fixed_taxid) <- tolower(colnames(fixed_taxid))

stopifnot(all(c("iteration", "taxid", "set") %in% colnames(fixed_taxid)))

fixed_taxid$taxid <- as.character(fixed_taxid$taxid)

# ---------------------------
# 2. Read taxid → species map
# ---------------------------
taxid_map <- read.csv(
  "/home/zhangzhiyuan/Desktop/vhppipred_cases/PPIpredictHumanVirus/zzyscript/paperdataset/data/linshi.csv",
  stringsAsFactors = FALSE
)

colnames(taxid_map) <- c("LatestSppName", "taxid")
taxid_map$taxid <- as.character(taxid_map$taxid)

# ---------------------------
# 3. Ensure one-to-one mapping
# ---------------------------
taxid_map_unique <- taxid_map %>%
  distinct(taxid, LatestSppName) %>%
  group_by(taxid) %>%
  slice(1) %>%
  ungroup()

# ---------------------------
# 4. Join (explicit many-to-many)
# ---------------------------
fixed_long <- fixed_taxid %>%
  left_join(
    taxid_map_unique,
    by = "taxid",
    relationship = "many-to-many"
  )

# ---------------------------
# 5. Sanity check
# ---------------------------
if (any(is.na(fixed_long$LatestSppName))) {
  stop("Some taxids could not be mapped to LatestSppName")
}

# ---------------------------
# 6. Sort safely
# ---------------------------
fixed_long <- fixed_long %>%
  arrange(.data$iteration, .data$set, .data$LatestSppName)

# ---------------------------
# 7. Write output
# ---------------------------
out_file <- "/home/zhangzhiyuan/Desktop/vhppipred_cases/PPIpredictHumanVirus/zzyscript/paperdataset/data/fixed_100_splits_iteration_taxid_species.csv"

write.csv(fixed_long, out_file, row.names = FALSE, quote = FALSE)

cat("✅ Done. File written to:\n", out_file, "\n")

