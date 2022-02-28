library(Seurat)
library(dplyr)
library(ggplot2)

setwd("~/GitHub/population_model")

data_path <- list(
  # c("data\\Columella_Atlas.rds", "Col")
  c("data\\sc_1_COPILOT.rds", "sc_1"),
  c("data\\sc_10_at_COPILOT.rds", "sc_10"),
  c("data\\pp1_COPILOT.rds", "pp1")
)

data <- data.frame(matrix(nrow=0, ncol=4))
names(data) <- c("UMAP_1", 'UMAP_2', 'time', 'source')

for(f in data_path){
  meta_data <- readRDS(f[1])
  data_temp <- data.frame(meta_data@reductions[["umap"]]@cell.embeddings)
  data_temp$time <- meta_data@meta.data[["consensus.time.group.50"]]
  data_temp$source <- f[[2]]
  data <- rbind(data, data_temp)
  remove(meta_data)
}

data <- data %>% mutate(time = as.double(substr(time, 2, nchar(time))))
data <- data %>% arrange(time)

print(data %>% ggplot(aes(x=UMAP_1, y=UMAP_2, colour=source)) + geom_point(size=1))

# meta_data <- readRDS("data\\Root_Atlas.rds")

# saveRDS(data, "data\\data.rds")