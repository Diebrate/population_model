library(torch)
library(ktools)
library(dplyr)
library(ggplot2)

setwd("~/GitHub/population_model")
meta_data <- readRDS("data\\sc_1_COPILOT.rds")

data <- data.frame(meta_data@reductions[["umap"]]@cell.embeddings)
data$time <- meta_data@meta.data[["consensus.time.group.50"]]

remove(meta_data)
