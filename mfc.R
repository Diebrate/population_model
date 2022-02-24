library(torch)
library(ktools)
library(dplyr)
library(ggplot2)

setwd("~/GitHub/population_model")
meta_data <- readRDS("data\\sc_1_COPILOT.rds")

data <- data.frame(meta_data@reductions[["umap"]]@cell.embeddings)
data$time <- meta_data@meta.data[["consensus.time.group.50"]]
# data$time <- meta_data@meta.data[["consensus.time.group"]]

remove(meta_data)

data <- data %>% mutate(time = as.double(substr(time, 2, nchar(time))))
data <- data %>% arrange(time)

# vlength <- function(v){
#   l <- (v[, 1]$reshape(list(-1, 1)) - v[, 1])$pow(2)
#   l <- l + (v[, 2]$reshape(list(-1, 1)) - v[, 2])$pow(2)
#   return(l$sum(dim=2))
# }

kernelp <- function(x, s1=1, s2=1){
  k <- (x[, 1]$reshape(list(-1, 1)) - x[, 1])$pow(2) / s1
  k <- k + (x[, 2]$reshape(list(-1, 1)) - x[, 2])$pow(2) / s2
  k <- torch_exp(-0.5 * k) / (2 * pi * sqrt(s1 * s2))
  return(k$sum(dim=1) / x$shape[1])
}

kernelp_pred <- function(x, y, s1=1, s2=1){
  k <- (x[, 1]$reshape(list(-1, 1)) - y[, 1])$pow(2) / s1
  k <- k + (x[, 2]$reshape(list(-1, 1)) - y[, 2])$pow(2) / s2
  k <- torch_exp(-0.5 * k) / (2 * pi * sqrt(s1 * s2))
  return(k$sum(dim=1) / x$shape[1])
}

nt <- data %>% group_by(time) %>% count() %>% nrow()
dt <- 1 / nt

d_in <- 3
d_out <- 2
d_hid <- 128 

model <- nn_sequential(
  nn_linear(d_in, d_hid),
  nn_relu(),
  nn_linear(d_hid, d_hid),
  nn_tanh(),
  nn_linear(d_hid, d_hid),
  nn_relu(),
  nn_linear(d_hid, d_hid),
  nn_tanh(),
  nn_linear(d_hid, d_out)
)

optimizer <- optim_adam(model$parameters, lr = 0.02)
scheduler <- lr_step(optimizer, 5, gamma=0.2, last_epoch=-1)

x0 <- data %>% filter(time == 0) %>% select(c(UMAP_1, UMAP_2))
xf <- data %>% filter(time == (nt - 1)) %>% select(c(UMAP_1, UMAP_2)) %>% as.matrix() %>% torch_tensor()

x_tensor <- list()
for(t in 1:nt){
  x_temp <- data %>% filter(time == (t - 1)) %>% select(c(UMAP_1, UMAP_2)) %>% as.matrix() %>% torch_tensor()
  x_tensor <- append(x_tensor, list(x_temp))
}

n_iter <- 100
n_sample <- 100
noise1 <- 0.5
noise2 <- 0.5
m <- distr_multivariate_normal(torch_zeros(2), dt * torch_tensor(diag(c(noise1, noise2))))

r_v <- 0.005
r_ent <- 0.05
r_kl_mid <- 5 + 10 * (1:nt) / nt
r_kl_f <- 15

s1 <- 1
s2 <- 1
mkernel <- distr_multivariate_normal(torch_zeros(2), torch_tensor(diag(c(s1, s2))))

obj <- numeric(0)
for(i in 1:n_iter){
  c <- torch_zeros(n_sample)
  x <- x0 %>% sample_n(n_sample, replace=FALSE) %>% as.matrix() %>% torch_tensor()
  # x <- x + mkernel$sample(n_sample)
  for(t in 1:(nt - 1)){
    inp <- torch_hstack(list(x, torch_ones(n_sample, 1) * (t - 1) / nt))
    v <- model(inp)
    phat <- kernelp(x, s1=s1, s2=s2)
    pt <- kernelp_pred(x_tensor[[t]], x, s1=s1, s2=s2)
    c <- c + dt * 0.5 * v$pow(2)$sum(dim=2) * r_v
    c <- c + dt * r_ent * phat$pow(2)
    c <- c + r_kl_mid[t] * (phat - pt)$pow(2)
    x <- x + v * dt + m$sample(n_sample)
  }
  pf_hat <- kernelp(x, s1=s1, s2=s2)
  pf <- kernelp_pred(xf, x, s1=s1, s2=s2)
  c <- c + r_kl_f * (pf_hat - pf)$pow(2)
  c <- c$mean()
  if(as.logical(c$isnan())){
    print(paste0('encountered nan at iteration ', as.character(i)))
  }else{
    print(paste0('c = ', as.character(as.double(c)), ' at iteration ', as.character(i)))
  }
  optimizer$zero_grad()
  c$backward()
  scheduler$step()
  optimizer$step()
  obj <- c(obj, as.double(c))
}

n_sim <- 100
x_sim <- x0 %>% sample_n(n_sim, replace=TRUE) %>% as.matrix() %>% torch_tensor()
data_sim <- torch_hstack(list(x_sim, torch_zeros(n_sim, 1)))
for(t in 1:(nt - 1)){
  inp <- torch_hstack(list(x_sim, torch_ones(n_sim, 1) * (t - 1) / nt))
  v <- model(inp)
  x_sim <- x_sim + v * dt+ m$sample(n_sim)
  data_sim_temp <- torch_hstack(list(x_sim, torch_ones(n_sim, 1) * t / nt))
  data_sim <- torch_vstack(list(data_sim, data_sim_temp))
}
data_sim <- data.frame(as.matrix(data_sim))
names(data_sim) <- c("UMAP_1", 'UMAP_2', 'time')

print(data %>% ggplot(aes(x=UMAP_1, y=UMAP_2, colour=time)) + geom_point(size=1))
print(data_sim %>% ggplot(aes(x=UMAP_1, y=UMAP_2, colour=time)) + geom_point(size=1))