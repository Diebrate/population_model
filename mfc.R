library(torch)
library(ktools)
library(dplyr)
library(ggplot2)

setwd("~/GitHub/population_model")

data <- readRDS("data\\data.rds")

data <- data %>% sample_n(10000)

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

nt_frac <- 0.2
nt_batch <- floor(nt_frac * nt)
if(nt_batch == 0){
  nt_batch <- 1
}
batch_num <- nt %/% nt_batch

d_in <- 3
d_out <- 2
d_hid <- 100

model <- vector("list", length=nt_batch)
optimizer <- vector("list", length=nt_batch)
scheduler <- vector("list", length=nt_batch)
cost <- vector("list", length=nt_batch)

for(b in 1:batch_num){
  model[[b]] <- nn_sequential(
    nn_linear(d_in, d_hid),
    nn_tanh(),
    nn_linear(d_hid, d_hid),
    nn_tanh(),
    nn_linear(d_hid, d_hid),
    nn_tanh(),
    nn_linear(d_hid, d_out)
  )
  optimizer[[b]] <- optim_adam(model[[b]]$parameters, lr = 0.01)
  scheduler[[b]] <- lr_step(optimizer[[b]], 5, gamma=0.2, last_epoch=-1)
  
}

n_iter <- 256
n_sample <- 1000
noise1 <- 1
noise2 <- 1
m <- distr_multivariate_normal(torch_zeros(2), dt * torch_tensor(diag(c(noise1, noise2))))

r_v <- 0.01
r_ent <- 1
r_kl_mid <- 10
r_kl_f <- 10

s1 <- 0.5
s2 <- 0.5
mkernel <- distr_multivariate_normal(torch_zeros(2), torch_tensor(diag(c(s1, s2))))

for(b in 1:batch_num){
  
  t0 <- (b - 1) * nt_batch
  if(b == batch_num){
    tf <- nt - 1
  }else{
    tf <- t0 + nt_batch - 1
  }
  
  x0 <- data %>% filter(time == t0) %>% select(c(UMAP_1, UMAP_2))
  xf <- data %>% filter(time == tf) %>% select(c(UMAP_1, UMAP_2)) %>% as.matrix() %>% torch_tensor()
  
  x_tensor <- list()
  for(t in t0:tf){
    x_temp <- data %>% filter(time == t) %>% select(c(UMAP_1, UMAP_2)) %>% as.matrix() %>% torch_tensor()
    x_tensor <- append(x_tensor, list(x_temp))
  }
  
  obj <- numeric(0)
  for(i in 1:n_iter){
    c <- torch_zeros(n_sample)
    x <- x0 %>% sample_n(n_sample, replace=TRUE) %>% as.matrix() %>% torch_tensor()
    # x <- x + mkernel$sample(n_sample)
    for(t in t0:tf + 1){
      inp <- torch_hstack(list(x, torch_ones(n_sample, 1) * t / nt))
      v <- model[[b]](inp)
      phat <- kernelp(x, s1=s1, s2=s2)
      pt <- kernelp_pred(x_tensor[[t - t0]], x, s1=s1, s2=s2)
      c <- c + dt * 0.5 * v$pow(2)$sum(dim=2) * r_v
      c <- c + dt * r_ent * phat$log()
      c <- c + r_kl_mid * (phat$log() - pt$log())
      x <- x + v * dt + m$sample(n_sample)
    }
    pf_hat <- kernelp(x, s1=s1, s2=s2)
    pf <- kernelp_pred(xf, x, s1=s1, s2=s2)
    c <- c + r_kl_f * (pf_hat$log() - pf$log())
    c <- c$mean()
    if(as.logical(c$isnan())){
      print(paste0('encountered nan at iteration ', as.character(i), ' batch ', as.character(b)))
    }else{
      print(paste0('c = ', as.character(as.double(c)), ' at iteration ', as.character(i), ' batch ', as.character(b)))
    }
    optimizer[[b]]$zero_grad()
    c$backward()
    scheduler[[b]]$step()
    optimizer[[b]]$step()
    obj <- c(obj, as.double(c))
  }
  
  cost[[b]] <- obj
  
}

x_init <- data %>% filter(time == 0) %>% select(c(UMAP_1, UMAP_2))
n_sim <- 100
x_sim <- x_init %>% sample_n(n_sim, replace=TRUE) %>% as.matrix() %>% torch_tensor()
data_sim <- torch_hstack(list(x_sim, torch_zeros(n_sim, 1)))
for(t in 1:(nt - 1)){
  inp <- torch_hstack(list(x_sim, torch_ones(n_sim, 1) * (t - 1) / nt))
  v <- model[[min(ceiling(t / nt_batch), batch_num)]](inp)
  x_sim <- x_sim + v * dt+ m$sample(n_sim)
  data_sim_temp <- torch_hstack(list(x_sim, torch_ones(n_sim, 1) * t / nt))
  data_sim <- torch_vstack(list(data_sim, data_sim_temp))
}
data_sim <- data.frame(as.matrix(data_sim))
names(data_sim) <- c("UMAP_1", 'UMAP_2', 'time')

print(data %>% ggplot(aes(x=UMAP_1, y=UMAP_2, colour=time)) + geom_point(size=1))
print(data_sim %>% ggplot(aes(x=UMAP_1, y=UMAP_2, colour=time)) + geom_point(size=1))