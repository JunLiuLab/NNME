## simulate data from max of two Gaussian processes
library(RandomFields)
library(geoR)
library(doParallel)
library(foreach)
registerDoParallel(cores=6)

tau = 1
beta = 16
beta2 = 4
sigma = 0.2
model <- RMgauss(var=tau^2, scale=sqrt(1/beta)) #+ #scale : d/scale
#RMnugget(var=sigma^2)

model2 <- RMgauss(var=tau^2, scale=sqrt(1/beta2))

from <- -1
to <- 1
x.seq <- seq(from, to, length=1000) 
y.seq <- seq(from, to, length=1000)

for(i in 1:10)
{
  simu <- RFsimulate(model, x=x.seq, y=y.seq, n=1,spConform=FALSE)
  simu2 <- RFsimulate(model2, x=x.seq, y=y.seq, n=1,spConform=FALSE)
  #plot(simu)
  ##write.csv(simu, "simulation_GP_1_100_0.2.csv")
  
  #dat = simu #simu[,,1] #RFspDataFrame2dataArray(simu)
  #plot(simu[1,], type="l")
  
  # get optimal value
  scale = sqrt((1 + 4*beta * 0.1^2 )/beta)
  sig2 = tau^2/(1 + 4*beta * 0.1^2)
  nugget = tau^2 - sig2 + sigma^2
  
  # sample data
  # # uniform
  N = 10000
  # x = sample(1:1000, N, replace = T)
  # y = sample(1:1000, N, replace = T)
  
  # mixtures of normal
  #c1
  x1 = round(rnorm(N*0.7, 0.3, 0.1)*1000)
  y1 = round(rnorm(N*0.7, 0.4, 0.15)*1000)
  
  #c2
  x2 = round(rnorm(N*0.3, 0.6, 0.15)*1000)
  y2 = round(rnorm(N*0.3, 0.7, 0.1)*1000)
  
  x = c(x1, x2)
  y = c(y1, y2)
  
  xy = unique(cbind(x, y))
  xy = xy[rowMins(xy)>=1 & rowMaxs(xy)<=1000,]
  n = nrow(xy)
  xy = xy[sample(n,n),]
  d = sapply(1:nrow(xy), function(i) max(simu[xy[i,1], xy[i,2]], simu2[xy[i,1], xy[i,2]])) 

  coords = cbind(x.seq[xy[,1]],y.seq[xy[,2]] )
  
  
  ## add measurement error
  for(sigma_u in c(0.02, 0.05, 0.1, 0.2))
  {
    w = coords+ matrix(rnorm(2*n) *sigma_u, nrow=n)
    write.table(cbind(coords,w, d + rnorm(n) * sigma, d), paste0("simulation_GP/testdata_mxG_prior_max/simulation_GP_train_1_",beta,"_0.2_", sigma_u, "_", i, ".txt"), col.names = F, row.names = F, sep="\t",quote=F)
  }
  
  # prediction
  ind = seq(5,1000, by= 10)
  locs = expand.grid(x.seq[ind], y.seq[ind])
  locs = as.matrix(locs)
  true_value1 = simu[ind, ind]
  true_value1 = reshape2::melt(true_value1)
  true_value2 = simu2[ind, ind]
  true_value2 = reshape2::melt(true_value2)
  
  true_value = true_value1
  true_value[,3] = sapply(1:nrow(true_value1), function(i) max(true_value1[i,3],true_value2[i,3]))
  true_value[,1:2] = locs
  
  write.table(true_value, paste0("simulation_GP/testdata_mxG_prior_max/simulation_GP_test_1_",beta,"_0.2_0.1_", i, ".txt"), col.names = F, row.names = F, sep="\t",quote=F)
}

# plot
library(ggplot2)
library(reshape2)
ggplot() + geom_tile(data=true_value, mapping=aes(x=Var1, y=Var2, fill=value)) + scale_fill_gradient2()



# plot prediction
predictions <- read.csv("simulation_GP/testdata_mxG_prior_max/simulation_GP_Kridging_beta16_mxG_prediction.csv")
true_value = read.table(paste0("simulation_GP/testdata_mxG_prior_max/simulation_GP_test_1_16_0.2_0.1_1.txt"))
train_dat = read.table(paste0("simulation_GP/testdata_mxG_prior_max/simulation_GP_train_1_16_0.2_0.1_1.txt"))
# idx = which(true_value[,1] > 0.2 & true_value[,2] < -0.2)
# idx = c(idx, which(true_value[,1] < -0.5 & true_value[,2] >0.5))
# true_value = true_value[-idx, ]
predictions <- cbind(true_value, predictions)
colnames(predictions)[1:3] = c("long", "lat","f")


for(fd in c("priorx_param_K4", "learn_priorx"))
{
  
  for(n in c(125, 250, 500, 1000, 2000))
  {
    for(sigma_u in c(0.02, 0.05, 0.1, 0.2))
    {
      
      nnpredict <- read.table(paste0("simulation_GP/testdata_mxG_prior_max/new_prediction", 
                                     "/simulation_GP_",fd,"_1_50_",n,"_9_32_5_32_1_16_0.2_",sigma_u,"_predict.txt")) 
      predictions <- cbind(predictions, nnpredict[,1])
    }
  }
}
colnames(predictions)[-1:-27] = paste("nnme", rep(c("GM_K4", "NICE"), each=20), rep(c(500, 1000, 2000, 4000, 8000), each=4), c(0.02, 0.05, 0.1, 0.2), sep="_")
saveRDS(predictions, file="simulation_GP/testdata_mxG_prior_max/predictions_all_9_5.rds")

vma = max(predictions$f) + 0.2
vmi = min(predictions$f) - 0.2

library(ggplot2)
library(cowplot)
p1 = ggplot() + geom_tile(data=predictions, mapping=aes(x=long, y=lat, fill=nnme_NICE_2000_0.1)) + scale_fill_gradient2(limits = c(vmi, vma)) +  geom_density_2d(data=train_dat, aes(x = V1, y = V2), color="black", alpha=0.7, linetype=2) + guides(fill=FALSE) + xlab("") + ylab("")  + theme_classic()#+ ggtitle("nme_500_0.1")
p2 = ggplot() + geom_tile(data=predictions, mapping=aes(x=long, y=lat, fill=nnme_NICE_8000_0.1)) + scale_fill_gradient2(limits = c(vmi, vma)) + geom_density_2d(data=train_dat, aes(x = V1, y = V2), color="black", alpha=0.7, linetype=2) + guides(fill=FALSE) + xlab("") + ylab("") + theme_classic()

p3 = ggplot() + geom_tile(data=predictions, mapping=aes(x=long, y=lat, fill=nnme_NICE_2000_0.05)) + scale_fill_gradient2(limits = c(vmi, vma)) +  geom_density_2d(data=train_dat, aes(x = V1, y = V2), color="black", alpha=0.7, linetype=2) + guides(fill=FALSE) + xlab("") + ylab("")  + theme_classic()
p4 = ggplot() + geom_tile(data=predictions, mapping=aes(x=long, y=lat, fill=nnme_NICE_8000_0.05)) + scale_fill_gradient2(limits = c(vmi, vma)) +  geom_density_2d(data=train_dat, aes(x = V1, y = V2), color="black", alpha=0.7, linetype=2) + guides(fill=FALSE) + xlab("") + ylab("")  + theme_classic()

pdf("simulation_GP/testdata_mxG_prior_max/GP_max_true2-legend.pdf", width = 6) #6.2
ggplot() + geom_tile(data=predictions, mapping=aes(x=long, y=lat, fill=f)) + scale_fill_gradient2(limits = c(vmi, vma))  +  geom_density_2d(data=train_dat, aes(x = V1, y = V2), color="black", alpha=0.7, linetype=2)  + xlab("") + ylab("") + theme_classic() +theme(legend.position = "bottom",legend.title = element_blank())
dev.off()

pdf("simulation_GP/testdata_mxG_prior_max/GP_max_NICE_prediction_0.1-2.pdf", width = 13)
theme_set(theme_cowplot())
plot_grid(p1,p2,nrow=1)
dev.off()

pdf("simulation_GP/testdata_mxG_prior_max/GP_max_MICE_prediction_0.05-2.pdf", width = 13)
theme_set(theme_cowplot())
plot_grid(p3,p4,nrow=1)
dev.off()


p1 = ggplot() + geom_tile(data=predictions, mapping=aes(x=long, y=lat, fill=kale_500_0.1)) + scale_fill_gradient2(limits = c(vmi, vma)) +  geom_density_2d(data=train_dat, aes(x = V1, y = V2), color="black", alpha=0.7, linetype=2) + guides(fill=FALSE) + xlab("") + ylab("") + theme_classic()
p2 = ggplot() + geom_tile(data=predictions, mapping=aes(x=long, y=lat, fill=kale_2000_0.1)) + scale_fill_gradient2(limits = c(vmi, vma)) +  geom_density_2d(data=train_dat, aes(x = V1, y = V2), color="black", alpha=0.7, linetype=2) + guides(fill=FALSE) + xlab("") + ylab("") + theme_classic()
p3 = ggplot() + geom_tile(data=predictions, mapping=aes(x=long, y=lat, fill=kale_500_0.05)) + scale_fill_gradient2(limits = c(vmi, vma))+  geom_density_2d(data=train_dat, aes(x = V1, y = V2), color="black", alpha=0.7, linetype=2)  + guides(fill=FALSE) + xlab("") + ylab("") + theme_classic()
p4 = ggplot() + geom_tile(data=predictions, mapping=aes(x=long, y=lat, fill=kale_2000_0.05)) + scale_fill_gradient2(limits = c(vmi, vma)) +  geom_density_2d(data=train_dat, aes(x = V1, y = V2), color="black", alpha=0.7, linetype=2) + guides(fill=FALSE) + xlab("") + ylab("") + theme_classic()
pdf("simulation_GP/testdata_mxG_prior_max/kridging_prediction_0.1-2.pdf", width = 13)
plot_grid(p1,p2, nrow=1)
dev.off()
pdf("simulation_GP/testdata_mxG_prior_max/kridging_prediction_0.05-2.pdf", width = 13)
plot_grid(p3,p4, nrow=1)
dev.off()
