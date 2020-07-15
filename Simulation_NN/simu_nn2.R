library(stringr)
library(dplyr)
library(matrixStats)
library(ggplot2)
library(reshape2)
library(RandomFields)
library(geoR)
library(doParallel)
library(foreach)
registerDoParallel(cores=2)

sigma_u = 0.1
sigma = 0.3

RFoptions(modus_operandi="easygoing")
estmodel <- RMgauss(var=NA, scale=NA) + #scale : d/scale
  RMnugget(var=NA) + RMtrend(NA)

result = matrix(0, 5, 6)
colnames(result) <- c("KILE_ise", "KILE_iae", "KALE_ise", "KALE_iae", "spline_ise", "spline_iae") ##, "rel_ise", "rel_iae")
for( i in c(0))
{
  train_dat <- read.table(paste0("simu_nn2/nn_5_32_train_0.2_0.2_", i, ".txt"))
  test_dat <- read.table(paste0("simu_nn2/nn_5_32_test_", i, ".txt"))
  n = 500
  sigma_u = 0.2
  sigma = 0.2
  
  w = as.matrix(train_dat[1:n,1:2] + matrix(rnorm(2 * n, sd=sigma_u), ncol=2))
  #w = train_dat[1:n,3:4]
  d = train_dat[1:n,6] + rnorm(n, sd=sigma)
  
  locs = as.matrix(test_dat[,1:2])
  
  fit <- RFfit(estmodel, x = w[,1], y = w[,2], data=d)
  param = summary(fit)$param
  KILE<-krige.conv(data=d,coords=w,locations=locs, krige=krige.control(cov.model = "gaussian", cov.pars = c(param[1,1], param[1,2]), nugget = param[1,3]))
  print(mean((KILE$predict - test_dat[,3])^2))
  #result[i, 3] = result[i, 1]/var(train_dat[-1:-n,6])
  #result[i+1, 2] = mean(abs(KILE$predict - test_dat[,2]))
  #result[i, 4] = result[i, 2]/mean(abs(train_dat[-1:-n,6]))
  
  # KALE
  est_beta = 1/(param[1,2]^2 - 4*sigma_u^2)
  est_tau2 = param[1,1] * (1 + 4*sigma_u^2 * est_beta)
  est_nugget = param[1,3] - (est_tau2 - param[1,1]) 
  
  new_tau2 = est_tau2/(1 + 2*est_beta * sigma_u^2)
  new_scale2 = (1 + 2*est_beta * sigma_u^2 )/est_beta
  
  ds = w %*% t(w)
  dds = diag(ds) # w2
  ds = dds - 2*ds
  ds = t(ds) + dds
  Sigma = param[1,1] * exp(-ds/param[1,2]^2)
  
  # Sigma = matrix(0, n, n)
  # for(ii in 1:(n-1))
  # {
  #   for(jj in (ii+1) :n)
  #   {
  #     ds = sum((w[ii,] - w[jj,])^2)
  #     Sigma[ii,jj] = Sigma[jj,ii] = param[1,1] * exp(-ds/param[1,2]^2)
  #   }
  # }
  
  #Sigma = Sigma + diag(param[1,3] + param[1,1], nrow(Sigma))
  
  Sigma = Sigma + diag(param[1,3], nrow(Sigma))
  
  temp =solve(Sigma, d - param[1,4])
  
  tcoords = t(w)
  Sigma_new = foreach(i=1:nrow(locs), .combine = rbind) %dopar% {
    ds = colSums((tcoords - locs[i,])^2)
    new_tau2 * exp(-ds/new_scale2) #+ new_nuggle*(ds==0)
  }
  
  kale_pred = Sigma_new %*% temp+ param[1,4]
  
  print(mean((kale_pred - test_dat[,3])^2))
  
  write.csv(cbind(KILE$predict, kale_pred), file=paste("simu_nn2/nn_5_32",sigma_u, sigma, n, "kridging_prediction.txt", sep="_"))

}



plot(test_dat[,1], test_dat[,2], type="l", col=2)
lines(test_dat[,1], spline_pred)
lines(test_dat[,1], KILE$predict)

## read nn result
sx = 0.2
sy = 0.2
nn_all = matrix(0,5 ,0)
for(sx in c(0.1, 0.2))
{
  for(sy in c(0.2))
  {
    nnresult <- read.table(paste0("simu_nn2/nn_5_32_", sx, "_", sy, ".txt"))

    # select the best over three repeats by train_sy
    colnames(nnresult) <- c("N", 'repeat', "NN_train", "NN_ise", "train_sy",  "train_loss", "NME_ise", "NN_iae", "NME_iae")
    
    nn_all = cbind(nn_all, t(nnresult %>% group_by(N, `repeat`) %>% top_n(-1, NME_ise)  %>%group_by(N) %>% summarise(ise_mean = mean(NME_ise), ise_sd = sd(NME_ise, na.rm=T)/sqrt(50), iae_mean = mean(NN_ise), iae_sd = sd(NN_ise,na.rm=T)/sqrt(50))))
    
   # round(t(nnresult %>% group_by(N) %>% summarise(ise_mean = mean(NME_ise), ise_sd = sd(NME_ise, na.rm=T)/sqrt(50), iae_mean = mean(NME_iae), iae_sd = sd(NME_iae,na.rm=T)/sqrt(50))),4)
    
  #  round(t(nnresult %>% group_by(N, `repeat`) %>% top_n(-1, train_sy)  %>%group_by(N) %>% summarise(ise_mean = mean(NME_ise), ise_sd = sd(NME_ise, na.rm=T)/sqrt(50), iae_mean = mean(NME_iae), iae_sd = sd(NME_iae,na.rm=T)/sqrt(50))),4)
  }
}
round(nn_all[2:5,],4)
#sx 0.1, sy 0.2/0.4: nn1
# sx 0.2, sy 0.4: nn
#sx 0.2, sy 0.2: nn1

# read in kridging
kridge = matrix(0,4,0)
for(sx in c(0.1, 0.2))
{
  for(sy in c(0.2))
  {
    for(n in c(500, 1000, 2000))
    {
      temp <- read.csv(paste("simu_nn2/nn_5_32", sx, sy, n, "kridging.csv",sep="_"), row.names = 1)
      temp = as.matrix(temp)
      kridge <- cbind(kridge, c(colMeans(temp[,c(3,1)]),colSds(temp[,c(3,1)])/sqrt(50)))
      
    }
  }
}

round(kridge, 4)


# plot prediction
testdat <- read.table("simu_nn2/nn_5_32_test_0.txt")
temp1 <- read.table("simu_nn2/nn1_5_32_0_0.2_500_prediction.txt")
temp2 <- read.table("simu_nn2/nn1_5_32_0_0.2_1000_prediction.txt")
temp3 <- read.table("simu_nn2/nn1_5_32_0_0.1_2000_prediction.txt")
predictions <- data.frame("long"=testdat[,1], "lat" = testdat[,2], "nme_500_0.2"=temp1[,2], "nme_1000_0.2"=temp2[,2],"nme_2000_0.1"=temp3[,2])

vma = max(testdat$V3) + 0.2
vmi = min(testdat$V3) - 0.2

p2 = ggplot() + geom_tile(data=predictions, mapping=aes(x=long, y=lat, fill=nme_500_0.2)) + scale_fill_gradient2(limits = c(vmi, vma), midpoint = 6) + theme(legend.position = "none") + ggtitle("nnme_500_0.2")
p3 = ggplot() + geom_tile(data=predictions, mapping=aes(x=long, y=lat, fill=nme_1000_0.2)) + scale_fill_gradient2(limits = c(vmi, vma), midpoint = 6) + theme(legend.position = "none") + ggtitle("nnme_1000_0.2")
p4 = ggplot() + geom_tile(data=predictions, mapping=aes(x=long, y=lat, fill=nme_2000_0.1)) + scale_fill_gradient2(limits = c(vmi, vma), midpoint = 6) + theme(legend.position = "none")+ ggtitle("nnme_2000_0.1")

pdf("simu_nn2/nme_prediction.pdf", width = 21)
plot_grid(p2, p3, p4,nrow=1)
dev.off()


temp1 <- read.csv("simu_nn2/nn_5_32_0.2_0.2_500_kridging_prediction.txt")
temp2 <- read.csv("simu_nn2/nn_5_32_0.2_0.2_1000_kridging_prediction.txt")
temp3 <- read.csv("simu_nn2/nn_5_32_0.1_0.2_2000_kridging_prediction.txt")
predictions <- data.frame("long"=testdat[,1], "lat" = testdat[,2], "nme_500_0.2"=temp1[,3], "nme_1000_0.2"=temp2[,3],"nme_2000_0.1"=temp3[,3])

vma = max(testdat$V3) + 0.2
vmi = min(testdat$V3) - 0.2

p2 = ggplot() + geom_tile(data=predictions, mapping=aes(x=long, y=lat, fill=nme_500_0.2)) + scale_fill_gradient2(limits = c(vmi, vma), midpoint = 6) + theme(legend.position = "none") + ggtitle("kale_500_0.2")
p3 = ggplot() + geom_tile(data=predictions, mapping=aes(x=long, y=lat, fill=nme_1000_0.2)) + scale_fill_gradient2(limits = c(vmi, vma), midpoint = 6) + theme(legend.position = "none") + ggtitle("kale_1000_0.2")
p4 = ggplot() + geom_tile(data=predictions, mapping=aes(x=long, y=lat, fill=nme_2000_0.1)) + scale_fill_gradient2(limits = c(vmi, vma), midpoint = 6) + theme(legend.position = "none")+ ggtitle("kale_2000_0.1")

pdf("simu_nn2/kale_prediction.pdf", width = 21)
plot_grid(p2, p3, p4,nrow=1)
dev.off()

predictions <- data.frame("long"=testdat[,1], "lat" = testdat[,2], "truef"=testdat[,3])
pdf("simu_nn2/testdata0_truef.pdf")
ggplot() + geom_tile(data=predictions, mapping=aes(x=long, y=lat, fill=truef)) + scale_fill_gradient2(limits = c(vmi, vma), midpoint = 6) + theme(legend.position = "none") + ggtitle("true_f")
dev.off()

vars = rep(0,50)
for(i in 0:49){ 
  test_dat <- read.table(paste0("simu_nn2/nn_5_32_test_", i, ".txt"))
  vars[i+1] = var(test_dat[,3])
}