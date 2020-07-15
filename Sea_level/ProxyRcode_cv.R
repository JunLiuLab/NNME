#######Set a working directory and read in the data set#######
#setwd("../Sealevel_Supp/")
args = commandArgs(trailingOnly = T)
dd = args[1]
data0<-read.csv(paste0("Data_NC_m_shuffle_",dd,".csv"),header = F)
param <- read.csv(paste0("sea_level_cv_params-tanh-2gamma-prior-sel_",dd,".csv"),header = F)
#############################################################
########Required Packages and Libraries########
#install.packages("rjags")
library(rjags)
#install.packages("R2jags")
library(R2jags)
#install.packages("coda")
library(coda)
#install.packages("runjags")
library(runjags)
#install.packages("MASS")
library(MASS)
#install.packages("fields")
library(fields)
##############################################################

########Setting up a row in the dataset for GIA rates (mm/yr)########
yocc<-2.010  			##Year of core collection
gia<-data0[,5]     ##Rates for glacial isostatic adjustment
data0[,2] = ((yocc-data0[,1]/1000)*gia)+data0[,2]


########Model setup######## 
myiterations <- 3000 # 6000
myburnin <-500 # 1000
mythin <- 2 # 3

###The model file (saved in wd)###
myjagsfile<-"../IGP_Proxy.txt"

######## cross validation ########
fold_size = 34
mse = rep(0, 5)
wmse = rep(0, 5)
nsample=100

for( f  in 1:5)
{
  st = fold_size* (f-1) + 1
  ed = min(fold_size * f, nrow(data0))
  data = data0[-st : -ed, ]
  x_pred = data0[st : ed, 1]/1000
  sd_x = data0[st : ed, 3]/2000
  y_true = data0[st : ed, 2]
  ########Set up the data########
  N<-nrow(data)

  x1<-data[,1]  		
  x<-x1/1000        ##Age in thousands of years
  gia = data[,5]
  #y<-((yocc-x)*gia)+data[,2] 	##Adjust sea level for GIA
  y = data[,2] 
  #plot(x,y)
  
  varx1<-(((data[,3]/1000)/2)^2)		##Age error is a 2-sigma error	
  vary1<-data[,4]^2					##RSL error is a 1-sigma error
  
  ###############################################################

  ########Setting up the covariance and precision matrices########
  V22<-(((gia^2)*varx1)+vary1)
  V12<--gia*varx1
  V21<--gia*varx1
  V11<-varx1
  V<-array(NA,c(2,2,nrow(data)))
  P<-array(NA,c(2,2,nrow(data)))
  for(i in 1:N)
  {
    V[,,i]<- matrix(c(V11[i],V12[i],V21[i],V22[i]),2,2)
    P[,,i]<-solve(V[,,i])
  }

  ############# RE-ADJUST DATA TO START AT (0.01,0) ############
  ### this makes the integration limits straight-forward
  minx=min(x)-0.01
  x=x-minx
  x_pred = x_pred - minx
  ############################################################
  D<-cbind(x,y)

  ############# Set up the grid for the GP ###################
  nw=50      # Sets the number of grid points for the derivative      
  up=max(x)+.001
  low=.001
  xstar=seq(low,up,by=(up-low)/(nw-1)) 
  Nstar=length(xstar)

  ########Distance Matrix########
  Dist1<-rdist(xstar)
  ##############################################################
  
  ########Initialize quadrature for the integration########
  L=30            ## this sets the precision of the integration quadrature (higher is better but more computationally expensive)
  index=1:L        
  cosfunc=cos(((2*index-1)*pi)/(2*L))
  
  quad1=array(dim=c(nrow=N,ncol=Nstar,L))
  quad2=array(dim=c(nrow=N,ncol=Nstar,L))
  
  for(j in 1:Nstar)
  {   for(k in 1:N) 
  { quad1[k,j,]=abs((x[k]*cosfunc/2)+(x[k]/2)-xstar[j])^1.99
    quad2[k,j,]=((x[k]/2)*(pi/L))*(sqrt(1-cosfunc^2))
  }
  }
  
  ###############################################################

  ###The Data, Initial values and parameters to save###
  mydata <- list(n=N,m=Nstar,P=P,D=D,Dist1=Dist1,quad1=quad1,quad2=quad2,kappa=1.99)    
  myinitial<-function(){list("beta0"=rnorm(1,0,3),
                            "sigmasq.g"=runif(1,0,10),"p"=rbeta(1,4,1),"sigmasq.y"=runif(1,0,10))}
  mypars <- c("beta0","sigmasq.y","sigmasq.g","p","w.m","mu.y")
  
  ########Run the model########
  start.time <- Sys.time()
  run.mcmc <- jags(data=mydata, inits=myinitial, parameters.to.save=mypars, model.file=myjagsfile,n.chains=2, n.iter=myiterations, n.burnin=myburnin,n.thin=mythin,DIC=TRUE)
  end.time <- Sys.time()
  runtime<-end.time-start.time
  #####################################
  
  ################ prediction ##################### 
  ###Expected value of the integrated Gaussian Process
  m = length(x_pred)
  quad1=array(dim=c(nrow=m,ncol=Nstar,L))
  quad2=array(dim=c(nrow=m,ncol=Nstar,L))
  
  beta0 = median(run.mcmc$BUGSoutput$sims.list$beta0)
  p = median(run.mcmc$BUGSoutput$sims.list$p)
  K = matrix(1+0.00001, Nstar, Nstar)
  for(i in 1:(Nstar-1))
  {
    ######Exponential
    for(j in (i+1):Nstar)
    {
      K[i,j]<-p^(Dist1[i,j]^1.99)
      K[j,i]<-K[i,j]
      
    }
  }
  
  sigmasq.g = median(run.mcmc$BUGSoutput$sims.list$sigmasq.g)
  K.w.inv <- solve(K)
  K.inv <- K.w.inv/sigmasq.g
  
  y_pred = matrix(0, m, nsample)
  w = matrix(0, m, nsample)
  for(r in 1:nsample)
  {
    x_pred2 = x_pred + rnorm(m, 0, sd_x)
    for(j in 1:Nstar)
    {   
      for(k in 1:m) 
        { 
          quad1[k,j,]=abs((x_pred2[k]*cosfunc/2)+(x_pred2[k]/2)-xstar[j])^1.99
          quad2[k,j,]=((x_pred2[k]/2)*(pi/L))*(sqrt(1-cosfunc^2))
        }
    }
    K.gw = matrix(0, m, Nstar)
    for(i in 1:Nstar) { 
      for(j in 1:m) {
        K.gw[j,i]<-sum(p^quad1[j,i,]*quad2[j,i,])  #### Quadrature function 
        
      } #End j loop 
      
    } #End i loop	
    
    dydt = run.mcmc$BUGSoutput$sims.list$w.m
    w.tilde.m<-K.gw%*%K.w.inv%*%apply(dydt,2,quantile,prob=0.5,na.rm=T)
    y_pred[,r]<-beta0+w.tilde.m
    # compute weight
    x_pred2 = 2.01 - (x_pred2 + minx)
    x_pred2[x_pred2 < 1e-8] = 1e-8
    w[,r] = param[f,5]*dgamma(x_pred2, param[f,1], rate=param[f,2]) + param[f,6]*dgamma(x_pred2, param[f,3], rate=param[f,4])
   
  }
  y_pred= rowMeans(y_pred)
  wy_pred = rowSums(y_pred*w)/rowSums(w)
  mse[f] = mean((y_pred - y_true)^2)
  wmse[f] = mean((wy_pred - y_true)^2)
  
  #########Save the Results#########
  ##results.jags<-as.mcmc(run.mcmc)
  save(y_pred, wy_pred, y_true, x, x_pred, run.mcmc,file=paste0("NCResults_run2_",dd,".mcmc.", f,".RData"))
  ##save(results.jags,file="NCResults_resultsjags.RData")
  ##stop()
}
#mse = mean(mse)
#wmse = mean(wmse)
save(mse, wmse,file=paste0("NCResults_run2_",dd,".results.RData"))




