library(readr)
library(tidyverse)
library(ggplot2)
library(gridExtra)
source('mad/featureimp.R')

# --------------- FUNCTIONS -----------------
# FUNCTIONS 
fun1 <- function(x){
  return (x[1] + x[2] + x[1] * x[2])
}
fun2 <- function(x){
  return (x[1]**2 + x[2]**2)
}
# FUNCTIONS TO GENERATE THE ARTIFICIAL DATASETS
generate_normal <- function(mu,sigma,n){
  r1 <- NULL
  r2 <- NULL
  for(i in 1:length(mu)){
    r1 <- rnorm(n,mu[i],sigma[i])
    r2 <- cbind(r2,r1)
  }
  return(r2)
}
generate_mixnormal <- function(mu,sigma,n,sigmatimes=10,e=0.05){
  r1 <- NULL
  r2 <- NULL
  
  for(i in 1:length(mu)){
    components <- sample(1:2,prob=c(1-e,e),size=n,replace=TRUE)
    sds <- c(sigma[i],sigmatimes*sigma[i])
    mus <- c(mu[i],mu[i])
    r1 <- rnorm(n,mean=mus[components],sd=sds[components])
    r2 <- cbind(r2,r1)
  }
  return(r2)
}


# --------------- EXPERIMENTS OF ARTIFICIAL DATASETS -----------------
          # ----- PARAMETERS -----
errorrate=c(0.005,0.01,0.05,0.1)
mixsigma=rep(c(10:2),length(errorrate))*1
mixe=c(rep(errorrate,rep(length(mixsigma)/length(errorrate),length(errorrate))))
ttimes = 1.2
mu_list_x1=rep(1,length(mixe))
mu_list_x2=rep(1,length(mixe))
sigma_list_x1=rep(1,length(mixe))
sigma_list_x2=sigma_list_x1*ttimes

num_exp = 1 # the number of experiments
num_trials = 300  # the number of trials in each experiment
samplesize = 1000 # the sample size of each trial
interval_a = c(30) # the number of intervals used in the conditional calculation


          # ----- RUNNING -----
set.seed(0)
funlist = NULL #function
randlist = NULL #random distribution
interlist = NULL #intervals when get the cutpoints
inmlist = NULL #interval cutting method
para_X1_list_a = NULL 
para_X1_list_b = NULL
para_X2_list_a = NULL
para_X2_list_b = NULL
skew_list = NULL #skewness of x1
numexplist = NULL #numbers of the trials
mixsiglist = NULL #mix error distribution variance
mixelist = NULL #mix error distribution occur probility 
ite = 1 
err_stats = matrix(NA, nrow = 10000, ncol = 80)#The row number is the different cirsumstance we may want to try
res_trials = matrix(NA, num_trials, 12)
err = matrix(NA, num_exp, 6)
cond_mean = matrix(NA, num_exp, 12)
cond_var = matrix(NA, num_exp, 12)
cond_quan = matrix(NA, num_exp, 12)
res_trials1 = matrix(NA, num_trials, 12)
err1 = matrix(NA, num_exp, 6)
cond_mean1 = matrix(NA, num_exp, 12)
cond_var1 = matrix(NA, num_exp, 12)
cond_quan1 = matrix(NA, num_exp, 12)
time_loop = Sys.time()
for(fun in c("fun1","fun2")){
  for(rand_m in c( "normal")){
    
    funnum = paste0(fun,"_",rand_m)
    
    for (para_num in 1:length(mu_list_x1)) {
      
      
      mu = c(mu_list_x1[para_num], mu_list_x2[para_num])
      sigma = c(sigma_list_x1[para_num], sigma_list_x2[para_num])
      
      for(interval in c(interval_a)){
        funlist = c(funlist, fun)
        randlist = c(randlist, rand_m)
        interlist = c(interlist, interval)
        if(rand_m == "gamma"){
          para_X1_list_a = c(para_X1_list_a,shape1[1])
          para_X1_list_b = c(para_X1_list_b,scale1[1])
          para_X2_list_a = c(para_X2_list_a,shape1[2])
          para_X2_list_b = c(para_X2_list_b,scale1[2])
          skew_list = c(skew_list,2/sqrt(shape1[1]))# skewness is related with parameter x1&x2 ,but x1&x2 may have a little difference on its variance
          
        }
        else if(rand_m == "normal"){
          para_X1_list_a = c(para_X1_list_a,mu[1])
          para_X1_list_b = c(para_X1_list_b,sigma[1])
          para_X2_list_a = c(para_X2_list_a,mu[2])
          para_X2_list_b = c(para_X2_list_b,sigma[2])
          skew_list = c(skew_list,0)
          
        }
        else if(rand_m == "lognormal"){
          para_X1_list_a = c(para_X1_list_a,log_mu[1])
          para_X1_list_b = c(para_X1_list_b,log_sigma[1])
          para_X2_list_a = c(para_X2_list_a,log_mu[2])
          para_X2_list_b = c(para_X2_list_b,log_sigma[2])
          skew_list = c(skew_list,(exp(log_sigma[1]**2)+2)*sqrt((exp(log_sigma[1]**2)-1)))
          
        }
        numexplist = c(numexplist,paste0("Exp_num= ",num_exp," ,Trials_each_exp= ",num_trials,
                                         " ,Samplesize_in_trials= ",samplesize))
        inmlist = c(inmlist, "quantile")
        mixsiglist = c(mixsiglist,mixsigma[para_num])
        mixelist = c(mixelist,mixe[para_num])
        
        for(i in 1:num_exp){
          
          for(j in 1:num_trials){
            
            if(rand_m == "gamma"){
              X = generate_gamma(shape1,scale1,samplesize)
            }
            else if(rand_m == "normal"){
              X1 = generate_mixnormal(mu[1],sigma[1],samplesize,mixsigma[para_num],mixe[para_num])
              X2 = generate_normal(mu[2],sigma[2],samplesize)
              X3 = generate_normal(mu[1],sigma[1],samplesize)
              
            }
            else if(rand_m == "lognormal"){
              X = generate_lognormal(log_mu,log_sigma,samplesize)
            }
            X_mix = cbind(X1,X2)
            X = cbind(X3,X2)
            colnames(X_mix) = c("x1", "x2")
            Y_mix = apply(X_mix, 1, get(fun))
            colnames(X) = c("x1", "x2")
            Y = apply(X, 1, get(fun))
            
            xyz = featureImportance$new(X_mix, Y_mix, interval, "quantile")
            res_trials[j,1:2] = xyz$cal_feaimp_var_mean()
            res_trials[j,3:4] = xyz$cal_feaimp_MeanAD_C_mean_Mean()
            res_trials[j,5:6] = xyz$cal_feaimp_MeanAD_C_mean_Median()
            res_trials[j,7:8] = xyz$cal_feaimp_MeanAD_C_median_Mean()
            res_trials[j,9:10] = xyz$cal_feaimp_MeanAD_C_median_Median()
            xyz1 = featureImportance$new(X, Y, interval, "quantile")
            res_trials1[j,1:2] = xyz1$cal_feaimp_var_mean()
            res_trials1[j,3:4] = xyz1$cal_feaimp_MeanAD_C_mean_Mean()
            res_trials1[j,5:6] = xyz1$cal_feaimp_MeanAD_C_mean_Median()
            res_trials1[j,7:8] = xyz1$cal_feaimp_MeanAD_C_median_Mean()
            res_trials1[j,9:10] = xyz1$cal_feaimp_MeanAD_C_median_Median()
            
            
            if(fun == "fun1"){
              res_trials[j,11] = sqrt((1 + mean(X_mix[,2])) ** 2 * var(X_mix[,1]))
              res_trials[j,12] = sqrt((1 + mean(X_mix[,1])) ** 2 * var(X_mix[,2]))
              res_trials1[j,11] = sqrt((1 + mean(X[,2])) ** 2 * var(X[,1]))
              res_trials1[j,12] = sqrt((1 + mean(X[,1])) ** 2 * var(X[,2]))
              
            }
            else if(fun == "fun2"){
              res_trials[j,11] = sd(X_mix[,1]**2)
              res_trials[j,12] = sd(X_mix[,2]**2)
              res_trials1[j,11] = sd(X[,1]**2)
              res_trials1[j,12] = sd(X[,2]**2)
              
            }
            else if(fun == "fun3"){
              res_trials[j,11] = sd(X_mix[,1]**3)
              res_trials[j,12] =  sd(X_mix[,2]**3)
              res_trials1[j,11] = sd(X[,1]**3)
              res_trials1[j,12] =  sd(X[,2]**3)
              
            }
            else if(fun == "fun4"){
              res_trials[j,11] = sd(X_mix[,1])
              res_trials[j,12] = sd(X_mix[,2])
              res_trials1[j,11] = sd(X[,1])
              res_trials1[j,12] = sd(X[,2])
              
            }
          }
          for(j in 1:6){
            err[i, j] = sum(res_trials[,2*j-1] >= res_trials[,2*j])
            err1[i, j] = sum(res_trials1[,2*j-1] >= res_trials1[,2*j])
            #calculate the mixed feature importance X1,X2 and its variance
            cond_mean[i,(2*j-1):(2*j)] = colMeans(res_trials[,(2*j-1):(2*j)])
            cond_var[i,(2*j-1):(2*j)] = 0#apply(res_trials[,(2*j-1):(2*j)], 2, sd)
            #calculate the no mixed feature importance X3,X2
            cond_mean1[i,(2*j-1):(2*j)] = colMeans(res_trials1[,(2*j-1):(2*j)])
            cond_var1[i,(2*j-1):(2*j)] = 0#apply(res_trials1[,(2*j-1):(2*j)], 2, sd)
            #calculate the mixed and no mixed feature importance x1,x3 at the 5% and 95% percent point
            cond_quan[i,(2*j-1):(2*j)] = quantile(res_trials[,(2*j-1)],c(0.05,0.95))#calculate x1
            cond_quan1[i,(2*j-1):(2*j)] = quantile(res_trials1[,(2*j-1)],c(0.05,0.95))#calculate x3
          }
          
        }
        
        #calculate the mixed distribution feature importance error rate
        err_stats[ite,1:30 ] = c(colMeans(err / num_trials),colMeans(cond_mean),colMeans(cond_var))
        #calculate the no mixed distribution feature importance error rate
        err_stats[ite,31:60 ] = c(colMeans(err1 / num_trials),colMeans(cond_mean1),colMeans(cond_var1))
        #calculate the mixed and no mixed feature importance x1,x3 at the 5% and 95% percent point
        err_stats[ite,61:80 ] = c(colMeans(cond_quan)[1:10],colMeans(cond_quan1)[1:10])
        ite = ite + 1 
        
        
      }
    }
  }
}
print(difftime(Sys.time(), time_loop, units = 'mins'))

err_stats = err_stats[!is.na(err_stats[,1]), ] #  why not use the na ?????
results = data.frame(cbind(funlist, randlist,  interlist, 
                           err_stats,mixsiglist,mixelist,
                           para_X1_list_a,para_X1_list_b,
                           para_X2_list_a,para_X2_list_b,
                           numexplist))
{
  colnames(results) = c("funcsion", "distr",  "intervals", 
                        
                        "mix_err_var_mean","mix_err_MeanAD_C_mean_Mean","mix_err_MeanAD_C_mean_Median","mix_err_MeanAD_C_median_Mean",
                        "mix_err_MeanAD_C_median_Median","mix_err_best",
                        
                        "mix_X1_var_mean","mix_X2_var_mean",
                        "mix_X1_MeanAD_C_mean_Mean","mix_X2_MeanAD_C_mean_Mean",
                        "mix_X1_MeanAD_C_mean_Median","mix_X2_MeanAD_C_mean_Median",
                        "mix_X1_MeanAD_C_median_Mean","mix_X2_MeanAD_C_median_Mean",
                        "mix_X1_MeanAD_C_median_Median","mix_X2_MeanAD_C_median_Median",
                        "mix_X1_best","mix_X2_best",
                        
                        "mix_var_X1_var_mean","mix_var_X2_var_mean",
                        "mix_var_X1_MeanAD_C_mean_Mean","mix_var_X2_MeanAD_C_mean_Mean",
                        "mix_var_X1_MeanAD_C_mean_Median","mix_var_X2_MeanAD_C_mean_Median",
                        "mix_var_X1_MeanAD_C_median_Mean","mix_var_X2_MeanAD_C_median_Mean",
                        "mix_var_X1_MeanAD_C_median_Median","mix_var_X2_MeanAD_C_median_Median",
                        "mix_var_X1_best","mix_var_X2_best",
                        
                        "err_var_mean","err_MeanAD_C_mean_Mean","err_MeanAD_C_mean_Median","err_MeanAD_C_median_Mean",
                        "err_MeanAD_C_median_Median","err_best",
                        
                        "X1_var_mean","X2_var_mean",
                        "X1_MeanAD_C_mean_Mean","X2_MeanAD_C_mean_Mean",
                        "X1_MeanAD_C_mean_Median","X2_MeanAD_C_mean_Median",
                        "X1_MeanAD_C_median_Mean","X2_MeanAD_C_median_Mean",
                        "X1_MeanAD_C_median_Median","X2_MeanAD_C_median_Median",
                        "X1_best","X2_best",
                        
                        "var_X1_var_mean","var_X2_var_mean",
                        "var_X1_MeanAD_C_mean_Mean","var_X2_MeanAD_C_mean_Mean",
                        "var_X1_MeanAD_C_mean_Median","var_X2_MeanAD_C_mean_Median",
                        "var_X1_MeanAD_C_median_Mean","var_X2_MeanAD_C_median_Mean",
                        "var_X1_MeanAD_C_median_Median","var_X2_MeanAD_C_median_Median",
                        "var_X1_best","var_X2_best",
                        
                        "mix_X1_5%_var_mean","mix_X1_95%_var_mean",
                        "mix_X1_5%_MeanAD_C_mean_Mean","mix_X1_95%_MeanAD_C_mean_Mean",
                        "mix_X1_5%_MeanAD_C_mean_Median","mix_X1_95%_MeanAD_C_mean_Median",
                        "mix_X1_5%_MeanAD_C_median_Mean","mix_X1_95%_MeanAD_C_median_Mean",
                        "mix_X1_5%_MeanAD_C_median_Median","mix_X1_95%_MeanAD_C_median_Median",
                        
                        "mix_X3_5%_var_mean","mix_X3_95%_var_mean",
                        "mix_X3_5%_MeanAD_C_mean_Mean","mix_X3_95%_MeanAD_C_mean_Mean",
                        "mix_X3_5%_MeanAD_C_mean_Median","mix_X3_95%_MeanAD_C_mean_Median",
                        "mix_X3_5%_MeanAD_C_median_Mean","mix_X3_95%_MeanAD_C_median_Mean",
                        "mix_X3_5%_MeanAD_C_median_Median","mix_X3_95%_MeanAD_C_median_Median",
                        
                        "mix_sigma","mix_error_rate","x1_a","x1_b","x2_a","x2_b","exp_num")
  
}# SET NAMES
filepath = "d://Rfile/results20191016_mixdist.csv"
write_csv(as.data.frame(Sys.time()), filepath ,append = T)
write_csv(as.data.frame(t(colnames(results))), filepath ,append = T)
write_csv(results, filepath,append = T)
#save(results,file = 'data/artficial.rda')
