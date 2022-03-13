######   Library loading ############

library(class)
library(MASS)
library(kernlab)
library(mlbench)
library(reshape2)
library(ROCR)
library(ggplot2)
library(ada)
library(adabag)
library(ipred)
library(survival)
library(rchallenge)
library(PerformanceAnalytics)
library(knitr)
library(acepack)
library(caret)
library(HSAUR2)
library(corrplot)

######  Loading dataset   ##############

library(readr)

library(dplyr)

mnist_train <- read_csv("https://pjreddie.com/media/files/mnist_train.csv", col_names = FALSE)

mnist_test <- read_csv("https://pjreddie.com/media/files/mnist_test.csv", col_names = FALSE)


########### Lets see the data   ##########33
head(mnist_train)
tail(mnist_train)
str(mnist_train)
str(mnist_test)

which(is.na(mnist_train))
which(is.na(mnist_test))


######  Extracting data for only 1 and 7    ################


mtrain=mnist_train[mnist_train$X1==1 | mnist_train$X1==7, ]
mtest=mnist_test[mnist_test$X1==1 | mnist_test$X1==7, ]
head(mtrain)
head(mtest)


##############  Definition of dataset #############3

x_train=mtrain[,-1]
y_train=mtrain[,1,drop=TRUE]
x_test=mtest[,-1]
y_test=mtest[,1,drop=TRUE]
sum(is.na(x_test))
####  Distribution of the label #############3
barplot(prop.table(table(y_train)), col=2:3, xlab='digits')


############# 1 nearest neighbors binary classifier on the test set, and the  corresponding confusion matrix ##########


k   <- 1
y.te.hat <- knn(x_train, x_test, y_train, k=k) # Predicted responses in test set

conf.mat.te <- table(y_test, y.te.hat)
conf.mat.te

############   1-NN confusion matrix on the training set #################
k        <- 1
y.tr.hat <- knn(x_train, x_train, y_train, k=k) # Predicted responses in test set

conf.mat.tr <- table(y_train, y.tr.hat)
conf.mat.tr

#########   7-NN confusion matrix on the test set##################
k   <- 7
# True responses in test set
y.te.hat <- knn(x_train, x_test, y_train, k=k) # Predicted responses in test set

conf.mat.te <- table(y_test, y.te.hat)
conf.mat.te

#########   7-NN confusion matrix on the training set##################
k        <- 7
y.tr.hat <- knn(x_train, x_train, y_train, k=k) # Predicted responses in test set

conf.mat.tr <- table(y_train, y.tr.hat)
conf.mat.tr

#########   9-NN confusion matrix on the test set##################

k   <- 9
y.te.hat <- knn(x_train, x_test, y_train, k=k) # Predicted responses in test set

conf.mat.te <- table(y_test, y.te.hat)
conf.mat.te
#########   9-NN confusion matrix on the training set##################
k        <- 9
y.tr.hat <- knn(x_train, x_train, y_train, k=k) # Predicted responses in test set

conf.mat.tr <- table(y_train, y.tr.hat)
conf.mat.tr

########  ROC Curve:Comparing models ROC Curve  ######
library(ROCR)

y.roctr <- ifelse(y_train==1,1,0)
y.rocte <- ifelse(y_test==1,1,0)

kNN.mod <- class::knn(x_train,x_test, y.roctr, k=1, prob=TRUE)
prob    <- attr(kNN.mod, 'prob')
prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1

pred1.knn <- prediction(prob, y.rocte)
perf1.knn <- performance(pred1.knn, measure='tpr', x.measure='fpr')



kNN.mod <- class::knn(x_train,x_test, y.roctr, k=7, prob=TRUE)
prob    <- attr(kNN.mod, 'prob')
prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1

pred7.knn <- prediction(prob, y.rocte)
perf7.knn <- performance(pred7.knn, measure='tpr', x.measure='fpr')


kNN.mod <- class::knn(x_train,x_test, y.roctr, k=9, prob=TRUE)
prob    <- attr(kNN.mod, 'prob')
prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1

pred9.knn <- prediction(prob, y.rocte)
perf9.knn <- performance(pred9.knn, measure='tpr', x.measure='fpr')

plot(perf1.knn, col=2, lwd= 2, lty=2, main=paste('Comparison of Predictive ROC curves'))
plot(perf7.knn, col=3, lwd= 2, lty=3, add=TRUE)
plot(perf9.knn, col=4, lwd= 2, lty=4, add=TRUE)
abline(a=0,b=1)
legend('bottomright', inset=0.05, c('1NN','7NN', '9NN'),  col=2:4, lty=2:4)

library(xtable)
xtable(conf.mat.tr)


#######3  Bonus 2 ############3

#######3  Loadin prostate cancer data #########3
dt<-read.csv("prostate-cancer-1.csv", header=TRUE)
# load the data
xy <- dt # Store data in xy frame


#########Let's have a look at the first six and the last six observations

head(xy)
tail(xy)



#######   Let's also ascertain the types of the variables 
str(xy)
which(is.na(xy))



x.tr=xy[,2:ncol(xy)]
y.tr=xy[,1,drop=TRUE]
x.te=x.tr
y.te=y.tr


##################### Comparative ROC Curves on the test set for the models ##################

library(ROCR)

kNN.mod <- class::knn(x.tr, x.te, y.tr, k=1, prob=TRUE)
prob    <- attr(kNN.mod, 'prob')
prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1

pred.1NN <- prediction(prob, y.te)
perf.1NN <- performance(pred.1NN, measure='tpr', x.measure='fpr')

kNN.mod <- class::knn(x.tr, x.te, y.tr, k=3, prob=TRUE)
prob    <- attr(kNN.mod, 'prob')
prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1

pred.3NN <- prediction(prob, y.te)
perf.3NN <- performance(pred.3NN, measure='tpr', x.measure='fpr')

kNN.mod <- class::knn(x.tr, x.te, y.tr, k=5, prob=TRUE)
prob    <- attr(kNN.mod, 'prob')
prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1

pred.5NN <- prediction(prob, y.te)
perf.5NN <- performance(pred.5NN, measure='tpr', x.measure='fpr')

kNN.mod <- class::knn(x.tr, x.te, y.tr, k=7, prob=TRUE)
prob    <- attr(kNN.mod, 'prob')
prob    <- 2*ifelse(kNN.mod == "0", 1-prob, prob) - 1

pred.7NN <- prediction(prob, y.te)
perf.7NN <- performance(pred.7NN, measure='tpr', x.measure='fpr')

plot(perf.1NN, col=2, lwd= 2, lty=2, main=paste('Comparison of Predictive ROC curves'))
plot(perf.3NN, col=3, lwd= 2, lty=3, add=TRUE)
plot(perf.5NN, col=4, lwd= 2, lty=4, add=TRUE)
plot(perf.7NN, col=5, lwd= 2, lty=5, add=TRUE)
abline(a=0,b=1)
legend('bottomright', inset=0.05, c('1NN','3NN','5NN', '7NN'),  col=2:5, lty=2:5)




##############  Comparative boxplot  ########################

set.seed (19671210)          # Set seed for random number generation to be reproducible
n=nrow(xy)
epsilon <- 1/3               # Proportion of observations in the test set
nte     <- round(n*epsilon)  # Number of observations in the test set
ntr     <- n - nte

R <- 100   # Number of replications
test.err <- matrix(0, nrow=R, ncol=5)

for(r in 1:R)
{
  # Split the data
  
  id.tr   <- sample(sample(sample(n)))[1:ntr]                   # For a sample of ntr indices from {1,2,..,n}
  id.te   <- setdiff(1:n, id.tr)
  
  y.tee         <- y.tr[id.te]                                        # True responses in test set
  
  # First machine: 1NN
  
  y.tee.hat     <- knn(x.tr[id.tr,], x.tr[id.te,], y.tr[id.tr], k=1)        # Predicted responses in test set
  ind.err.te   <- ifelse(y.tee!=y.tee.hat,1,0)                      # Random variable tracking error. Indicator
  test.err[r,1]  <- mean(ind.err.te)
  
  # Second machine: k=3
  y.tee.hat     <- knn(x.tr[id.tr,], x.tr[id.te,], y.tr[id.tr], k=3) # Predicted responses in test set
  ind.err.te   <- ifelse(y.tee!=y.tee.hat,1,0)                      # Random variable tracking error. Indicator
  test.err[r,2]  <- mean(ind.err.te)
  
  # Third machine: k=5
  y.tee.hat     <- knn(x.tr[id.tr,], x.tr[id.te,], y.tr[id.tr], k=5)       # Predicted responses in test set
  ind.err.te   <- ifelse(y.tee!=y.tee.hat,1,0)                      # Random variable tracking error. Indicator
  test.err[r,3]  <- mean(ind.err.te)
  
  
  # Fourth machine: k=7
  y.tee.hat     <- knn(x.tr[id.tr,], x.tr[id.te,], y.tr[id.tr], k=7)     # Predicted responses in test set
  ind.err.te   <- ifelse(y.tee!=y.tee.hat,1,0)                      # Random variable tracking error. Indicator
  test.err[r,4]  <- mean(ind.err.te)
  }
test <- data.frame(test.err)
Method<-c('1NN', '3NN', '5NN', '7NN')
colnames(test) <- Method
boxplot(test)
