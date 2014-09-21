#!/usr/bin/env Rscript

library(RRF)

data <- read.csv('/home/kjs/repos/kaggle-aes-seizure-prediction/train_features.csv')

data$target <- 0
data[grep('preictal', data$filen), 'target'] <- 1

data.dog <- data[grep('Dog', data$filen), ]
all.null.dog <- grep('*[a-z]_e(1[6-9]|2[0-9])', colnames(data.dog))
data.dog <- data.dog[, -all.null.dog]
data.dog$filen <- NULL

data.human <- data[grep('Patient', data$filen), ]

set.seed(629)

#How does na.action work? Not on predictors?
#One dog does not have data for e15... how to handle???
e15 <- grep('*[a-z]_e15', colnames(data.dog), value=T)
for (col in e15) {
  data.dog[is.na(data.dog[col]), col] <- mean(data.dog[[col]], na.rm=TRUE)
}

#Experiment in cross validation... need to extract target from sample groups
X <- data.dog
id <- sample(1:5, nrow(X), replace=TRUE)
ListX <- split(X, id)

train <- sample(1:nrow(data.dog), .9*nrow(data.dog))
dog.train <- data.dog[train, ]
dog.validate <- data.dog[-train, ]

dog.train.means <- data.frame(
higuchi = apply(dog.train[, grep('higuchi', colnames(dog.train))], 1, mean),
hjorthmob= apply(dog.train[, grep('hjorthmob', colnames(dog.train))], 1,mean),
hjorthcom= apply(dog.train[, grep('hjorthcom', colnames(dog.train))], 1,mean),
pfd = apply(dog.train[, grep('pfd', colnames(dog.train))], 1, mean),
mean = apply(dog.train[, grep('mean', colnames(dog.train))], 1, mean))

target <- as.factor(dog.train$target)
dog.train$target <- NULL

rf <- RRF(dog.train, target, flagReg = 0)
impRF <- rf$importance
impRF <- impRF[, "MeanDecreaseGini"]
impRF[order(impRF)]

rrf <- RRF(dog.train, target, flagReg=1)

imp <- impRF/(max(impRF))

TT <- function(actual, predicted) {
  tt <- data.frame(actual=actual, predicted=predicted)
  TP <- dim(tt[which(tt$actual == 1 & tt$predicted == 1), ])[1]
  TN <- dim(tt[which(tt$actual == 0 & tt$predicted == 0), ])[1]
  FP <- dim(tt[which(tt$actual == 0 & tt$predicted == 1), ])[1]
  FN <- dim(tt[which(tt$actual == 1 & tt$predicted == 0), ])[1]
  print(paste("TP=", TP, "FP=", FP, "TN=", TN, "FN=", FN))
}

models <- vector("list", 8)
for (gamma in seq(.05, .4, .05)) {
  coefReg <- (1-gamma)+gamma*imp
  grf <- RRF(dog.train, target, coefReg=coefReg, flagReg=0)
  models[[paste(gamma)]] <- grf
  print(paste("For gamma of",gamma,"and feature size of",length(grf$feaSet)))
  TT(target, grf$predicted)
}

glm <- glm(target ~ dog.train$higuchi_e12 + dog.train$meanval_e8 + 
                    dog.train$meanval_e3 + dog.train$hjorthmob_e2 +
                    dog.train$higuchi_e5 + dog.train$higuchi_e3 +
                    dog.train$hjorthcom_e14, family = binomial)

glm.means <- glm(target ~ dog.train.means$higuchi +
                          dog.train.means$hjorthmob +
                          dog.train.means$hjorthcom +
                          dog.train.means$pfd +
                          dog.train.means$mean, family = binomial)

