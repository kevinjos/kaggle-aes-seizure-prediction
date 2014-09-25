#!/usr/bin/env Rscript

library(RRF)
library(ggplot2)
library(MASS)
library(arm)

data <- read.csv('/home/kjs/repos/kaggle-aes-seizure-prediction/dog_features.csv')

data$target <- 0
data[grep('preictal', data$filen), 'target'] <- 1
data$target <- as.factor(data$target)
data <- data[, !(colnames(data) %in% grep('samplesize', colnames(data), value=TRUE))]

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

#Generate summary features across electrodes
data.dog$higuchi_mean <- apply(data.dog[, grep('higuchi', colnames(data.dog))], 1, mean)
data.dog$hjorthmob_mean <- apply(data.dog[, grep('hjorthmob', colnames(data.dog))], 1, mean)
data.dog$hjorthcom_mean <- apply(data.dog[, grep('hjorthcom', colnames(data.dog))], 1, mean)
data.dog$pfd_mean <- apply(data.dog[, grep('pfd', colnames(data.dog))], 1, mean)
data.dog$mean_mean <- apply(data.dog[, grep('mean', colnames(data.dog))], 1, mean)

data.dog$higuchi_sd <- apply(data.dog[, grep('higuchi', colnames(data.dog))], 1, sd)
data.dog$hjorthmob_sd <- apply(data.dog[, grep('hjorthmob', colnames(data.dog))], 1, sd)
data.dog$hjorthcom_sd <- apply(data.dog[, grep('hjorthcom', colnames(data.dog))], 1, sd)
data.dog$pfd_sd <- apply(data.dog[, grep('pfd', colnames(data.dog))], 1, sd)
data.dog$mean_sd <- apply(data.dog[, grep('mean', colnames(data.dog))], 1, sd)

data.dog$higuchi_range <- apply(data.dog[, grep('higuchi', colnames(data.dog))], 1, 
          function(x) max(x) - min(x))
data.dog$hjorthmob_range <- apply(data.dog[, grep('hjorthmob', colnames(data.dog))], 1, 
          function(x) max(x) - min(x))
data.dog$hjorthcom_range <- apply(data.dog[, grep('hjorthcom', colnames(data.dog))], 1,
          function(x) max(x) - min(x))
data.dog$pfd_range <- apply(data.dog[, grep('pfd', colnames(data.dog))], 1, 
          function(x) max(x) - min(x))
data.dog$mean_range <- apply(data.dog[, grep('mean', colnames(data.dog))], 1, 
          function(x) max(x) - min(x))

#set aside a validation set from the training set.. should implement ~10-fold cross validation at some point
train <- sample(1:nrow(data.dog), .9*nrow(data.dog))
dog.train <- data.dog[train, ]
dog.validate <- data.dog[-train, ]


#RRF func wants target out of the feature set
target <- as.factor(dog.train$target)
dog.train$target <- NULL


Validate <- function(model, validation.set) {
  Predictions <- predict(model, validation.set[, !(colnames(validation.set) %in% c("target"))])
  Actual <- validation.set$target
  cm <- table(Actual, Predictions)
  accuracy <- (cm[1] + cm[4])/(cm[1]+cm[2]+cm[3]+cm[4])
  recall <- cm[4]/(cm[3]+cm[4])
  precision <- cm[4]/(cm[2]+cm[4])
  fscore <- (2 * recall * precision) / (recall + precision)
  acc <- data.frame(tp=cm[4], fp=cm[2], tn=cm[1], fn=cm[3], 
                    accuracy=accuracy, fscore=fscore, recall=recall, precision=precision,
                    features=length(model$feaSet))
  return(acc)
}

if (FALSE) {
rf <- RRF(x=dog.train, y=target, ntree=501, maxnodes=101, importance=TRUE, do.trace=10, 
          coefReg=1, flagReg=1)
impRF <- rf$importance
impRF <- impRF[, "MeanDecreaseGini"]
impRF[order(impRF)]
imp <- impRF/(max(impRF))

models <- vector("list")
for (gamma in seq(.2, .95, .05)) {
  coefReg <- (1-gamma)+gamma*imp
  grf <- RRF(x=dog.train, y=target, ntree=501, maxnodes=101, importance=TRUE, do.trace=10, 
             coefReg=coefReg, flagReg=1)
  models[[paste(gamma)]] <- c(grf, Validate(grf, dog.validate))
  print(paste("For gamma of",gamma,"and feature size of",length(grf$feaSet)))
}
}

#Messing with logistic regression
formula <- "target ~ higuchi_e1 + pfd_e0 + hjorthcom_e14 + pfd_e7 + hjorthmob_e10 + hjorthmob_e14 +
                    maxval_e7 + meanval_e1 + hjorthcom_e0 + hjorthcom_e3 + hjorthcom_e5 + higuchi_e14 + 
                    higuchi_e12 + minval_e4 + pfd_e12 + pfd_e14 + minval_e12 + higuchi_sd + hjorthcom_sd + pfd_sd"
glm <- glm(formula = formula, family="binomial", data=dog.train)
glm.aic <- stepAIC(glm)
glm.bayes <- bayesglm(formula=formula, family="binomial", data=dog.train)
Y <- dog.train$target
X <- as.matrix(dog.train[, !(colnames(dog.train) %in% 'target')])
glm.reg <- glmnet(x=X, y=Y, family="binomial")

if (FALSE) {
Y_VAL <- factor(dog.validate$target)
X_VAL <- as.matrix(dog.validate[, !(colnames(dog.validate) %in% 'target')])


dog.train$predict <- predict(glm.aic, type="response")
plot <- ggplot(dog.train, aes(x=1:nrow(dog.train), y=predict, color=target)) + geom_point(stat='identity')

p <- predice(glm.reg, newx=X_VAL, type="response")
dog.validate$predict <- p[,50]
plot <- ggplot(dog.validate, aes(x=1:nrow(dog.validate), y=predict, color=target)) + geom_point(stat='identity')
}



