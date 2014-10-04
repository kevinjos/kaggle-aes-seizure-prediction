#!/usr/bin/env Rscript

library(RRF)
library(ggplot2)
library(MASS)
library(arm)
library(glmnet)

#Read in data and make train and test sets
data <- read.csv('/home/kjs/repos/kaggle-aes-seizure-prediction/data/features.csv',
                stringsAsFactors=FALSE)
dtest <- data[grep('test', data$filen), ]
dtrain <- data[-grep('test', data$filen), ]

dtrain$target <- 0
dtrain[grep('preictal', dtrain$filen), 'target'] <- 1

dtrain.dog <- dtrain[grep('Dog', dtrain$filen), ]
dtrain.dog.filen <- dtrain.dog$filen
dtrain.dog$filen <- NULL
dtrain.dog <- data.frame(apply(dtrain.dog, 2, as.numeric))

dtrain.pat <- dtrain[grep('Patient', dtrain$filen), ]
dtrain.pat.filen <- dtrain.pat$filen
dtrain.pat$filen <- NULL
dtrain.pat <- data.frame(apply(dtrain.pat, 2, as.numeric))

dtrain.filen <- dtrain$filen
dtrain$filen <- NULL
dtrain <- data.frame(apply(dtrain, 2, as.numeric))

dtest.dog <- dtest[grep('Dog', dtest$filen), ]
dtest.dog.filen <- dtest.dog$filen
dtest.dog$filen <- NULL
dtest.dog <- data.frame(apply(dtest.dog, 2, as.numeric))

dtest.pat <- dtest[grep('Patient', dtest$filen), ]
dtest.pat.filen <- dtest.pat$filen
dtest.pat$filen <- NULL
dtest.pat <- data.frame(apply(dtest.pat, 2, as.numeric))

dtest.filen <- dtest$filen
dtest$filen <- NULL
dtest <- data.frame(apply(dtest, 2, as.numeric))

Range <- function(x) {return(max(x, na.rm=TRUE) - min(x, na.rm=TRUE))}
Mean <- function(x) {return(mean(x, na.rm=TRUE))}
SD <- function(x) {return(sd(x, na.rm=TRUE))}

FixUpData <- function(dtrain){
#Generate summary features across electrodes
#Mean
dtrain[, 'svd_entropy_mean'] <- apply(dtrain[, grep('svd_entropy', colnames(dtrain))], 1, Mean)
dtrain[, 'fisher_mean'] <- apply(dtrain[, grep('fisher', colnames(dtrain))], 1, Mean)
dtrain[, 'dfa_mean'] <- apply(dtrain[, grep('dfa', colnames(dtrain))], 1, Mean)
dtrain[, 'spectral_entropy_mean'] <- apply(dtrain[, grep('spectral_entropy', 
                                          colnames(dtrain))], 1, Mean)
dtrain[, 'higuchi_mean'] <- apply(dtrain[, grep('higuchi', colnames(dtrain))], 1, Mean)
dtrain[, 'hjorthmob_mean'] <- apply(dtrain[, grep('hjorthmob', colnames(dtrain))], 1, Mean)
dtrain[, 'hjorthcom_mean'] <- apply(dtrain[, grep('hjorthcom', colnames(dtrain))], 1, Mean)
dtrain[, 'pfd_mean'] <- apply(dtrain[, grep('pfd', colnames(dtrain))], 1, Mean)
dtrain[, 'mean_mean'] <- apply(dtrain[, grep('mean', colnames(dtrain))], 1, Mean)
#Standard deviation
dtrain[, 'svd_entropy_sd'] <- apply(dtrain[, grep('svd_entropy', colnames(dtrain))], 1, SD)
dtrain[, 'fisher_sd'] <- apply(dtrain[, grep('fisher', colnames(dtrain))], 1, SD)
dtrain[, 'dfa_sd'] <- apply(dtrain[, grep('dfa', colnames(dtrain))], 1, SD)
dtrain[, 'spectral_entropy_sd'] <- apply(dtrain[, grep('spectral_entropy', 
                                          colnames(dtrain))], 1, SD)
dtrain[, 'higuchi_sd'] <- apply(dtrain[, grep('higuchi', colnames(dtrain))], 1, SD)
dtrain[, 'hjorthmob_sd'] <- apply(dtrain[, grep('hjorthmob', colnames(dtrain))], 1, SD)
dtrain[, 'hjorthcom_sd'] <- apply(dtrain[, grep('hjorthcom', colnames(dtrain))], 1, SD)
dtrain[, 'pfd_sd'] <- apply(dtrain[, grep('pfd', colnames(dtrain))], 1, SD)
dtrain[, 'mean_sd'] <- apply(dtrain[, grep('mean', colnames(dtrain))], 1, SD)
#Range
dtrain[, 'svd_entropy_range'] <- apply(dtrain[, grep('svd_entropy', colnames(dtrain))], 1, 
          Range)
dtrain[, 'fisher_range'] <- apply(dtrain[, grep('fisher', colnames(dtrain))], 1, Range)
dtrain[, 'dfa_range'] <- apply(dtrain[, grep('dfa', colnames(dtrain))], 1, Range)
dtrain[, 'spectral_entropy_range'] <- apply(dtrain[, grep('spectral_entropy', 
                                          colnames(dtrain))], 1, Range)
dtrain[, 'higuchi_range'] <- apply(dtrain[, grep('higuchi', colnames(dtrain))], 1, Range)
dtrain[, 'hjorthmob_range'] <- apply(dtrain[, grep('hjorthmob', colnames(dtrain))], 1, Range)
dtrain[, 'hjorthcom_range'] <- apply(dtrain[, grep('hjorthcom', colnames(dtrain))], 1, Range)
dtrain[, 'pfd_range'] <- apply(dtrain[, grep('pfd', colnames(dtrain))], 1, Range)
dtrain[, 'mean_range'] <- apply(dtrain[, grep('mean', colnames(dtrain))], 1, Range)

#Apply mean val of inter-sample feature across electrods to electrodes not present in sample
NAToMean <- function(x, cnames) {
  cnames <- cnames[which(is.na(x))]
  for (cname in cnames) {
    x[cname] <- mean(as.numeric(x[grep(paste0(strsplit(cname, "_")[[1]][1], "_e"), cnames)]), 
                     na.rm=TRUE)
  }
  return(x)
}
dtrain <- data.frame(t(apply(dtrain, 1, function(x) NAToMean(x, colnames(dtrain)))))
return(dtrain)
}

dtrain <- FixUpData(dtrain)
dtrain.dog <- FixUpData(dtrain.dog)
dtrain.pat <- FixUpData(dtrain.pat)
dtest <- FixUpData(dtest)
dtest.dog <- FixUpData(dtest.dog)
dtest.pat <- FixUpData(dtest.pat)

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
set.seed(629)
#set aside a validation set from the training set.. should implement ~10-fold cross validation at some point
train.rows <- sample(1:nrow(dtrain), .9*nrow(dtrain))
train <- dtrain[train.rows, ]
validate <- dtrain[-train.rows, ]

#RRF func wants target out of the feature set
target <- as.factor(train[,'target'])
train$target <- NULL

rf <- RRF(x=train, y=target, ntree=501, maxnodes=244, importance=TRUE, do.trace=10, 
          coefReg=1, flagReg=1)
impRF <- rf$importance
impRF <- impRF[, "MeanDecreaseGini"]
impRF[order(impRF)]
imp <- impRF/(max(impRF))

models <- vector("list")
for (gamma in seq(.2, .95, .05)) {
  coefReg <- (1-gamma)+gamma*imp
  grf <- RRF(x=train, y=target, ntree=501, maxnodes=244, importance=TRUE, do.trace=100, 
             coefReg=coefReg, flagReg=1)
  models[[paste(gamma)]] <- c(grf, Validate(grf, validate))
  print(paste("For gamma of",gamma,"and feature size of",length(grf$feaSet)))
}

save(file="/home/kjs/repos/kaggle-aes-seizure-prediction/data/RRFModels.Rdata", models)

class(models$`0.35`) <- "RRF"
output <- predict(models$`0.35`, dtest, type="prob")
res <- data.frame(clip=paste0(as.character(dtest.filen), ".mat"), 
                  preictal=round(output[, 2], 4))
res <- res[with(res, order(clip)), ]
write.csv(res, row.names=FALSE, quote=FALSE,
        file='/home/kjs/repos/kaggle-aes-seizure-prediction/data/predictions/prediction2.csv')

if (FALSE) {
#Messing with logistic regression
formula <- "target ~ higuchi_e1 + pfd_e0 + hjorthcom_e14 + pfd_e7 + hjorthmob_e10 + hjorthmob_e14 +
                    maxval_e7 + meanval_e1 + hjorthcom_e0 + hjorthcom_e3 + hjorthcom_e5 + higuchi_e14 + 
                    higuchi_e12 + minval_e4 + pfd_e12 + pfd_e14 + minval_e12 + higuchi_sd + hjorthcom_sd + pfd_sd"
glm <- glm(formula = formula, family="binomial", dtrain.train)
glm.aic <- stepAIC(glm)
glm.bayes <- bayesglm(formula=formula, family="binomial", dtrain.train)
Y <- dog.train$target
X <- as.matrix(dog.train[, !(colnames(dog.train) %in% 'target')])
glm.reg <- glmnet(x=X, y=Y, family="binomial")

Y_VAL <- factor(dog.validate$target)
X_VAL <- as.matrix(dog.validate[, !(colnames(dog.validate) %in% 'target')])


dog.train$predict <- predict(glm.aic, type="response")
plot <- ggplot(dog.train, aes(x=1:nrow(dog.train), y=predict, color=target)) + geom_point(stat='identity')

p <- predice(glm.reg, newx=X_VAL, type="response")
dog.validate$predict <- p[,50]
plot <- ggplot(dog.validate, aes(x=1:nrow(dog.validate), y=predict, color=target)) + geom_point(stat='identity')
}
}
