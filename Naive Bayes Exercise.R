library(mclust)
library(mlr)
library(tidyverse)
library(kknn)
library(e1071)

Weather <- c("sunny", "sunny","sunny", "rainy", "cloudy", "rainy", "rainy", "cloudy")
Temperature <- c("hot","hot","hot","mild","cold","mild","cold","hot")
Wind <- c("strong","medium","low","strong","low","strong","strong","medium")
Class <- c("no","no","yes","no","yes","no","yes","yes")

naive <- data.frame(Weather, Temperature, Wind, Class)

# to which class will observation x = (rainy, hot, medium) be assigned??

# we want to see if the probability of belonging to the class "yes" is bigger than the probability of belonging to the class "no"

# let's start by calculating P(Class = Yes|rainy, hot, medium) 

APrioriYes <- nrow(naive[naive$Class == "yes",]) / (nrow(naive[naive$Class == "yes",]) + nrow(naive[naive$Class == "no",]))

YesRainy <- nrow(naive[naive$Class == "yes" & naive$Weather == "rainy",]) / 4
YesHot <- nrow(naive[naive$Class == "yes" & naive$Temperature == "hot",]) / 4
YesMedium <- nrow(naive[naive$Class == "yes" & naive$Wind == "medium",]) / 4

probYes <- APrioriYes * YesRainy * YesMedium * YesHot

# now P(Class = No/rainy, hot, medium)

APrioriNo <- nrow(naive[naive$Class == "no",]) / (nrow(naive[naive$Class == "yes",]) + nrow(naive[naive$Class == "no",]))

NoRainy <- nrow(naive[naive$Class == "no" & naive$Weather == "rainy",]) / 4
NoHot <- nrow(naive[naive$Class == "no" & naive$Temperature == "hot",]) / 4
NoMedium <- nrow(naive[naive$Class == "no" & naive$Wind == "medium",]) / 4

probNo <- APrioriNo * NoRainy * NoMedium * NoHot

# comparison

probYes > probNo

# it will be classified as NO

# -------------------------------------------------------------- Additive Smoothing
# if we want to do the same thing for P(C = yes | cloudy,mild,low), we have a problem
# no mild weather is classified as yes, so the probability will always be zero
# this problem can be fixed -> k = 1 to avoid 0 probability 

# ---------- GAUSSIAN NAIVE BAYES for continuous variables on iris dataset

# create a task

naivetask <- makeClassifTask(data=iris, target="Species")
naivetask

# create a learner
listLearners()$class
naivelearner <- makeLearner("classif.naiveBayes")

# train the model (optional)

model <- train(naivelearner, naivetask)
model

# evaluate performance (5-fold cross validation)

kfold <- makeResampleDesc("RepCV", folds=5)

kfoldCV <- resample(learner=naivelearner, 
                    task=naivetask,
                    resampling = kfold)

calculateConfusionMatrix(kfoldCV$pred)

# note: naive bayes works very well with spam detection (probability of finding words in an email)
# why naive??? strong assumption: independence between features






