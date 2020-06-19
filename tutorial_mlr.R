# R's equivalent of scikitlearn (unified interfacem same functions and same arguments)
# example with K-nn


library(mclust)
library(mlr)
library(tidyverse)
library(kknn)


# load the data

diabetesTib <- as_tibble(diabetes) 

diabetesTib
summary(diabetesTib)

# plot data
ggplot(diabetesTib, aes(glucose, insulin, colour=class)) + 
  geom_point()

ggplot(diabetesTib, aes(sspg, insulin, colour=class)) + 
  geom_point()

ggplot(diabetesTib, aes(sspg, glucose, colour=class)) + 
  geom_point()

# with knn, we want to predict a model that is able to predict the class of the patient

######## 1. DEFINE A TASK:
# what is a task in mlr? a task is a definition that you want to achieve:
# - you need to specify the dataset 
# - the target variable (in this case, categorical)
# now we want to train it

diabetesTask <- makeClassifTask(data=diabetesTib, target="class")
diabetesTask

######## 2. DEFINE A LEARNER:
# a learner is the algorithm we choose to learn a model
# we must tell the learner:
# 1. which kind of task we want to do
# 2. The algorithm we want to use (knn, logistic regression, DT etc...)
# 3. Extra options (how the learner will learn the model)

knn <- makeLearner("classif.knn", par.vals = list("k" = 2)) # 2NN model

# LIST if MLR'S algorithms ----
listLearners()$class

# it is convenient to define tasks and learners separately so we can try defferent learners on the same task
# to see which one performs the best
######## 3. TRAINING THE MODEL:
knnModel <- train(knn, diabetesTask)

######## 4a. TESTING PERFORMANCE (THE BAD WAY) common pitfall:
# don't test the data on the same data you used for training the data
# this is how the bad way look like:
knnPred <- predict(knnModel, newdata = diabetesTib)
performance(knnPred) # min missclassification error (very low)


######## 4a. TESTING PERFORMANCE (THE RIGHT WAY) cross-validation:
# 10 folds cross-validation (repeat the process 50 times, very robust estimate)
kfold <- makeResampleDesc("RepCV", folds=10, reps=50)

kfoldCV <- resample(learner=knn, 
                    task=diabetesTask,
                    resampling = kfold)
# in the end you get the average of these values

# confusion matrix
calculateConfusionMatrix(kfoldCV$pred)
roc <- generateThreshVsPerfData(kfoldCV$pred, list(fpr, tpr))

# ----------------------- summary
# the buinding blocks of mlr:
# 1. Task (classification, regression, clustering, survival)
# 2. Learner (conceptual description of the machine learning algorithm)
# 3. Train: training the supervised learning models based on the task
# 4. Predict: using the models to make predictions (on new data)
# 5. performance and resampling: assessing a model's quality
# 6. benchmarking: comparison of meny different learners on many different tasks
# 7. tuning: optimizing learner's hyperparameters
# 8. feature selection: looking for a better subset of features


# load the dataset and split the data randomly into training (600) and test (246)
df.veh <- read.csv(file.choose())

set.seed(42)
rows <- sample(nrow(df.veh))
random.veh <- df.veh[rows, ]
train.veh <- random.veh[1:600, ]
test.veh <- random.veh[601:846, ]

# create the corresponding classification task
task <- makeClassifTask(data=train.veh, target="Class")

# define a learner
learner1 <- makeLearner("classif.knn", par.vals = list("k" = 10))

# train
knnModel <- train(learner1, task)

# predict
pred1 <- predict(knnModel, newdata = test.veh)

# performance
table(pred1$data[, c("truth", "response")])
performance(pred = pred1, measures = list(mmce, acc))


# comparison with more k
sqrt(600)
# let's try with 29 ks

# define a learner
learner2 <- makeLearner("classif.knn", par.vals = list("k" = 4))

# train
knnModel2 <- train(learner2, task)

# predict
pred2 <- predict(knnModel2, newdata = test.veh)

# performance
table(pred2$data[, c("truth", "response")])
performance(pred = pred2, measures = list(mmce, acc))






