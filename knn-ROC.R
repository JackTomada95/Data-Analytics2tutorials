Ytrue <- c(0,0,1,0,1,0,0,0,1,0)
Ypred <- c(0.15,0.29,0.76,0.39,0.41,0.18,0.44,0.09,0.49,0.31)

df <- data.frame(Ytrue, Ypred)
df$output <- NA

# r = 0.5
df$output <- df$Ypred > 0.5
df

TP <- 1
TN <- 7
FP <- 0
FN <- 2

all <- c(TP, FN, FP, TN)
conf.matrix <- matrix(all, nrow = 2)
rownames(conf.matrix) <- c("Positive", "Negative")
colnames(conf.matrix) <- c("Positive", "Negative")

conf.matrix

# calculate the accuracy, recall and precision

accuracy <- (TP + TN) / sum(TP, TN, FP, FN)
precision <- TP / (TP + FP) # rate of accurately detected positives among the predicted positives
recall <- TP / (TP + FN) # rate of detecting positives among all the actual positives
F1score <- 2 * ((precision * recall) / (precision + recall)) # harmonic mean of precision and recall

# if r = 0.4
df2 <- data.frame(Ytrue, Ypred)
df2$output <- NA
df2$output <- df$Ypred > 0.4
df2

TP2 <- 3
TN2 <- 6
FP2 <- 1
FN2 <- 0

all2 <- c(TP2, FN2, FP2, TN2)
conf.matrix2 <- matrix(all2, nrow = 2)
rownames(conf.matrix2) <- c("Positive", "Negative")
colnames(conf.matrix2) <- c("Positive", "Negative")

conf.matrix2

accuracy2 <- (TP2 + TN2) / sum(TP2, TN2, FP2, FN2)
accuracy2 - accuracy
precision2 <- TP2 / (TP2 + FP2) 
recall2 <- TP2 / (TP2 + FN2) 
F1score2 <- 2 * ((precision2 * recall2) / (precision2 + recall2))

accuracy2 - accuracy
F1score2 - F1score
# r = 0.4 is better




# ------------------------------------------------------------------------------- K- Nearest Neighbour

# k-nearest neighbour with r

load("NNData")

is.factor(data$class)

# we already see that the means of x and y are different. (class 1 and class 2 are different)
apply(data[data$class==1,][,1:2], 2, mean)
apply(data[data$class==2,][,1:2], 2, mean)

# now KNN
# generally you pick the square root of the number of datapoints as the value of k
# to normalize or not to normalize? usually it is the case (not this one)
# installation of class library

sqrt(60)
# 7 or 8 is the right value of k

data$class <- as.factor(data$class)

# shuffle the dataset

set.seed(123)
r.data <- sample(nrow(data))
r.data
r.data <- data[r.data,]
r.data

# create the training dataset
data_train <- r.data[1:40,]
data_test <- r.data[41:60,]

library(class)
data_pred <- knn(data_train, data_test, r.data[1:40 ,3], k = 7)

# now validation

data_pred

table(data_pred, r.data[41:60,3]) # it works perfectly

library(ggplot2)

table.plot <- data.frame(r.data[41:60 ,1:2], data_pred)

ggplot(data = table.plot, aes(x=x, y=y, color=data_pred)) + geom_point(size=3)



# ---------------------------------------------------------------- ROC curve

#the roc curve shows the tradeoff between specificity ans sensitivity depending on different values of t (thereshold parameneter)
# sensitivity <- true positive rate (TP / (TP + FN))
# false positive rate <- 1 - true negative rate (specificity) <- (TN / (TN + FP))
# you can compare the performance of different algotrithmns, the ones with the biggest auc is the best













