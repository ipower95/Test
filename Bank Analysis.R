####Setup####
setwd("C:/Users/Use/Desktop/R codes/bank")
Raw.data <- read.csv("bank-full.csv", sep = ";", header = T)
install.packages("ggplot2")
install.packages("caret")
install.packages("doParallel")
install.packages("tree")
install.packages("randomForest")
install.packages("gbm")
install.packages("sjPlot")
install.packages("lattice")
install.packages("kknn")
install.packages("MASS")
install.packages("outliers")
install.packages("e1071")
install.packages("gbm")
install.packages("kknn")
#install.packages("doSNOW")

library(ggplot2)
library(caret)
library(doParallel)
library(tree)
library(randomForest)
library(gbm)
library(sjPlot)
library(lattice)
library(kknn)
library(MASS)
library(outliers)
library(e1071)
library(class)
library(kknn)

#library("doSNOW")


####Random Sampling####
m = dim(Raw.data)[1]
set.seed(1)
val = sample(1:m,size=round(m*0.3),replace=FALSE,prob=rep(1/m,m))
Train = Raw.data[-val,]
Test = Raw.data[val,]
View(Test)

####Simple Tree####
#Simple Tree, (no tuning)
tree.train <- tree(y~.,data = Train)
plot(tree.train)
text(tree.train,cex=0.8) 
View(tree.train)
tree.pred <- predict(tree.train,Test,type="class") #predict(model,test,type?)
result.tree.1 <- confusionMatrix(tree.pred,Test$y) #package confusion matrix
result.tree.1

##old method of doing the confusion matrix
#performance.1 <- table(tree.pred,Test$y)
#Accuracy.1 <- (performance.1[1,1]+performance.1[2,2])/sum(performance.1[,])
#Accuracy.1*100
#performance.1


#Simple Tree with (Tuned formula based on hypothesis)
tree.train.2 <- tree(y~age+job+marital+education+default+balance+housing+loan+contact+duration, data = Train)
plot(tree.train.2)
text(tree.train.2,cex=0.8) 
tree.pred.2 <- predict(tree.train.2,Test,type="class") 
result.tree.2 <- confusionMatrix(tree.pred.2,Test$y)
result.tree.2
#drop in accuracy

#Simple tree based on (varimplot of random forest)
tree.train.3 <- tree(y~age+duration+month+day+contact+poutcome+housing, data = Train)
plot(tree.train.3)
text(tree.train.3,cex=0.8) 
tree.pred.3 <- predict(tree.train.3,Test,type="class") 
result.tree.3 <- confusionMatrix(tree.pred.3,Test$y)
result.tree.3
#exactly same with 1st tree

#tree pruning
prune.y <- prune.misclass(tree.train,best=9) #best 9 terminal nodes aka leaves
plot(prune.y)
text(prune.y,cex=0.8)
prune.pred <- predict(prune.y,Test,type="class")
result.prune.1 <- confusionMatrix(prune.pred,Test$y)
result.prune.1

prune.y.2 <- prune.misclass(tree.train,best=7)
plot(prune.y.2)
text(prune.y.2,cex=0.8)
prune.pred.2 <- predict(prune.y.2,Test,type="class")
result.prune.2 <- confusionMatrix(tree.pred.2,Test$y)
result.prune.2
#pruning doesn't do much for the model. the cut off point is best of 8 originally it is 9

####Random Forrest####

##finding the best formula
#setting label
rf.label <- as.factor(Train$y)
#all variables
rf.train.1 <- Train[,1:16]
set.seed(20)
rf.1 <- randomForest(x= rf.train.1, y = rf.label, importance = T)
rf.1
varImpPlot(rf.1)
rf.1.pred <- predict(rf.1, Test[,1:16])
result.rf.1 <- confusionMatrix(rf.1.pred, Test$y)
result.rf.1
#rf.perf.1<- table(rf.1.pred,Test$y) #get table the old way
#rf.perf.1
#m-try = 4 by default thus we adopt this as the random forest model


#randomly choose variables (based on hypothesis)
set.seed(20)
rf.train.2 <- Train[, c("age","job","marital","education","default","balance","housing","loan","contact","duration")]
rf.2 <- randomForest(x= rf.train.2, y = rf.label, importance = T)
rf.2
varImpPlot(rf.2)
rf.2.pred <- predict(rf.2, Test[,c("age","job","marital","education","default","balance","housing","loan","contact","duration")])
result.rf.2 <- confusionMatrix(rf.2.pred, Test$y)
result.rf.2

#top 6 variables from rf.1 (using VarImplot of first random forest model)
set.seed(20)
rf.train.3 <- Train[, c("age","duration","month","day","contact","poutcome","housing")]
rf.3 <- randomForest(x= rf.train.3, y = rf.label, importance = T)
rf.3
varImpPlot(rf.3)
rf.3.pred <- predict(rf.3, Test[,c("age","duration","month","day","contact","poutcome","housing")])
result.rf.3 <- confusionMatrix(rf.3.pred, Test$y)
result.rf.3
#higher class 2 error. (since this data set is skewed more to "No", accurate prediction on yes is more important)
#thus we choose rf.1

####Bagging####
#bagging using all predictors (16 variables)
rf.train.4 <- Train[,1:16]
set.seed(20)
rf.4 <- randomForest(x= rf.train.4, y = rf.label, importance = T, mtry = 16)
rf.4
varImpPlot(rf.4)
rf.4.pred <- predict(rf.4, Test[,1:16])
result.bagging <- confusionMatrix(rf.4.pred, Test$y)
result.bagging
#performs better than rf.1
#compare rf.1 & rf.4 
#cross validation if got time
#we adopt rf.4(bagging)

####Boosting####
 
set.seed(20)
boost.label <- ifelse(Train$y=="yes",1,0)
Train.boost <- data.frame(Train,boost.label)
boost.label <- ifelse(Test$y=="yes",1,0)
Test.boost <- data.frame(Test,boost.label)
rm(boost.label)

boost.Train <- gbm(boost.label~.-y,data=Train.boost,distribution="bernoulli",n.trees=5000,interaction.depth=4)
summary(boost.Train)
boost.probs <- predict(boost.Train,newdata=Test,type="response",n.trees=5000)
boost.pred <- rep(0,13563)
boost.pred[boost.probs>=0.5] <- 1
boost.pred <- ifelse(boost.pred == 1, "yes", "no")
boost.pred <- as.factor(boost.pred)
result.boost <- confusionMatrix(boost.pred,Test$y)
result.boost

boost.Train.2 <- gbm(boost.label~.-y,data=Train.boost,distribution="bernoulli",n.trees=500,interaction.depth=4)
summary(boost.Train)
boost.probs.2 <- predict(boost.Train,newdata=Test,type="response",n.trees=500)
boost.pred.2 <- rep(0,13563)
boost.pred.2[boost.probs.2>=0.5] <- 1
boost.pred.2 <- ifelse(boost.pred.2 == 1, "yes", "no")
boost.pred.2 <- as.factor(boost.pred.2)
result.boost.2 <- confusionMatrix(boost.pred.2,Test$y)
result.boost.2
#when n trees is at 500, no "yes prediction was made", n = 3000 there are yes prediction
#loads of tuning parameter, would take ages to fully capitalize on the algorithm
#processing power needed is high


####K-NN####
#Setup
Raw.data.2 <- Raw.data
Raw.data.2[,c(2:5,7:9,11,16)] <- sapply(Raw.data.2[,c(2:5,7:9,11,16)], as.numeric)
str(Raw.data.2)

#normalize
normal <- function(x) {
  return ((x - min(x)) /(max(x)-min(x))) }

Bank.norm <- data.frame(lapply(Raw.data.2[1:16], normal), y = Raw.data.2$y)
str(Bank.norm)
Train.3 <- Bank.norm[-val,]
Test.3 <- Bank.norm[val,]

##K's nature in KNN
#using k=8
knn.pred <- knn(Train.3[1:16],Test.3[1:16],Train.3$y,k=8)
result.knn <- confusionMatrix(knn.pred,Test.3$y)
result.knn

#rule of thumb selecting k sqrt(n) k=177
knn.pred.2 <- knn(Train.3[1:16],Test.3[1:16],Train.3$y,k=177)
result.knn.2 <- confusionMatrix(knn.pred.2,Test.3$y)
result.knn.2

#formula optimisation (based on Random Forest result)
#k=1
knn.pred.3 <- knn(Train.3[,c("age","duration","month","day","contact","poutcome","housing")],Test.3[,c("age","duration","month","day","contact","poutcome","housing")],Train.3$y,k=1)
result.knn.3 <- confusionMatrix(knn.pred.3,Test.3$y)
result.knn.3

#k=177
knn.pred.4 <- knn(Train.3[,c("age","duration","month","day","contact","poutcome","housing")],Test.3[,c("age","duration","month","day","contact","poutcome","housing")],Train.3$y,k=177)
result.knn.4 <- confusionMatrix(knn.pred.3,Test.3$y)  
result.knn.4


####KKNN#### 

kknn.pred <- kknn(y~., Train, Test, distance = 3, kernel = "triangular")
summary(kknn.pred)
result.kknn <- confusionMatrix(kknn.pred$fitted.values,Test$y)
result.kknn

#optimised formula, k = 1
kknn.pred.2 <- kknn(y~age+duration+month+day+contact+poutcome+housing, k=1, Train, Test, distance = 3, kernel = "triangular")
summary(kknn.pred.2)
result.kknn.2 <- confusionMatrix(kknn.pred.2$fitted.values,Test$y)
result.kknn.2

####LDA####
#inclusion of all variables
lda.fit <- lda(y~., Train)
lda.pred <- predict(lda.fit,Test)
result.lda <- confusionMatrix(lda.pred$class,Test$y)
result.lda

#top 6 based on varImplot from Random forest
lda.fit.2 <- lda(y~age+duration+month+day+contact+poutcome+housing, Train)
lda.pred.2 <- predict(lda.fit.2,Test)
result.lda.2 <- confusionMatrix(lda.pred.2$class,Test$y)
result.lda.2

#https://machinelearningmastery.com/linear-discriminant-analysis-for-machine-learning/ (richard look in to this)
#consider taking out, outliers, making the distribution normal, and same variance.. so do check it out
#check the extentions as well, QDA, FDA, RDA

####Logistic Regression####

#all variables
logreg.fit <- glm(y~., family = "binomial", data = Train)
summary(logreg.fit)
logreg.probs <- predict(logreg.fit,Test,type = "response")
str(logreg.probs)
summary(logreg.probs)
logreg.probs[1:10] #first tenth
Test[1,] #y = No, and probs is less than 0.5
Test[5,] #y = No, probs is near 0.5, which is still correct
logreg.probs[11:20]
Test[18,] #y = yes, prob is greater than 0.5, which is correct
contrasts(Test$y) # No = 0 and Yes = 1, therefore, y >= 0.5 is yes, while y < 0.5 is no, since skewed towards No
logreg.pred <- rep("no", 13563)
logreg.pred[logreg.probs > 0.5] <- "yes"
logreg.pred <- as.factor(logreg.pred)
result.logreg <- confusionMatrix(logreg.pred, Test$y)
result.logreg

#top 6 in varimplot in random forest
logreg.fit.2 <- glm(y~age+duration+month+day+contact+poutcome+housing, family = "binomial", data = Train)
summary(logreg.fit.2)
logreg.probs.2 <- predict(logreg.fit.2,Test,type = "response")
summary(logreg.probs.2)
logreg.pred.2 <- rep("no", 13563)
logreg.pred.2[logreg.probs.2 > 0.5] <- "yes"
logreg.pred.2 <- as.factor(logreg.pred.2)
result.logreg.2 <- confusionMatrix(logreg.pred.2, Test$y)
result.logreg.2

#based on summary of first regression (significance value)
logreg.fit.3 <- glm(y~job+marital+education+balance+housing+loan+contact+day+month+campaign+previous+poutcome, family = "binomial", data = Train)
summary(logreg.fit.3)
logreg.probs.3 <- predict(logreg.fit.3,Test,type = "response")
summary(logreg.probs.3)
logreg.pred.3 <- rep("no", 13563)
logreg.pred.3[logreg.probs.3 > 0.5] <- "yes"
logreg.pred.3 <- as.factor(logreg.pred.3)
result.logreg.3 <- confusionMatrix(logreg.pred.3, Test$y)
result.logreg.3

logreg.fit.4 <- glm(y~.-age-pdays-default, family = "binomial", data = Train)
summary(logreg.fit.4)
logreg.probs.4 <- predict(logreg.fit.4,Test,type = "response")
summary(logreg.probs.4)
logreg.pred.4 <- rep("no", 13563)
logreg.pred.4[logreg.probs.4 > 0.5] <- "yes"
logreg.pred.4 <- as.factor(logreg.pred.4)
result.logreg.4 <- confusionMatrix(logreg.pred.4, Test$y)
result.logreg.4



####Extra work (cleaning data) Which includes understanding the data, stratifying, removing outliers, and performing K-fold####
#out of SOP work!
#further experiment (split the data)
library(splitstackshape)
####Understanding the data####
#Names of the variables and variable type
names(Raw.data)
str(Raw.data)
#most are factors
#we believe that age, job, marital, education, default, balance, housing, loan, contact, duration

#data that has outliers, if we want to use these, we have to consider taking out the outliers, also high unknowns
table(Raw.data$pdays)
table(Raw.data$previous)
table(Raw.data$poutcome) #we know that if, Success, it will probably be a yes in y, we kinda know this is a strong predictor

#we assume that dates and month is not significant, but well, it maybe. (this is a hypothesis)
#Hypothesis 1: does age affect y?
ggplot(Train, aes(x = age, fill = y)) +
  geom_bar() +
  ggtitle("Age vs y") +
  xlab("Age") +
  ylab("Number of ys")
# Age poutcome with y
ggplot(Train, aes(x = age, fill = y)) +
  facet_wrap(~poutcome) +
  geom_bar() +
  ggtitle("Age vs y") +
  xlab("Age") +
  ylab("Number of ys")
#Hypothesis 2: Job affect y?
ggplot(Train, aes(x = job, fill = y)) +
  geom_bar() +
  ggtitle("Job vs y") +
  xlab("Job") +
  ylab("Number of ys")

#Hypothesis 3: Marital affect y?
ggplot(Train, aes(x = marital, fill = y)) +
  geom_bar() +
  ggtitle("Marital vs y") +
  xlab("Marital") +
  ylab("Number of ys")

#Hypothesis 4: Education affect y?
ggplot(Train, aes(x = education, fill = y)) +
  geom_bar() +
  ggtitle("Education vs y") +
  xlab("Education") +
  ylab("Number of ys")

#spread in age and education to see if they have relationship
ggplot(Train, aes(x = age, fill = education ))+
  geom_bar()

ggplot(Test, aes(x = age, fill = education ))+
  geom_bar()

#to confirm what im seeing
Education.25.age <- length(which((Train$age == 25)&(Train$education == "secondary")))

#Hypothesis 5: does balance affect y?
ggplot(Train, aes(x = balance, fill = y)) +
  geom_bar() +
  ggtitle("Balance vs y") +
  xlab("Balance")+
  scale_x_continuous(limits = c(-1000, 5000))+
  scale_y_continuous(limits = c(0,100))
#save to say that, the higher the balance, the more ys?

#Hypothesis 6: does housing affect y?
ggplot(Train, aes(x = housing , fill = y)) +
  geom_bar() +
  ggtitle("Housing vs y") +
  xlab("Housing")
#when no housing loan, they would be more happy to make a y

#Hypothesis 7: does loan affect y?

ggplot(Train, aes(x = loan, fill = y)) +
  geom_bar() +
  ggtitle("loan vs y") +
  xlab("loan")
#when not on loan, they would be more happy to make a y
#I can combine the load and housing together to see whether or not it performs better

#Hypothesis 8: does contact affect y?
ggplot(Train, aes(x = contact, fill = y)) +
  geom_bar() +
  ggtitle("Contact vs y") +
  xlab("Contact")
#Cellular has high ys. Probably the personal touch?

#Hypothesis 9: does day affect y? date and month combined
ggplot(Train, aes(x = day , fill = y)) +
  facet_wrap(~month)+
  geom_bar() +
  ggtitle("Day vs y") +
  xlab("Day")

#Hypothesis 10: does month  affect y?
ggplot(Train, aes(x = month, fill = y)) +
  geom_bar() +
  ggtitle("Month vs y") +
  xlab("Month")

#Hypothesis 11: does duration affect y?
ggplot(Train, aes(x = duration, fill = y)) +
  geom_bar() +
  ggtitle("Duration vs y") +
  xlab("Duration")+
  scale_x_continuous(limits = c(0, 2000))

#Hypothesis 12: does campaign affect y?
ggplot(Train, aes(x = campaign , fill = y)) +
  geom_bar() +
  ggtitle("Campaign vs y") +
  xlab("Campaign")
#Persistance is not exactly key. above 10 Tries, they won't accept

#Hypothesis 13: does pdays affect y?
ggplot(Train, aes(x = pdays, fill = y)) +
  geom_bar() +
  ggtitle("Pdays vs y") +
  xlab("Pdays")
summary(Train$pdays)
#probably a good predictor. Not contacted people tend to deposit

#Hypothesis 14: does previous affect y?
ggplot(Train, aes(x = previous, fill = y)) +
  geom_bar() +
  ggtitle("Previous vs y") +
  xlab("Previous")
#probably a good predictor. 1 day

#Hypothesis 17: does outcome of poutcome affect y?
ggplot(Train, aes(x = poutcome, fill = y)) +
  geom_bar() +
  ggtitle("Previous Marketing vs y") +
  xlab("Previous Marketing")
#there is a high ratio for poutcome in success, BUT!, most of the marketing information is unknown, thus a bad predictor
rm(Education.25.age)
####stratified sampling 30% based on dependent variable y####
# we won't be using this however
library(caret)
library(ggplot2)
library(doParallel)

#create a list of seed, here change the seed for each resampling
set.seed(123)
train.index <- createDataPartition(Raw.data$y, p=0.7, list = F)
Train.2 <- Raw.data[train.index,]
Test.2 <- Raw.data[-train.index,]
rm(train.index)

####Outliers####
#https://www.youtube.com/watch?v=ckxEZDN1iok
#for numerical variables, outliers
Raw.data.3 <- Raw.data
str(Raw.data.3)
#age, balance, duration, pdays, campaign, previous
#Using package
library(outliers)
hist(Raw.data.3$age)
hist(Raw.data.3$balance)
hist(Raw.data.3$duration)
hist(Raw.data.3$pdays)
hist(Raw.data.3$campaign)
hist(Raw.data.3$previous)
#using subset, outliers are able to be removed

####P-days covert####
#feel that -1, is not justified in the system
pdays <- Raw.data$pdays
pdays[pdays==-1] <- 0
Raw.data.3 <- data.frame(Raw.data[,1:13],pdays,Raw.data[,15:17])
rm(pdays)
m = dim(Raw.data.3)[1]
set.seed(1)
val = sample(1:m,size=round(m*0.3),replace=FALSE,prob=rep(1/m,m))
Train.4 = Raw.data.3[-val,]
Test.4 = Raw.data.3[val,]

####K-fold cross validation pessimistic approach (DO NOT RUN IT Cancerous to your computer)####
#library(caret)
#library(doSNOW)

#set.seed(2348)
#cv.10.folds <- createMultiFolds(rf.label, k = 10, times = 10)

# Check stratification
#table(rf.label)
#342 / 549

#table(rf.label[cv.10.folds[[33]]])
#308 / 494


# Set up caret's trainControl object per above.
#ctrl.1 <- trainControl(method = "repeatedcv", number = 5, repeats = 5,
                       #index = cv.10.folds)


# Set up doSNOW package for multi-core training. (do note, on the core of your own computer)
#cl <- makeCluster(6, type = "SOCK") #be VERY careful with the cores
#registerDoSNOW(cl)


# Set seed for reproducibility and train
#set.seed(34324)
#rf.1.cv.1 <- train(x = rf.train.1, y = rf.label, method = "rf", tuneLength = 3,
                   #ntree = 500, trControl = ctrl.1)

#Shutdown cluster
#stopCluster(cl)

# Check out results
#rf.1.cv.1


# Let's try 5-fold CV repeated 10 times. 
#set.seed(5983)
#cv.5.folds <- createMultiFolds(rf.label, k = 5, times = 10)

#ctrl.2 <- trainControl(method = "repeatedcv", number = 5, repeats = 10,
                       #index = cv.5.folds)

#cl <- makeCluster(6, type = "SOCK")
#registerDoSNOW(cl)

#set.seed(89472)
#rf.1.cv.2 <- train(x = rf.train.1, y = rf.label, method = "rf", tuneLength = 3,
                   #ntree = 1000, trControl = ctrl.2)

#Shutdown cluster
#stopCluster(cl)

# Check out results
#rf.1.cv.2

#TRY AT YOUR OWN RISK, MY PC DIED AFTER THIS CODE



