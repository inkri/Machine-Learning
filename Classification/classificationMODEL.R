##Step 1: Data Load:
setwd("C:/Users/xxxxxxx/Desktop/HACK17")
cars=read.table("cars.txt",header=F, sep=",")
#Column Renaming.
cars$buying=cars$V1
cars$V1=NULL
cars$maint=cars$V2
cars$V2=NULL
cars$doors=cars$V3
cars$V3=NULL
cars$persons=cars$V4
cars$V4=NULL
cars$lug_boot=cars$V5
cars$V5=NULL
cars$safety=cars$V6
cars$V6=NULL
cars$class=cars$V7
cars$V7=NULL
summary(cars)
str(cars)
levels(cars$class)
summary(cars$buying)
summary(cars$maint)
summary(cars$class)
head(cars)
tail(cars)
class(cars)
levels(cars$persons)



#Variable Importance in dataset
library(randomForest)
fit=randomForest(class~., data=cars)
(VI_F=importance(fit))
library(caret)
varImp(fit)
varImpPlot(fit,type=2)

#############################Naive Bayes Classifier#########################################
library("psych")
library("e1071")
library(caret)
carss=cars
table(carss$class)
str(carss)
#Variable dependency:
pairs.panels(carss)
#Training and Test Datasets.
set.seed(1234)
inTrain=createDataPartition(carss$class, p=0.7, list=FALSE)
Training=carss[inTrain,]
Testing=carss[-inTrain,]
#Model
class1=naiveBayes(class~., data=Training)
class1
preds1=predict(class1,Testing[,-7])
preds1
table(preds1,Testing$class)
round(sum(preds1==Testing$class,na.rm=TRUE)/length(Testing$class),digits=2)
confusionMatrix(preds1,Testing$class)
#Accuracy : 0.8511

############################k-NN Classifier#################################################
carskn=cars
library(car)
head(carskn)
#Data cleaning:
#Var1:"high:3"  "low:1"   "med:2"   "vhigh:4"
levels(carskn$buying)
carskn$buying=recode(carskn$buying,"'low'=1")
carskn$buying=recode(carskn$buying,"'med'=2")
carskn$buying=recode(carskn$buying,"'high'=3")
carskn$buying=recode(carskn$buying,"'vhigh'=4")
carskn$buying=as.numeric(as.character(carskn$buying))
#Var2:"high:3"  "low:1"   "med:2"   "vhigh:4"
levels(carskn$maint)
carskn$maint=recode(carskn$maint,"'low'=1")
carskn$maint=recode(carskn$maint,"'med'=2")
carskn$maint=recode(carskn$maint,"'high'=3")
carskn$maint=recode(carskn$maint,"'vhigh'=4")
carskn$maint=as.numeric(as.character(carskn$maint))
#Var3:"5more:5"
levels(carskn$doors)    
carskn$doors=recode(carskn$doors,"'5more'=5")
carskn$doors=as.numeric(as.character(carskn$doors))
#Var4:"more:5"
levels(carskn$persons)  
carskn$persons=recode(carskn$persons,"'more'=5")
carskn$persons=as.numeric(as.character(carskn$persons))
#Var5:"big:3"   "med:2"   "small:1"
levels(carskn$lug_boot)
carskn$lug_boot=recode(carskn$lug_boot,"'small'=1")
carskn$lug_boot=recode(carskn$lug_boot,"'med'=2")
carskn$lug_boot=recode(carskn$lug_boot,"'big'=3")
carskn$lug_boot=as.numeric(as.character(carskn$lug_boot))
#Var6:"high:3" "low:1"  "med:2"
levels(carskn$safety)
carskn$safety=recode(carskn$safety,"'low'=1")
carskn$safety=recode(carskn$safety,"'med'=2")
carskn$safety=recode(carskn$safety,"'high'=3")
as.numeric(carskn$safety)
carskn$safety=as.numeric(as.character(carskn$safety))
head(carskn)
str(carskn)
levels(cars$class)
library(class)
table(carskn$class)
head(carskn)
set.seed(9852)
#runif(5)
#gp=runif(nrow(carskn))
#carskn=carskn[order(gp),]
str(carskn)
head(carskn)
summary(carskn[,c(1:6)])
#Training and Test Datasets.
#inTrain=createDataPartition(carskn$class, p=0.7, list=FALSE)
#Training=carskn[inTrain,]
#Training_target=carskn[inTrain,7]
#Training_target=data.frame(Training_target)
#Testing=carskn[-inTrain,]
#Testing_target=carskn[-inTrain,7]
#Testing_target=data.frame(Testing_target)

ind=sample(2,nrow(carskn),replace=TRUE,prob=c(0.8,0.2))
training_data=carskn[ind==1,]
testing_data=carskn[ind==2,]

#Model
k=sqrt(1728) 
k=41   #41.56922
#m1=knn(train=training_data, test=testing_data, cl=training_data[,7], k=41)

m2 <- knn(
  subset(training_data, select = -class), 
  subset(testing_data, select = -class),
  factor(training_data$class),
  k = 41, prob=TRUE, use.all = TRUE)
m2
levels(m2)
attributes(m2)
summary(m2)
table(testing_data$class,m2)
mean(m2==testing_data$class)
confusionMatrix(table(m2, testing_data$class))
#Accuracy : 0.9011

###############################Decision Tree############################################
carsDT=cars
library(ISLR)
library(tree)
library(caret)
head(carsDT)
#Split the dataset
set.seed(1234)
train=sample(1:nrow(carsDT),nrow(carsDT)/2)
test=-train
training_data=carsDT[train,]
testing_data=carsDT[test,]
#Model
mod1=tree(class~.,training_data)
library(tree)
fit=tree(class~.,training_data,subset=train)
summary(fit)
summary(mod1)
mod1
plot(mod1)
text(mod1,pretty=0)
library(caret)
install.packages("rlang")
library(rlang)
library(ggplot2)
varImp(mod1)
#Checking model
pred1=predict(mod1,testing_data,type="class")
mean(pred1 == testing_data[,7])
mean(pred1 != testing_data[,7])  #8.1%
table(testing_data$class,pred1)
#####Cross validation to check where to stop pruning
set.seed(2)
cvtree=cv.tree(mod1,FUN=prune.misclass)
names(cvtree)
summary(cvtree)
cvtree
plot(cvtree$size,cvtree$dev,type='b') #Size 14
#Pruning tree
prumod1=prune.misclass(mod1,best=14)
#prumod2=prune.misclass(mod1,best=9)
summary(prumod1)
#summary(prumod2)
plot(prumod1)
text(prumod1,pretty=0)
#Check how prune model is doing
pred2=predict(prumod1,testing_data,type="class")
mean(pred2 == testing_data[,7])
mean(pred2 != testing_data[,7])   #7.8%
table(pred2,testing_data[,7])
install.packages("caret")
install.packages("Rcpp")
library(caret)
library(Rcpp)
confusionMatrix(testing_data$class, pred2)
#Accuracy : 0.9213

############################DT:CART->rpart############################################################
install.packages("rpart")
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)
#Datasets
carsRP=cars
set.seed(1243)
#Model
rmod=rpart(class~.,data=carsRP[1:1000,], method="class")
summary(rmod)
#Plot
rpart.plot(rmod)
rpart.plot(rmod,type=3,extra=101,fallen.leaves=T)
#Prediction
pred1=predict(rmod,carsRP[1001:1728,], type="class")
table(carsRP[1001:1728,7], pred1)
library(caret)
confusionMatrix(table(carsRP[1001:1728,7], pred1,dnn=list("Actual","Prediction")))
mean(pred1 == carsRP[1001:1728,7])
mean(pred1 != carsRP[1001:1728,7])   #29.9%
#Accuracy : 0.7005

###############################DT:C50######################################################
install.packages("C50")
library(C50)
#Dataset
carsC50=cars
ind=sample(2,nrow(carsC50),replace=TRUE,prob=c(0.7,0.3))
training_data=carsC50[ind==1,]
testing_data=carsC50[ind==2,]
table(carsC50$class)
head(carsC50)
str(carsC50)
set.seed(2143)
#Model
m1=C5.0(training_data[,-7],training_data[,7])
m1
summary(m1)
#Predition
p1=predict(m1,testing_data)
p1
summary(p1)
#Plot
plot(m1)
table(testing_data[,7], p1)
library(caret)
confusionMatrix(table(testing_data[,7], p1,dnn=list("Actual","Prediction")))
mean(p1 == testing_data[,7])
mean(p1 != testing_data[,7])   #4.1%
#Accuracy : 0.9583

##########################DT:Random Forest##################################################
carsRF=cars
str(carsRF)
summary(carsRF)
#Data Partition
set.seed(3321)
ind=sample(2,nrow(carsRF),replace=TRUE,prob=c(0.7,0.3))
training_data=carsRF[ind==1,]
testing_data=carsRF[ind==2,]
#RF Model
library(randomForest)
set.seed(222)
rmod1=randomForest(class~.,data=training_data)
rmod1
summary(rmod1)
attributes(rmod1)
#Prediction and Confusion matrix on Test Dataset
rfp1=predict(rmod1,testing_data)
rfp1
head(rfp1)
library(caret)
confusionMatrix(testing_data$class,rfp1, dnn=list("Actual","Predition"))
#Error rate of RF
plot(rmod1)
#Tune RF Model, mtry
t1=tuneRF(training_data[,-7],training_data[,7],stepFactor=0.5,plot=TRUE,ntreeTry=300,trace=TRUE,improve=0.05)
#mtry=5 at stepFactor=0.7
#mtry=8 at stepFactor=0.5
rmod2=randomForest(class~.,data=training_data,ntree=300,mtry=8,importance=TRUE,proximity=TRUE)
rmod2  
#OOB estimate of  error rate: 1.39%
#Train data
p1=predict(rmod2,training_data)
p1
confusionMatrix(training_data$class,p1, dnn=list("Actual","Predition"))
#Test Data
p2=predict(rmod2,testing_data)
p2
confusionMatrix(testing_data$class,p2, dnn=list("Actual","Predition"))
#Number of nodes for Trees
hist(treesize(rmod2),main="No. of Nodes",col="blue")
#Variable importance 
varImpPlot(rmod2)
importance(rmod2)
varUsed(rmod2)
#Partial Dependence Plot
names(training_data)
summary(training_data$class)
partialPlot(rmod2,training_data,buying,"acc")
partialPlot(rmod2,training_data,buying,"good")
partialPlot(rmod2,training_data,buying,"unacc")
partialPlot(rmod2,training_data,buying,"vgood")
#Extract single tree from forest
getTree(rmod2,1,labelVar=TRUE)  #Ist Tree
#Multi-Dimensional Scaling Plot of Proximity Matrix
MDSplot(rmod2,training_data$class)
##Predict for a sample.
head(cars)
tail(cars)
tdata=cars[1728,]
tdata$persons=4
tdata$safety="high"
tdata$doors=2
tdata
str(tdata)
str(training_data)
tdata$class=NULL
#Updating data type
tdata$persons=as.factor(as.character(tdata$persons))
tdata$safety=as.factor(as.character(tdata$safety))
tdata$doors=as.factor(as.character(tdata$doors))
sapply(tdata, class)
sapply(training_data, class)
#Adding levels:
levels(tdata$persons)=c(levels(tdata$persons),"2")
levels(tdata$persons)=c(levels(tdata$persons),"more")
levels(tdata$safety)=c(levels(tdata$safety),"med")
levels(tdata$safety)=c(levels(tdata$safety),"low")
levels(tdata$doors)=c(levels(tdata$doors),"3")
levels(tdata$doors)=c(levels(tdata$doors),"4")
levels(tdata$doors)=c(levels(tdata$doors),"5more")
tdata
#Prediction for tdata
ptest=predict(rmod2,tdata)
ptest

