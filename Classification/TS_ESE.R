setwd("C:/Users/abhishek.b.jaiswal/Desktop/DataScience/Kaggle_github/Electrolysia supplies electricity")
EC=read.csv("train.csv")
summary(EC)
names(EC)

#Data conversion for datetime:
bk=EC
#Year
bk$year = substr(bk$datetime, 1,4)
#Month
bk$month = substr(bk$datetime,6,7)
#date
bk$date = substr(bk$datetime,9,10)
#Hour
bk$hr = substr(bk$datetime,12,13)

str(bk)
bk$year=as.numeric(bk$year)
bk$month=as.numeric(bk$month)
bk$date=as.numeric(bk$date)
bk$hr=as.numeric(bk$hr)

bk$ID=NULL
bk$datetime=NULL
bk$ECR=bk$electricity_consumption
bk$electricity_consumption=NULL
EC=bk

#Variable Importance in dataset
library(randomForest)
fit=randomForest(ECR~., data=EC)
(VI_F=importance(fit))
library(caret)
varImp(fit)
varImpPlot(fit,type=2)

#Data Partition
set.seed(3321)
ind=sample(2,nrow(EC),replace=TRUE,prob=c(0.7,0.3))
training_data=EC[ind==1,]
testing_data=EC[ind==2,]

#RF Model
library(randomForest)
set.seed(222)
rmod1=randomForest(ECR~.,ntree=400,data=training_data)
rmod1  #ntree=500 and mtry=3
summary(rmod1)
attributes(rmod1)
?randomForest

#Prediction and Confusion matrix on Test Dataset
rfp1=predict(rmod1,testing_data)
rfp1
head(rfp1)
head(testing_data$ECR)

testing_data$Predicted=rfp1
testing_data$resdiual=(testing_data$ECR-testing_data$Predicted)
hist(testing_data$resdiual)

sqrt(3960.656)
testing_data$Predicted=NULL
testing_data$resdiual=NULL

#Error rate of RF
plot(rmod1)
#Tune RF Model, mtry
t1=tuneRF(training_data[,-10],training_data[,10],stepFactor=0.5,plot=TRUE,ntreeTry=300,trace=TRUE,improve=0.05)
#mtry = 1 	OOB error = 6374.833
#mtry = 3  OOB error = 3996.51
#mtry = 6 	OOB error = 3605.922
#mtry = 12 	OOB error = 3589.506
#New Model:
rmod2=randomForest(ECR~.,data=training_data,ntree=200,mtry=9,importance=TRUE,proximity=TRUE)
rmod2 
#ntree=200,mtry=6 :
#% Var explained: 69.23 
#Mean of squared residuals: 3616.556
#ntree=200,mtry=9:
#Mean of squared residuals: 3592.119
#% Var explained: 69.22

#Preditions:
p1=predict(rmod2,training_data)
p1
ptest=predict(rmod2,testing_data)
ptest
head(ptest)
head(testing_data$ECR)

#Number of nodes for Trees
hist(treesize(rmod2),main="No. of Nodes",col="blue")

#Variable importance 
varImpPlot(rmod2)
importance(rmod2)
varUsed(rmod2)

##################################################################################
################################################################################
setwd("C:/Users/abhishek.b.jaiswal/Desktop/DataScience/Kaggle_github/Electrolysia supplies electricity")
ECtest=read.csv("test.csv")
summary(ECtest)
names(ECtest)

#Data conversion for datetime:
bktest=ECtest
#Year
bktest$year = substr(bktest$datetime, 1,4)
#Month
bktest$month = substr(bktest$datetime,6,7)
#date
bktest$date = substr(bktest$datetime,9,10)
#Hour
bktest$hr = substr(bktest$datetime,12,13)

str(bktest)
bktest$year=as.numeric(bktest$year)
bktest$month=as.numeric(bktest$month)
bktest$date=as.numeric(bktest$date)
bktest$hr=as.numeric(bktest$hr)

bktest$ID=NULL
bktest$datetime=NULL
ECtest=bktest


#Prediction:
pt=predict(rmod2,ECtest[,-1])
pt
head(pt)
ECtest$electricity_consumption=pt
ECtest$hr=NULL

write.csv(ECtest,"Sample_Submission.csv")

##Abhishek Jaiswal
##inkrijaiswal@gmail.com