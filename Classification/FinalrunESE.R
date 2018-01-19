
##Step 1: Loading libraries:
library(usdm) 
library(mice) 
library(VIM) 
library(lattice) 
library(corrplot) 
library(PerformanceAnalytics) 
library(moments) 
library(car) 
library(Metrics) 
library(randomForest)

##Step 2: Load Data:
setwd("C:/Users/abhishek.b.jaiswal/Desktop/DataScience/Kaggle_github/Electrolysia supplies electricity")
ECtrain=read.csv("train.csv")
ECtest=read.csv("test.csv")
summary(ECtrain)
names(ECtrain)



##Step 3: Data cleaning:
names(ECtrain)
ECtest$electricity_consumption=1
ECtrain$dataset="train"
ECtest$dataset="test"
compl=rbind.data.frame(ECtrain,ECtest)


#Dividing Datetime column:
compl$year = substr(compl$datetime, 1,4)
compl$month = substr(compl$datetime, 6,7)
compl$date = substr(compl$datetime, 9,10)
compl$hour= substr(compl$datetime, 12,13)
compl$year=as.numeric(compl$year)
compl$month=as.numeric(compl$month)
compl$date=as.numeric(compl$date)
compl$hour=as.numeric(compl$hour)

str(compl)


#Dataset Rearranging:
names(compl)
compl=compl[,c(1,2,3,4,5,6,7,10,11,12,13,8,9)]

##Outlier checking:
boxplot(compl[,-c(1,2,7,8,9,10,11,12,13)])

##Outlier Treatment:
i=NA
j=c(3:6)

for (i in j){
  sd=sd(compl[,i])
  xbar=mean(compl[,i])
  uc=xbar + (3 * sd)
  lc=xbar - (3 * sd)
  rol=xbar + (6 * sd)
  lol=xbar - (6 * sd)
  compl=compl[!compl[,i] > rol,]
  compl=compl[!compl[,i] < lol,]
  compl[compl[,i] > uc,i]=uc
  compl[compl[,i] < lc,i]=lc
}

##Outlier checking:
boxplot(compl[,-c(1,2,7,8,9,10,11,12,13)])

#Separating Dataset:
ECtrain=compl[compl$data=="train",]
ECtest=compl[compl$data=="test",]
ECtrain$dataset=NULL
ECtest$dataset=NULL
ECtest$electricity_consumption=NULL
ECtrain=ECtrain[,-c(1,2)]


##Correlation:
corr=cor(ECtrain[-5], use = "complete.obs")
corrplot(corr, type = "upper", order = "original", 
         tl.col = "black", tl.srt = 45)

##Missing values:
sum(is.na(ECtrain))



##Step 4: Random Forest Model:
model=randomForest(electricity_consumption~., data = ECtrain)
model

#Variable Importance in dataset
(VI_F=importance(model))
library(caret)
varImp(model)
varImpPlot(model,type=2)

##Step 5:Prediction:
pred=predict(model, ECtrain)
rmse(ECtrain$electricity_consumption, pred)
plot(ECtrain$electricity_consumption,pred)

##Step 6:Tuning:
#Error rate of RF
plot(model)  #Tree:200
#Tune RF Model, mtry
t1=tuneRF(ECtrain[,-10],ECtrain[,10],stepFactor=0.5,plot=TRUE,ntreeTry=200,trace=TRUE,improve=0.05)
#mtry = 6 	OOB error = 3030.046
model2=randomForest(electricity_consumption~.,data=ECtrain,ntree=200,mtry=6,importance=TRUE,proximity=TRUE)
model2


#Prediction on ECtrain and ECtest Data by model2:
predtrain=predict(model2, ECtrain)
predtest=predict(model2, ECtest)
rmse(ECtrain$electricity_consumption, predtest)
plot(ECtrain$electricity_consumption,predtest)
rmse(ECtest$electricity_consumption, predtest)
plot(ECtest$electricity_consumption,predtest)


# Creating Submission File:
ntest$electricity_consumption=predict(model, ECtest)
datats=as.data.frame(cbind(ntest$ID,ntest$electricity_consumption))
names(datats)=c("ID","electricity_consumption")
write.csv(datats, "submission.csv", row.names = FALSE)