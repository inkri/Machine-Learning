##Loading libraries:
library(usdm) # VIF
library(mice) # Missing Value Imputation
library(VIM) # Missing Data Plot
library(lattice) # Diagnosis of Imputed Data
library(corrplot) # Correlation Plot
library(PerformanceAnalytics) # Univariate and Bivariate Analysis
library(moments) # Skewness
library(car) #OutlierTest
library(Metrics) #rmse,etc
library(randomForest)

##Load Data:
setwd("C:/Users/abhishek.b.jaiswal/Desktop/DataScience/Kaggle_github/Electrolysia supplies electricity")
ECtrain=read.csv("train.csv")
ECtest=read.csv("test.csv")
summary(ECtrain)
names(ECtrain)



##Data conversion and cleaning:
names(ECtrain)
ECtest$electricity_consumption=0
ECtrain$data="train"
ECtest$data="test"
compl=rbind.data.frame(ECtrain,ECtest)



compl$year = substr(compl$datetime, 1,4)
compl$month = substr(compl$datetime, 6,7)
compl$date = substr(compl$datetime, 9,10)
compl$hour= substr(compl$datetime, 12,13)

compl$year=as.numeric(compl$year)
compl$month=as.numeric(compl$month)
compl$date=as.numeric(compl$date)
compl$hour=as.numeric(compl$hour)

str(compl)


#Data Rearrange
names(compl)
compl=compl[,c(1,2,3,4,5,6,7,10,11,12,13,8,9)]

ECtrain=compl[compl$data=="train",]
ECtest=compl[compl$data=="test",]
ECtrain$data=NULL
ECtest$data=NULL
ECtest$electricity_consumption=NULL
ECtrain=ECtrain[,-c(1,2)]

#Data Split in Train and Test
set.seed(524)
index=sample(1:nrow(ECtrain), round(0.8 * nrow(ECtrain)))
training=ECtrain[index,]
testing=ECtrain[-index,]



##Correlation:

corr=cor(ECtrain[-5], use = "complete.obs")
corrplot(corr, type = "upper", order = "original", 
         tl.col = "black", tl.srt = 45)

##Missing values:
sum(is.na(ECtrain))

##Outlier Treatment:
i=NA
j=c(1:4,6:10)

for (i in j){
  sd=sd(ECtrain[,i])
  xbar=mean(ECtrain[,i])
  uc=xbar + (3 * sd)
  lc=xbar - (3 * sd)
  rol=xbar + (6 * sd)
  lol=xbar - (6 * sd)
  ECtrain=ECtrain[!ECtrain[,i] > rol,]
  ECtrain=ECtrain[!ECtrain[,i] < lol,]
  ECtrain[ECtrain[,i] > uc,i]=uc
  ECtrain[ECtrain[,i] < lc,i]=lc
}


rm(i)
rm(j)
rm(lc)
rm(lol)
rm(rol)
rm(sd)
rm(uc)
rm(xbar)
rm(compl)

##Random Forest Model:
model=randomForest(electricity_consumption~., data = ECtrain)
model

#Variable Importance in dataset
(VI_F=importance(model))
library(caret)
varImp(model)
varImpPlot(model,type=2)

# Prediction on Training Data:
pred=predict(model, ECtrain)
rmse(ECtrain$electricity_consumption, pred)
plot(ECtrain$electricity_consumption,pred)

##Tuning:
#Error rate of RF
plot(model)  #Tree:200
#Tune RF Model, mtry
t1=tuneRF(ECtrain[,-10],ECtrain[,10],stepFactor=0.5,plot=TRUE,ntreeTry=200,trace=TRUE,improve=0.05)
#mtry = 3  OOB error = 2971.393
#mtry = 6 	OOB error = 2615.622
#mtry = 12 	OOB error = 2610.655
model2=randomForest(electricity_consumption~.,data=ECtrain,ntree=200,mtry=6,importance=TRUE,proximity=TRUE)
model2


#Prediction on ECtrain Data by model2:
predtest=predict(model2, testing)
rmse(testing$electricity_consumption, predtest)
plot(testing$electricity_consumption,predtest)


# Creating Submission File:
ntest$electricity_consumption=predict(model, ECtest)
datats=as.data.frame(cbind(ntest$ID,ntest$electricity_consumption))
names(datats)=c("ID","electricity_consumption")
write.csv(datats, "submission.csv", row.names = FALSE)