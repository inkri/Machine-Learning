#Loan Prediction
##nAME   : Abhishek Jaiswal
##Mail ID: inkrijaiswal@gmail.com 


#Step 1: Loading data set.
setwd("C:/Users/abhishek.b.jaiswal/Desktop/DataScience/Kaggle_github/Loan_Prediction")
HousingFinance=read.csv('train.csv',header=T,na.strings=c(""))
head(HousingFinance)

#Step 2: Checking Datasets data and structure.
summary(HousingFinance)
class(HousingFinance)
dim(HousingFinance)
str(HousingFinance)

#Step 3: Data cleaning:
#A:Outlier Treatment:
HousingFinance2=HousingFinance
boxplot(HousingFinance2)
hist(HousingFinance2$ApplicantIncome)
boxplot(HousingFinance2$ApplicantIncome,horizontal=TRUE)$stats[c(1,2,3,4,5), ]
boxplot.stats(HousingFinance2$ApplicantIncome)
boxplot(HousingFinance2$CoapplicantIncome)
boxplot(HousingFinance2$CoapplicantIncome,horizontal=TRUE)$stats[c(1,2,3,4,5), ]
boxplot.stats(HousingFinance2$CoapplicantIncome)
summary(HousingFinance2$ApplicantIncome)
summary(HousingFinance2$CoapplicantIncome)
bench1=5795+1.5*IQR(HousingFinance2$ApplicantIncome)
bench2=2297+1.5*IQR(HousingFinance2$CoapplicantIncome)
#ApplicantIncome COLUMN
HousingFinance2$ApplicantIncome[HousingFinance2$ApplicantIncome > bench1] =bench1
summary(HousingFinance2$ApplicantIncome)
boxplot(HousingFinance2$ApplicantIncome)
#CoapplicantIncome COLUMN
HousingFinance2$CoapplicantIncome[HousingFinance2$CoapplicantIncome > bench2] =bench2
summary(HousingFinance2$CoapplicantIncome)
boxplot(HousingFinance2$CoapplicantIncome)

#B:Missing values:
#Count of missing values
sapply(HousingFinance2,function(x) sum(is.na(x)))
#By visual missing values
library(Amelia)
missmap(HousingFinance2, main = "Missing values vs observed")
library(mice)
library(VIM)
head(HousingFinance2)
md.pattern(HousingFinance2)
p=md.pairs(HousingFinance2)
p
pbox(HousingFinance,pos=1,int=FALSE,cex=0.7)
# Visual plot of missing data
mice_plot=aggr(HousingFinance2, col=c('navyblue','yellow'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(HousingFinance2), cex.axis=.7,
                  gap=3, ylab=c("Missing data","Pattern"))


##Missing values Treatment by kNN Imputation (categorical/continuous)
HousingFinancekNN=HousingFinance2
HousingFinancekNN=kNN(HousingFinancekNN)
summary(HousingFinancekNN)
HousingFinancekNN=subset(HousingFinancekNN,select=Loan_ID:Loan_Status)
sapply(HousingFinancekNN,function(x) sum(is.na(x)))




#Step 4: Model creation glm() function 
HousingFinancekNNbkp=HousingFinancekNN
names(HousingFinancekNN)
HousingFinancekNN$Loan_ID=NULL
HousingFinancekNN$Loan_Status=ifelse(HousingFinancekNN$Loan_Status=="Y",1,0)
model=glm(Loan_Status ~.,family=binomial(link='logit'),data=HousingFinancekNN)
summary(model)
nothing=glm(Loan_Status ~ 1,family=binomial,data=HousingFinancekNN)
summary(nothing)

#summary(model3)
##Model select by AIC and BIC:
#AIC(model,model2,model3)
library(MASS)
#Backward AIC
backwards=step(model)
summary(backwards)
#Forward AIC
forwards = step(nothing,scope=list(lower=formula(nothing),upper=formula(model)), direction="forward")
summary(forwards)
#Both AIC
bothways=step(nothing, list(lower=formula(nothing),upper=formula(model)),direction="both",trace=0)
summary(bothways)
finalmodel=glm(Loan_Status ~Married+Credit_History+Property_Area,family=binomial(link='logit'),data=HousingFinancekNN)
summary(finalmodel)
#finalmodel AIC: 580.79
HousingFinancekNN$Loan_ID=HousingFinancekNNbkp$Loan_ID
HousingFinancekNN$Loan_Status=HousingFinancekNNbkp$Loan_Status
HousingFinancekNN$Loan_Status=ifelse(HousingFinancekNN$Loan_Status=="Y",1,0)

#Step 5:Checking for Multicollinearity and Variable Importance
library(usdm)
HousingFinancekNNS=HousingFinancekNN[,-c(1,2,3,4,5,6,12,13)]

##Goodness of Fit
#1:Likelihood Ratio Test
#anova(model, model2, test ="Chisq")
#library(lmtest)
#lrtest(model, model2)

#2:Pseudo R^2
library(pscl)
pR2(finalmodel)

#3:Hosmer-Lemeshow Test
#library(MKmisc)
#HLgof.test(fit = fitted(finalmodel), obs = HousingFinancekNN$Loan_Status)
#library(ResourceSelection)
#hoslem.test(HousingFinancekNN$Loan_Status, fitted(finalmodel), g=10)


##Statistical Tests for Individual Predictors
#1:Variable Importance
library(caret)
varImp(finalmodel)

#2:Wald Test
library(survey)
names(HousingFinancekNN)
attach(HousingFinancekNN)
regTermTest(finalmodel, "Gender")
regTermTest(finalmodel, "ApplicantIncome")
regTermTest(finalmodel, "Credit_History")


#Step 6:Prediction and Validation.
pred=predict(finalmodel,newdata=HousingFinancekNN[,-12],type='response')
pred1=ifelse(pred < 0.5,0,1)
library(caret)
confusionMatrix(table(pred1,HousingFinancekNN$Loan_Status,dnn=list('predicted', 'actual')))

##Validation of Predicted Values
#1:Classification Rate
library(caret)
confusionMatrix(table(pred1,HousingFinancekNN$Loan_Status ,dnn=list('predicted', 'actual')))

#2:ROC Curve
library(InformationValue)
plotROC(actuals=HousingFinancekNN$Loan_Status,predictedScores=as.numeric(fitted(finalmodel)))



#########################################################################
#TESTING with New Data:
#SLoading data set.
setwd("C:/Users/abhishek.b.jaiswal/Desktop/DataScience/Kaggle_github/Loan_Prediction")
HousingFinanceTEST=read.csv('test.csv',header=T,na.strings=c(""))
head(HousingFinanceTEST)

#Data Cleaning
#Outliers
summary(HousingFinanceTEST$ApplicantIncome)
summary(HousingFinanceTEST$CoapplicantIncome)
bench3=5795+1.5*IQR(HousingFinanceTEST$ApplicantIncome)
bench4=2297+1.5*IQR(HousingFinanceTEST$CoapplicantIncome)
#ApplicantIncome COLUMN
HousingFinanceTEST$ApplicantIncome[HousingFinanceTEST$ApplicantIncome > bench3] =bench3
summary(HousingFinanceTEST$ApplicantIncome)
boxplot(HousingFinanceTEST$ApplicantIncome)
#CoapplicantIncome COLUMN
HousingFinanceTEST$CoapplicantIncome[HousingFinanceTEST$CoapplicantIncome > bench2] =bench2
summary(HousingFinanceTEST$CoapplicantIncome)
boxplot(HousingFinanceTEST$CoapplicantIncome)
#Missing values
HousingFinanceTEST=kNN(HousingFinanceTEST)
summary(HousingFinanceTEST)
HousingFinanceTEST=subset(HousingFinanceTEST,select=Loan_ID:Property_Area)
sapply(HousingFinanceTEST,function(x) sum(is.na(x)))


#Prediction of TEST Data
predTEST=predict(finalmodel,newdata=HousingFinanceTEST,type='response')
predTEST1=ifelse(predTEST < 0.5,0,1)
HousingFinanceTEST$Loan_Status=predTEST1
HousingFinanceTEST$Loan_Status=ifelse(HousingFinanceTEST$Loan_Status==1,"Y","N")

HousingFinanceTESTAV=HousingFinanceTEST
HousingFinanceTESTAV$Property_Area=NULL
head(HousingFinanceTESTAV)
write.csv(HousingFinanceTESTAV,"Sample_Submission.csv")
