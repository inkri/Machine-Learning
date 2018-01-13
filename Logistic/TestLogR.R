#Step 1. Load data and run numerical and graphical summaries.
setwd("C:/Users/abhishek.b.jaiswal/Desktop/DataScience/NSE_EXAM")
titanic=read.csv("titanicdata.csv")
titanic=titanic[,-c(1)]
titanic=titanic[,-3]
head(titanic)
install.packages("caret")
library(caret)
library(aod)
library(ggplot2)

#Step 2. Split the data INTO Traingdata(80%) and Testingdata(20%)
ind=sample(2,nrow(titanic),replace=TRUE,prob=c(0.8,0.2))
training_data=titanic[ind==1,]
testing_data=titanic[ind==2,]
head(training_data)
head(testing_data)

#OR
inTrain=createDataPartition(titanic$Survived, p=0.7, list=FALSE)
training_data=titanic[inTrain,]
testing_data=titanic[-inTrain,]

summary(titanic)
#Step 3. Data cleanup, if any.
is.na(titanic)
mean(titanic$Age)
median(titanic$Age)
sd(titanic$Age)
titanic$SexDummy<-ifelse(titanic$Sex=="female",1,0)
table(titanic$Sex)
summary(titanic$SexCode)

#Checking for Multicollinearity
install.packages("usdm")
library(usdm)
vifstep(titanic[,-3], th=2)
cor(titanic)
class(titanic$SexCode)
titanic$SexCode=numeric(titanic$SexCode)
str(titanic)

#AIC value
library(MASS)
step=stepAIC(fit1, direction="both")


#Step 4. Model creation
fit1=glm(Survived ~ . ,family="binomial",data=titanic)
summary(fit1)
confint(model)
str(titanic)
fit2=glm(Survived ~ Age+SexCode ,family="binomial",data=titanic)
summary(fit2)
confint(model2)

#Step 5. Use the fitted model to do prediction for the test data.
pred=predict(fit1, data=titanic,type="response")
pred1=ifelse(pred<0.5,0,1)
cm(table(titanic$Survived,pred1,dnn=list('actual','predicted')))
confusionMatrix(table(titanic$Survived,pred1,dnn=list('actual','predicted')))

pred2=predict(fit2,data=titanic,type="response")

sensitivity(titanic$Survived, pred, threshold = optCutOff)
specificity(titanic$Survived, pred, threshold = optCutOff)

#Step 6. Goodness of Fit.
#1. Likelihood Ratio Test
anova(fit1, fit2, test ="Chisq")
library(lmtest)
lrtest(fit1, fit2)

#2. Pseudo R^2 or Mc Fadden R^2
library(pscl)
pR2(fit2)

#3. Hosmer-Lemeshow Test
install.packages("ResourceSelection")
library(ResourceSelection)
hoslem.test(titanic$Survived,fitted(fit1),g=10) 

#Step 7. Statistical Tests for Individual Predictors
#1. Wald Test
library(survey)
regTermTest(fit1, "Age")
regTermTest(fit1, "SexCode")
regTermTest(fit1, "PClass")
str(titanic)

#2. Variable Importance
library(caret)
varImp(fit1)

#Step 8. Validation of Predicted Values
#1. Classification Rate
pred=predict(fit1, data=titanic,type="response")
pred1=ifelse(pred<0.5,0,1)
confusionMatrix(table(titanic$Survived,pred1,dnn=list('actual','predicted')))

#2. ROC Curve
library(pROC)
library(InformationValue)
plotROC(actuals=titanic$Survived,predictedScores=as.numeric(fitted(fit1)))


#3. K-Fold Cross Validation
ks_plot(actuals=titanic$Survived,predictedScores=as.numeric(fitted(fit1)))


##https://www.r-bloggers.com/evaluating-logistic-regression-models/