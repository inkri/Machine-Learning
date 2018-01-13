#https://www.r-bloggers.com/logistic-regression-with-r/
#https://www.r-bloggers.com/how-to-perform-a-logistic-regression-in-r/
#https://www.r-bloggers.com/evaluating-logistic-regression-models/

#Data cleaninh
##https://www.r-bloggers.com/missing-value-treatment/

#Continuous VS categorical variables.
#General LINEAR REGRESSION model:
# y=b0+b1x1+b2x2+e
##Independent variables(X's)
#Continuous: age/income --> use numerical values
#Categorical: gender/city --> use dummies
##Dependent variables(Y)
#Continuous: consumption/time spent --> use numerical value
#Categorical: yes/no --> use dummies

#Probility: 0<=p<=1
#For p>=0 : p=exp(b0+b1x1) = e(b0+b1x1)
#For p<=1 : p=(exp(b0+b1x1)/exp(b0+b1x1)+1) = e(b0+b1x1)/e(b0+b1x1)+1
#Algebra,Same as above: ln(p/1-p)=b0+b1x1


#About Logistic Regression:
#Dependent  variable(Y) is dichotomous (binary) or multinomial. 
#Independent variables(Xs)can be either continuous or categorical.


#Packages required to work on it.
#caret / aod / ggplot2



##CASE STUDY:
#R makes it very easy to fit a logistic regression model. glm() FUNCTION.
#Data set: Titanic dataset

#Step 1: Loading data set.
setwd("C:/Users/abhishek.b.jaiswal/Desktop/DataScience/sem1/3 BA/LogisticRegression")
titanic.raw=read.csv('titanic.csv',header=T,na.strings=c(""))
head(titanic.raw)

#Step 2: Checking Datasets data and structure.
summary(titanic.raw)
class(titanic.raw)
dim(titanic.raw)
str(titanic.raw)


#Step 3: Data cleaning.
##Missing values:
#By numerical
sapply(titanic.raw,function(x) sum(is.na(x)))
#sum(is.na(titanic.raw))
#!complete.cases(titanic.raw)
#https://www.r-bloggers.com/missing-value-treatment/
#By visual
library(Amelia)
missmap(titanic.raw, main = "Missing values vs observed")
#The variable cabin has too many missing values, we will not use it. 
#We will also drop PassengerId since it is only an index and Ticket.
data=subset(titanic.raw,select=c(2,3,5,6,7,8,10,12))
sapply(data,function(x) sum(is.na(x)))
missmap(data, main = "Missing values vs observed")
#There are different ways to do this, a typical approach is to replace the missing values with the average, the median or the mode of the existing one.
data$Age[is.na(data$Age)]=mean(data$Age,na.rm=T)
#As for the missing values in Embarked, since there are only two,replace with mode
data=data[!is.na(data$Embarked),]
rownames(data)=NULL


##Unique values:
sapply(titanic.raw, function(x) length(unique(x)))
sapply(data, function(x) length(unique(x)))
#As far as categorical variables are concerned, using the read.table() or read.csv() by default will encode the categorical variables as factors.
#A factor is how R deals categorical variables.
is.factor(data$Sex)
is.factor(data$Embarked)

#How the variables have been dummyfied by R and how to interpret them in a model.
#contrasts() function: how R is going to deal with the categorical variables
contrasts(data$Sex)
contrasts(data$Embarked)
#Two-way table of factor variables.
names(data)
xtabs(~Survived+Pclass,data=data)


##Data outliers:
#Fare:
hist(data$Fare)
boxplot(data$Fare,horizontal=TRUE)$stats[c(1,2,3,4,5), ]
boxplot.stats(data$Fare)
#Age:
hist(data$Age)
boxplot(data$Age,horizontal=TRUE)$stats[c(1,2,3,4,5), ]
boxplot.stats(data$Age)

#Step 4: Train and Test datasets
set.seed(1234)
library(caret)
inTrain=createDataPartition(data$Survived, p=0.7, list=FALSE)
train=data[inTrain,]
test=data[-inTrain,]

#Step 5: Model creation glm() function 
names(data)
model=glm(Survived ~.,family=binomial(link='logit'),data=train)
summary(model)
model2=update(model, .~. -Parch-Fare+Age-Embarked,family=binomial(link='logit'),data=train)
summary(model2)
names(data)

#Step 6:Prediction
newdata=subset(test,select=c(2,3,4,5,6,7,8))
fitted.results=predict(model,newdata,type='response')
pred1=ifelse(fitted.results > 0.5,1,0)

confusionMatrix(table(pred1,test$Survived ,dnn=list('predicted', 'actual')))



#Step 7:Model Evaluation and Diagnostics

##A: Goodness of Fit:
#1:Likelihood Ratio Test
#To check improvement in a full(Xs) model with fewer(Xs) model.
#lrtest() function from the lmtest package or using the anova() function in base.
anova(model, model2, test ="Chisq")
library(lmtest)
lrtest(model, model2)
#Reject H0.(Removing Xs does not make any progress)


#2:Pseudo R^2
#The measure ranges from 0 to just under 1, 
#with values closer to zero indicating that the model has no predictive power.
library(pscl)
# look for 'McFadden'
pR2(model)


#3:Hosmer-Lemeshow Test
#Small values with large p-values indicate a good fit to the data while large values with p-values below 0.05 indicate a poor fit.
library(MKmisc)
HLgof.test(fit = fitted(model), obs = train$Survived)
library(ResourceSelection)
hoslem.test(train$Survived, fitted(model), g=10)



##B:Statistical Tests for Individual Predictors
#1:Wald Test
#To evaluate the statistical significance of each coefficient in the model
library(survey)
regTermTest(model, "ForeignWorker")
regTermTest(model, "CreditHistory.Critical")


#2:Variable Importance
varImp(model)


##C:Validation of Predicted Values
#1:Classification Rate
pred = predict(model, newdata=test)
pred=ifelse(pred > 0.5,1,0)
accuracy <- table(pred, test[,"Survived"])
sum(diag(accuracy))/sum(accuracy)
confusionMatrix(table(pred,test$Survived ,dnn=list('predicted', 'actual')))


#2:ROC Curve
#That metric ranges from 0.50 to 1.00, and values above 0.80 indicate that the model does a good job in discriminating between the two categories which comprise our target variable.
library(InformationValue)
plotROC(actuals=train$Survived,predictedScores=as.numeric(fitted(model)))

#3:K-Fold Cross Validation
#The most common variation of cross validation is 10-fold cross-validation.
ctrl=trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)
model11=train(Survived ~.,method="glm",family=binomial(link='logit'),data=train,trControl = ctrl, tuneLength = 5)
pred = predict(model11, newdata=test)
pred=ifelse(pred > 0.5,1,0)
confusionMatrix(pred, test$Survived)

