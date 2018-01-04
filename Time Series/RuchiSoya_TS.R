#https://www.r-bloggers.com/time-series-analysis-in-r-part-1-the-time-series-object/
#https://www.r-bloggers.com/time-series-analysis-in-r-part-2-time-series-transformations/

#time series analysis and forecasting
#Load Data
setwd("C:/Users/abhishek.b.jaiswal/Desktop/DataScience/Kaggle_github/RuchiSoya_TimeSeries")
rsstock=read.csv("NSE-RUCHISOYA.csv")
rsstock=rsstock[rev(rownames(rsstock)),]

#tseries=ts(t, start = c(2000, 1), frequency = 1)
#print(tseries)

#Check the data.
head(rsstock)
str(rsstock)
names(rsstock)
summary(rsstock)

#Load Packages.
library(tseries)
library(forecast)
library(ggplot2)
attach(rsstock)
rsstock$High=NULL
rsstock$Low=NULL
rsstock$Last=NULL
rsstock$Close=NULL
rsstock$Total.Trade.Quantity=NULL
rsstock$Turnover..Lacs.=NULL

#Data Exploration
plot(Open)         #Non stationary data
plot.ts(Open)

#Step 1: Model Identification.
#Stationary Check-Dicky-Fuller Test
adf.test(Open,alternative ="stationary")

#p-value = 0.5026
#P Value > 0.05 Hence the data is non-stationary
d.Open=diff(Open)
d.Open
summary(Open)
summary(d.Open)
plot(d.Open)
adf.test(d.Open,alternative ="stationary")

#p-value = 0.01
#P Value < 0.05 Hence the data is stationary now.


acf(d.Open)      #AR(0)
pacf(d.Open)     #MA(1)
 

#Step 2: Model Estimation.
#Manual
arima(d.Open, order=c(0,0,1)) #or arima(Open, order=c(0,1,1)) #aic = 2054.96
arima(d.Open, order=c(1,0,1)) #or arima(Open, order=c(1,1,1)) #aic = 2051.75
arima(d.Open, order=c(2,0,1)) #or arima(Open, order=c(2,1,1)) #aic = 2053.73
#Auto ARMA FUNCTION, FOR GRNERATING BEST FIT MODEL.
auto.arima(Open)  #ARIMA(1,1,1) #AIC=2050.61

#Step 3: Diagnosis.
arima.final=arima(Open, order=c(1,1,1))
#Choose the one that has least AIC and significant co-efficients
tsdiag(arima.final)

#Forecasting using final model
arima.final=arima(Open, order=c(1,1,1))
predicted=predict(arima.final,n.ahead=5)
predicted
