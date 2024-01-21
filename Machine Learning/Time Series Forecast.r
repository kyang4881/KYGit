library(readr)
require(forecast)
require(tseries)

table <- read_csv("C:/Users/K/Downloads/table.csv")

returns <- (table$Close - table$Open)/table$Open

plot(returns, xlab = "Time (Months)", ylab = "Monthly Excess Return",
     main="Dow Jones Industrial Average Monthly Excess Return (January 2005 - Febrary 2017)",
     cex.main=0.8, type = "l")

acf(returns, main ="ACF of Time Series")

pacf(returns, main ="PACF of Time Series")

model <- arima(returns, order=c(2,0,2), method="ML"); model

ts.plot(model$residuals,main="Residual Plot",gpars=list(xlab="Time (Months)",ylab="Residuals"))

abline(h=0)

acf(model$residuals, main="ACF Plot of Residuals")

pacf(model$residuals, main="PACF Plot of Residuals")

qqnorm(model$residuals)

predict(model,n.ahead=2)

par(mfrow=c(1,2))

acf(diff(returns,2), main ="ACF of Time Series")

pacf(diff(returns,2), main ="PACF of Time Series")

plot(table$Open, xlab = "Time (Months)", ylab = "Price",
     main="Dow Jones Industrial Average Open Price (January 2005 - Febrary 2017)",
     cex.main=0.8, type = "l")

model2 <- arima(diff(returns,2), order=c(2,0,2), method="ML"); 

model2
