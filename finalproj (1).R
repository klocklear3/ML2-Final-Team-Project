library(randomForest)
library(e1071)
library(caret)
library(dplyr)
library(MASS)
library(stringr)
library(Matrix)
library(xgboost)
getwd()
setwd('C:/Users/eason/Desktop/ML2/finalpres')
df <- read.csv('train_users_2.csv')
df <- df[sample(nrow(df), nrow(df)*.06),] 
str(df)

##Converting factors to factors
cols <- c('gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser','country_destination')
df[cols] <- lapply(df[cols], factor)

##Converting dates to dates
df$date_account_created <- as.Date(df$date_account_created)
df$date_first_booking <- as.Date(df$date_first_booking)
df$timestamp_first_active <- as.Date(paste(substring(as.character(df$timestamp_first_active), 1, 4),substring(as.character(df$timestamp_first_active), 5, 6),substring(as.character(df$timestamp_first_active), 7, 8), sep = '-'))

##Checking variables
str(df)
summary(df)

##Remove id
df <- df[,-1]

##Remove date_first_booking
df <- df[,!names(df) %in% c("date_first_booking")]

##Inspecting age
length(which(df$age>100))
length(which(df$age<18))
hist(df[-which(df$age<18 | df$age>100),]$age)
hist(log(df[-which(df$age<18 | df$age>100),]$age))
##Log provides better distribution


#Turning all null ages to -1
df[is.na(df)] <- -1

###Splitting into train and test
trainIndex <- sample(nrow(df), nrow(df)*.7)

##TrainTestSplit
dftr <- df[trainIndex,]
dfte <- df[-trainIndex,]


#Calculating mean and SD of ages between 18-100 for train and test
avetr <- mean(dftr$age[which(18<dftr$age & dftr$age<100)])
stdevtr <- sd(dftr$age[which(18<dftr$age & dftr$age<100)])

avete <- mean(dfte$age[which(18<dfte$age & dfte$age<100)])
stdevte <- sd(dfte$age[which(18<dfte$age & dfte$age<100)])



##Replacing nulls and outliers with normally distributed ages
for(i in 1:nrow(dftr)){
     if(dftr$age[i] < 18 || dftr$age[i] > 100){
          dftr$age[i] <- max(avetr + rnorm(1)*stdevtr,18)
     }
}

for(i in 1:nrow(dfte)){
     if(dfte$age[i] < 18 || dfte$age[i] > 100){
          dfte$age[i] <- max(avete + rnorm(1)*stdevte,18)
     }
}


##Applying log transformation to train and test separately
dftr$age <- log(dftr$age)
dfte$age <- log(dfte$age)



###Random Forest model
rf.df <- randomForest(country_destination ~ .,
                      data= dftr,
                      mtry=5,
                      importance = TRUE)
yhat<-predict(rf.df,newdata=dfte)
rfacc <- mean(yhat == dfte$country_destination)
rfacc

varImpPlot(rf.df)


###SVM

svmtune.df <- tune(svm,
                 country_destination ~ .,
                 data = dftr,
                 kernel ="linear",
                 ranges=list(cost=c(0.1, 0.5, 1)))

yhatsvm <- predict(svmtune.df$best.model, dfte)
svmacc <- mean(yhatsvm == dfte$country_destination)
svmacc

###XGBoost
dfXG <- df
dfXGtr <- dftr
dfXGte <- dfte

##Coercing date_account_created to numeric for XGBoost
dac = as.data.frame(str_split_fixed(dfXGtr$date_account_created, '-', 3))
dfXGtr['dac_year'] = as.factor(dac[,1])
dfXGtr['dac_month'] = as.factor(dac[,2])
dfXGtr['dac_day'] = as.factor(dac[,3])
dfXGtr = dfXGtr[,-c(which(colnames(dfXGtr) %in% c('date_account_created')))]

dac = as.data.frame(str_split_fixed(dfXGte$date_account_created, '-', 3))
dfXGte['dac_year'] = as.factor(dac[,1])
dfXGte['dac_month'] = as.factor(dac[,2])
dfXGte['dac_day'] = as.factor(dac[,3])
dfXGte = dfXGte[,-c(which(colnames(dfXGte) %in% c('date_account_created')))]

##Coercing timestamp_first_active to numeric for XGBoost
dfXGtr[,'tfa_year'] = as.factor(substring(as.character(dfXGtr[,'timestamp_first_active']), 1, 4))
dfXGtr['tfa_month'] = as.factor(substring(as.character(dftr$timestamp_first_active), 6, 7))
dfXGtr['tfa_day'] = as.factor(substring(as.character(dftr$timestamp_first_active), 9, 10))
dfXGtr = dfXGtr[,-c(which(colnames(dfXGtr) %in% c('timestamp_first_active')))]

dfXGte[,'tfa_year'] = as.factor(substring(as.character(dfXGte[,'timestamp_first_active']), 1, 4))
dfXGte['tfa_month'] = as.factor(substring(as.character(dfte$timestamp_first_active), 6, 7))
dfXGte['tfa_day'] = as.factor(substring(as.character(dfte$timestamp_first_active), 9, 10))
dfXGte = dfXGte[,-c(which(colnames(dfXGte) %in% c('timestamp_first_active')))]


cd = dfXG$country_destination
label = as.integer(dfXG$country_destination)-1
#dfXG$country_destination = NULL

train.data = sparse.model.matrix(country_destination~.,data = dfXGtr)[,-1]
train.label = label[trainIndex]
test.data = sparse.model.matrix(country_destination~.,data = dfXGte)[,-1]
test.label = label[-trainIndex]
# Transform the two data sets into xgb.Matrix
xgb.train = xgb.DMatrix(data=train.data,label=train.label)
xgb.test = xgb.DMatrix(data=test.data,label=test.label)


# Define the parameters for multinomial classification
num_class = length(levels(cd))
params = list(
     booster="gbtree",
     eta=0.1,
     max_depth=9,
     gamma=3,
     subsample=0.5,
     colsample_bytree=.5,
     objective="multi:softprob",
     eval_metric="merror",
     num_class=num_class
)



# Train the XGBoost classifer
xgb.fit=xgb.train(
     params=params,
     data=xgb.train,
     nrounds=10000,
     watchlist=list(val1=xgb.train,val2=xgb.test),
     verbose=0
)

# Review the final model and results
xgb.fit

# Predict outcomes with the test data
xgb.pred = predict(xgb.fit,test.data,reshape=T)
xgb.pred = as.data.frame(xgb.pred)
colnames(xgb.pred) = levels(cd)

# Use the predicted label with the highest probability
xgb.pred$prediction = apply(xgb.pred,1,function(x) colnames(xgb.pred)[which.max(x)])
xgb.pred$label = levels(cd)[test.label+1]

# Calculate the final accuracy
xgbacc = mean(xgb.pred$prediction==xgb.pred$label)
xgbacc


##### Since a majority of the gender category is unknown, if I am sampling the data anyways, I might as well remove the rows with unknown gender
df <- read.csv('train_users_2.csv')
df <- df[-which(df$gender == '-unknown-'),]
df <- df[sample(nrow(df), nrow(df)*.1),]

str(df)

##Converting factors to factors
cols <- c('gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser','country_destination')
df[cols] <- lapply(df[cols], factor)

##Converting dates to dates
df$date_account_created <- as.Date(df$date_account_created)
df$date_first_booking <- as.Date(df$date_first_booking)
df$timestamp_first_active <- as.Date(paste(substring(as.character(df$timestamp_first_active), 1, 4),substring(as.character(df$timestamp_first_active), 5, 6),substring(as.character(df$timestamp_first_active), 7, 8), sep = '-'))

##Checking variables
str(df)
summary(df)

##Remove id
df <- df[,-1]

##Remove date_first_booking
df <- df[,!names(df) %in% c("date_first_booking")]

##Inspecting age
length(which(df$age>100))
length(which(df$age<18))
hist(df[-which(df$age<18 | df$age>100),]$age)
hist(log(df[-which(df$age<18 | df$age>100),]$age))
##Log provides better distribution


#Turning all null ages to -1
df[is.na(df)] <- -1

#train test split
trainIndex <- sample(nrow(df), nrow(df)*.7)


##TrainTestSplit
dftr <- df[trainIndex,]
dfte <- df[-trainIndex,]


#Calculating mean and SD of ages between 18-100 for train and test
avetr <- mean(dftr$age[which(18<dftr$age & dftr$age<100)])
stdevtr <- sd(dftr$age[which(18<dftr$age & dftr$age<100)])

avete <- mean(dfte$age[which(18<dfte$age & dfte$age<100)])
stdevte <- sd(dfte$age[which(18<dfte$age & dfte$age<100)])



##Replacing nulls and outliers with normally distributed ages
for(i in 1:nrow(dftr)){
     if(dftr$age[i] < 18 || dftr$age[i] > 100){
          dftr$age[i] <- max(avetr + rnorm(1)*stdevtr,18)
     }
}

for(i in 1:nrow(dfte)){
     if(dfte$age[i] < 18 || dfte$age[i] > 100){
          dfte$age[i] <- max(avete + rnorm(1)*stdevte,18)
     }
}


##Applying log transformation to train and test separately
dftr$age <- log(dftr$age)
dfte$age <- log(dfte$age)

###Splitting into train and test
trainIndex <- sample(nrow(df), nrow(df)*.7)


###Random Forest model
rf.df <- randomForest(country_destination ~ .,
                      data= dftr,
                      mtry=5,
                      importance = TRUE)
yhat<-predict(rf.df,newdata=dfte)
rfacc1 <- mean(yhat == dfte$country_destination)
rfacc1

varImpPlot(rf.df)


###SVM

svmtune.df <- tune(svm,
                   country_destination ~ .,
                   data = dftr,
                   kernel ="linear",
                   ranges=list(cost=c(0.1, 0.5, 1)))

yhatsvm <- predict(svmtune.df$best.model, dfte)
svmacc1 <- mean(yhatsvm == dfte$country_destination)
svmacc1

###XGBoost
dfXG <- df
dfXGtr <- dftr
dfXGte <- dfte

##Coercing date_account_created to numeric for XGBoost
dac = as.data.frame(str_split_fixed(dfXGtr$date_account_created, '-', 3))
dfXGtr['dac_year'] = as.factor(dac[,1])
dfXGtr['dac_month'] = as.factor(dac[,2])
dfXGtr['dac_day'] = as.factor(dac[,3])
dfXGtr = dfXGtr[,-c(which(colnames(dfXGtr) %in% c('date_account_created')))]

dac = as.data.frame(str_split_fixed(dfXGte$date_account_created, '-', 3))
dfXGte['dac_year'] = as.factor(dac[,1])
dfXGte['dac_month'] = as.factor(dac[,2])
dfXGte['dac_day'] = as.factor(dac[,3])
dfXGte = dfXGte[,-c(which(colnames(dfXGte) %in% c('date_account_created')))]

##Coercing timestamp_first_active to numeric for XGBoost
dfXGtr[,'tfa_year'] = as.factor(substring(as.character(dfXGtr[,'timestamp_first_active']), 1, 4))
dfXGtr['tfa_month'] = as.factor(substring(as.character(dftr$timestamp_first_active), 6, 7))
dfXGtr['tfa_day'] = as.factor(substring(as.character(dftr$timestamp_first_active), 9, 10))
dfXGtr = dfXGtr[,-c(which(colnames(dfXGtr) %in% c('timestamp_first_active')))]

dfXGte[,'tfa_year'] = as.factor(substring(as.character(dfXGte[,'timestamp_first_active']), 1, 4))
dfXGte['tfa_month'] = as.factor(substring(as.character(dfte$timestamp_first_active), 6, 7))
dfXGte['tfa_day'] = as.factor(substring(as.character(dfte$timestamp_first_active), 9, 10))
dfXGte = dfXGte[,-c(which(colnames(dfXGte) %in% c('timestamp_first_active')))]


cd = dfXG$country_destination
label = as.integer(dfXG$country_destination)-1
#dfXG$country_destination = NULL

train.data = sparse.model.matrix(country_destination~.,data = dfXGtr)[,-1]
train.label = label[trainIndex]
test.data = sparse.model.matrix(country_destination~.,data = dfXGte)[,-1]
test.label = label[-trainIndex]

# Transform the two data sets into xgb.Matrix
xgb.train = xgb.DMatrix(data=train.data,label=train.label)
xgb.test = xgb.DMatrix(data=test.data,label=test.label)


# Define the parameters for multinomial classification
num_class = length(levels(cd))
params = list(
     booster="gbtree",
     eta=0.1,
     max_depth=9,
     gamma=3,
     subsample=0.5,
     colsample_bytree=.5,
     objective="multi:softprob",
     eval_metric="merror",
     num_class=num_class
)



# Train the XGBoost classifer
xgb.fit=xgb.train(
     params=params,
     data=xgb.train,
     nrounds=10000,
     watchlist=list(val1=xgb.train,val2=xgb.test),
     verbose=0
)

# Review the final model and results
xgb.fit

# Predict outcomes with the test data
xgb.pred = predict(xgb.fit,test.data,reshape=T)
xgb.pred = as.data.frame(xgb.pred)
colnames(xgb.pred) = levels(cd)

# Use the predicted label with the highest probability
xgb.pred$prediction = apply(xgb.pred,1,function(x) colnames(xgb.pred)[which.max(x)])
xgb.pred$label = levels(cd)[test.label+1]

# Calculate the final accuracy
xgbacc1 = mean(xgb.pred$prediction==xgb.pred$label)
xgbacc1

