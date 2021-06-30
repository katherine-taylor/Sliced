# sliced week 4
# 02_sliced_practice.R

library(tidyverse)
library(tree)

train <- read_csv(here::here("data/week_04/train.csv"))
test <- read_csv(here::here("data/week_04/test.csv"))

glimpse(train)

# validation set
sample <- sample(1:nrow(train),nrow(train)*.2)
train_set <- train[-sample,]
val_set <- train[sample,]
# fit tree model
train_set$rain_tomorrow <- as.factor(train_set$rain_tomorrow)
val_set$rain_tomorrow <- as.factor(val_set$rain_tomorrow)

train_set$rain_today <- as.factor(train_set$rain_today)
val_set$rain_today <- as.factor(val_set$rain_today)
test$rain_today <- as.factor(test$rain_today)

library(Hmisc)
train_set$min_temp <- impute(train_set$min_temp, median)
train_set$max_temp <- impute(train_set$max_temp, median)
train_set$min_temp <- impute(train_set$min_temp, median)
train_set$rainfall <- impute(train_set$rainfall, median)
train_set$evaporation <- impute(train_set$evaporation, median)
train_set$sunshine <- impute(train_set$sunshine, median)
train_set$wind_gust_speed <- impute(train_set$wind_gust_speed, median)
train_set$wind_speed9am <- impute(train_set$wind_speed9am, median)
train_set$wind_speed3pm <- impute(train_set$wind_speed3pm, median)
train_set$sunshine <- impute(train_set$sunshine, median)
train_set$humidity3pm <- impute(train_set$humidity3pm, median)
train_set$humidity9am <- impute(train_set$humidity9am, median)
train_set$pressure3pm <- impute(train_set$pressure3pm, median)
train_set$pressure9am <- impute(train_set$pressure9am, median)
train_set$cloud3pm <- impute(train_set$cloud3pm, median)
train_set$cloud9am <- impute(train_set$cloud9am, median)
train_set$temp3pm <- impute(train_set$temp3pm, median)
train_set$temp9am <- impute(train_set$temp9am, median)
train_set$rain_today <- impute(train_set$rain_today, median)

val_set$min_temp <- impute(val_set$min_temp, median)
val_set$max_temp <- impute(val_set$max_temp, median)
val_set$min_temp <- impute(val_set$min_temp, median)
val_set$rainfall <- impute(val_set$rainfall, median)
val_set$evaporation <- impute(val_set$evaporation, median)
val_set$sunshine <- as.logical(impute(val_set$sunshine, median))
val_set$wind_gust_speed <- impute(val_set$wind_gust_speed, median)
val_set$wind_speed9am <- impute(val_set$wind_speed9am, median)
val_set$wind_speed3pm <- impute(val_set$wind_speed3pm, median)
val_set$sunshine <- impute(val_set$sunshine, median)
val_set$humidity3pm <- impute(val_set$humidity3pm, median)
val_set$humidity9am <- impute(val_set$humidity9am, median)
val_set$pressure3pm <- impute(val_set$pressure3pm, median)
val_set$pressure9am <- impute(val_set$pressure9am, median)
val_set$cloud3pm <- impute(val_set$cloud3pm, median)
val_set$cloud9am <- impute(val_set$cloud9am, median)
val_set$temp3pm <- impute(val_set$temp3pm, median)
val_set$temp9am <- impute(val_set$temp9am, median)
val_set$rain_today <- impute(val_set$rain_today, median)

test$min_temp <- impute(test$min_temp, median)
test$max_temp <- impute(test$max_temp, median)
test$min_temp <- impute(test$min_temp, median)
test$rainfall <- impute(test$rainfall, median)
test$evaporation <- impute(test$evaporation, median)
test$sunshine <- as.logical(impute(test$sunshine, median))
test$wind_gust_speed <- impute(test$wind_gust_speed, median)
test$wind_speed9am <- impute(test$wind_speed9am, median)
test$wind_speed3pm <- impute(test$wind_speed3pm, median)
test$sunshine <- impute(test$sunshine, median)
test$humidity3pm <- impute(test$humidity3pm, median)
test$humidity9am <- impute(test$humidity9am, median)
test$pressure3pm <- impute(test$pressure3pm, median)
test$pressure9am <- impute(test$pressure9am, median)
test$cloud3pm <- impute(test$cloud3pm, median)
test$cloud9am <- impute(test$cloud9am, median)
test$temp3pm <- impute(test$temp3pm, median)
test$temp9am <- impute(test$temp9am, median)
test$rain_today <- impute(test$rain_today, median)
# normal tree
tree_model <- tree(rain_tomorrow ~ ., data = train_set)

summary(tree_model)

plot(tree_model)
text(tree_model)

predictions <- predict(tree_model, val_set)

acc <- data.frame("actual" = as.numeric(val_set$rain_tomorrow), "predicted" = predictions[,1])
log(sum(abs(acc$actual - acc$predicted)))

test_predictions <- predict(tree_model, test)
sum(is.na(test_predictions))
sub <- data.frame("id" = test$id, "rain_tomorrow" = rf_model$test$votes[,1])
write_csv(sub, here::here("data/week_04/kath_submission_03.csv"))

rf_train <- train_set[,c(-9,-11,-12)]
test <- test[,c(-9,-11,-12)]
glimpse(rf_train)

library(randomForest)

rf_model<- randomForest(x= rf_train[-21], y = rf_train$rain_tomorrow,
             xtest = test)

predict(rf_model$forest, test)



glm(rain_tomorrow ~ rain_today + temp9am + wind_gust_speed, family = "binomial", data = train_set)
