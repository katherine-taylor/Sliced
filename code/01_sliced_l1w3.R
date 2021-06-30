# 01_sliced_practice.R
# 06-15-21

# libraries
library(tidyverse)
library(tidymodels)
library(MASS)

# import data
train <- read_csv(here::here("data/train.csv"))
test <- read_csv(here::here("data/test.csv"))
# set seed 
SEED <- 1

glimpse(train)

# create validation set
set.seed(SEED)
split <- sample(1:nrow(train), round(nrow(train) * 0.5))
train_df <- train[-(split), ]
val_df <- train[split,]


model_1 <- lm(profit~ship_mode + segment + city + state
              + postal_code + region + category + sub_category
              + sales + quantity + discount, data = train_df)

summary(model_1)

step_model_1 <- stats::step(model_1)

model_2 <- lm(profit + 3840 ~ sub_category + sales + quantity + discount, data = train_df)
summary(model_2)

par(mfrow = c(2,2))
plot(model_2)
par(mfrow = c(1,1))

model_3 <- lm((profit+3840)^1.5 ~ sub_category * sales * quantity * discount, data = train_df)
summary(model_3)
boxcox(model_3)
sd(model_3$residuals)

par(mfrow = c(2,2))
plot(model_3)
par(mfrow = c(1,1))

predictions <- predict(model_3, val_df)

sd(val_df$profit - predictions)

test_predictions <- predict(model_3, test)

submit_df <- data.frame("id" = test$id, "profit" = ((test_predictions^(1/1.5))-3840))
submission <- write_csv(submit_df, here::here("data/kath_submission_5.csv"))

sum(is.na(submit_df))

