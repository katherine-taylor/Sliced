# 03_sliced_l1w4_rf_model.R
# 06 - 28 - 21


# Libraries ---------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(here)
library(lubridate)
library(xgboost)



# Import Data -------------------------------------------------------------

train <- read_csv(here::here("data/week_04/train.csv")) |>
  mutate(rain_tomorrow = ifelse(rain_tomorrow, "Rain", "No Rain"))
test <- read_csv(here::here("data/week_04/test.csv"))


# Model Recipe ------------------------------------------------------------
set.seed(06282021)

rain_rec <- recipe(rain_tomorrow ~ ., data = train) |>
  update_role(id, new_role = "id_variable") |>
  step_mutate(year = year(date),
              month = month(date)) |>
  step_rm(date) |>
  step_log(rainfall, offset = 1, base = 2) |>
  step_other(location, threshold = 0.005) |>
  step_dummy(location) |>
  step_mutate(
    wind_gust_dir = str_sub(wind_gust_dir, 1, 1),
    wind_dir9am = str_sub(wind_dir9am, 1, 1),
    wind_dir3pm = str_sub(wind_dir3pm, 1, 1)
  ) |>
  step_novel(all_nominal_predictors()) |>
  step_other(all_nominal_predictors(), threshold = 0.01) |>
  step_unknown(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_impute_median(all_numeric_predictors()) |>
  step_zv(all_predictors())


# Workflow ----------------------------------------------------------------
xg_wf <- workflow() |>
  add_recipe(rain_rec) |>
  add_model(boost_tree("classification",
                       mtry = 10,
                       trees = 500,
                       learn_rate = .01) |>set_engine("xgboost"))


# Model Fit ---------------------------------------------------------------

xg_fit <- fit(xg_wf, data = train)
xg_fit


# Resampling --------------------------------------------------------------

set.seed(6282021)
rain_folds <- vfold_cv(train, v = 5, strata = rain_tomorrow)
rain_folds
rain_metrics <- metric_set(mn_log_loss, accuracy, sensitivity, specificity)

doParallel::registerDoParallel()
set.seed(6282021)
xg_rs <-
  fit_resamples(
    xg_wf,
    resamples = rain_folds,
    metrics = rain_metrics
  )

collect_metrics(xg_rs)