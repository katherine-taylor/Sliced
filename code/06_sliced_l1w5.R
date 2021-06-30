# 04_sliced_l1w5.R
# 6 - 29 - 21


# Libraries ---------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(textrecipes)
library(xgboost)
library(stopwords)


# Import Data -------------------------------------------------------------

train <- read_csv(here::here("data/week_05/train.csv"))
test <- read_csv(here::here("data/week_05/test.csv"))

skimr::skim(train)


# Data Exploration --------------------------------------------------------

train |>
  ggplot(aes(x = log(price))) +
  geom_histogram()
# definitely logging price


# Resampling --------------------------------------------------------------

set.seed(317)
price_folds <- vfold_cv(train, v = 5)
price_folds

# Recipe ------------------------------------------------------------------

price_rec <- recipe(price~., data = train) |>
  update_role(id,host_name, new_role = "id_variable") |>
  step_tokenize(name) |>
  step_stopwords(name) |>
  step_tokenfilter(name, max_tokens = 100) |>
  step_tfidf(name) |>
  step_rm(last_review) |>
  step_impute_median(reviews_per_month) |>
  step_other(neighbourhood, threshold = 0.01) |>
  step_dummy(all_nominal_predictors()) |>
  step_novel(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_unknown(all_nominal_predictors())


# Workflow ----------------------------------------------------------------

xg_wf <- workflow() |>
  add_recipe(price_rec) |>
  add_model(boost_tree("regression",
                       mtry = tune(),
                       trees = tune(),
                       learn_rate = 0.01) |>set_engine("xgboost"))

doParallel::registerDoParallel()
set.seed(517)

xg_rs <- tune_grid(
  xg_wf,
  resamples = price_folds,
  grid = crossing(mtry = c(5,8,10),
                         trees = c(1000,1500))
)

xg_rs
autoplot(xg_rs)

show_best(xg_rs, "rmse")
  
# Model Fit ---------------------------------------------------------------

best_rmse <- select_best(xg_rs, "rmse")

final_xg <- finalize_workflow(xg_wf, best_rmse)
final_xg

# Submission --------------------------------------------------------------

sub <- fit(final_xg, train) |>
  augment(test) 

submission <- sub |> select(id, .pred) |> mutate(price = .pred) |> select(-.pred)

write_csv(submission, here::here("data/week_05/kath_submission_06.csv"))
