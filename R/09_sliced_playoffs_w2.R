# 09_sliced_playoffs_w2.R
# Tue Aug  3 20:56:17 2021 ------------------------------



# Libraries ---------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(textrecipes)
library(baguette)
library(xgboost)

tidymodels_prefer()


# Read in Data ------------------------------------------------------------

train <- read_csv(here::here("data/week_10/train.csv"))
test <- read_csv(here::here("data/week_10/test.csv"))

# multiclass oh boy, also very few numerical features


# Exploration -------------------------------------------------------------

skimr::skim(train)
glimpse(train)


# Recipe ------------------------------------------------------------------

animal_recipe <- recipe(outcome_type ~ ., data = train) |>
  update_role(id, new_role = "id") |>
  step_rm(age_upon_outcome, breed, name) |>
  step_mutate(age = as.Date(datetime) - date_of_birth,
              age = as.numeric(age)) |> 
  step_date(date_of_birth, features = "year", keep_original_cols = FALSE) |>
  step_date(datetime, features = c("year","month"), keep_original_cols = FALSE) |>
  step_novel(all_nominal_predictors()) |>
  step_other(animal_type, color, threshold = 0.05) |>
  step_unknown(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors())

xg_recipe <- recipe(outcome_type ~ ., data = train) |>
  update_role(id, new_role = "id") |>
  step_rm(age_upon_outcome, breed, name, ) |>
  step_mutate(age = as.Date(datetime) - date_of_birth,
              age = as.numeric(age)) |> 
  step_date(date_of_birth, features = "year", keep_original_cols = FALSE) |>
  step_date(datetime, features = "year", keep_original_cols = FALSE) |>
  step_novel(all_nominal_predictors()) |>
  step_other(animal_type, color, threshold = 0.05) |>
  step_unknown(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors())

prep(animal_recipe) |>
  juice() |>
  skimr::skim()



# Bagged Model ------------------------------------------------------------

# Workflow ----------------------------------------------------------------

bag_spec <-
  bag_tree(min_n = 30) %>%
  set_engine("rpart", times = 25) %>%
  set_mode("classification")

imb_wf <-
  workflow() %>%
  add_recipe(animal_recipe) %>%
  add_model(bag_spec)


# Folds -------------------------------------------------------------------

set.seed(08032021)
animal_folds <- vfold_cv(train, v = 5, strata = outcome_type)
animal_folds
animal_metrics <- metric_set(mn_log_loss, accuracy, sensitivity, specificity)

doParallel::registerDoParallel()
set.seed(08032021)
imb_rs <-
  fit_resamples(
    imb_wf,
    resamples = animal_folds,
    metrics = animal_metrics
  )

collect_metrics(imb_rs)

sub <- fit(imb_wf, train) |>
  predict(new_data = test, type = "prob")

# first submission!

animal_pred <- cbind(test$id, sub) |>
  rename("id" = `test$id`,
    "adoption" = .pred_adoption,
         "no outcome" = `.pred_no outcome`,
         "transfer" = .pred_transfer)


write_csv(animal_pred,
          here::here("data/week_10/kath_submission_03.csv"))

# Tuned XGBoost -----------------------------------------------------------

xg_wf <- workflow() |>
  add_recipe(xg_recipe) |>
  add_model(boost_tree("classification",
                       mtry = tune(),
                       trees = tune(),
                       learn_rate = .01) |>set_engine("xgboost"))
set.seed(08032021)
doParallel::registerDoParallel()
xg_rs <- tune_grid(
  xg_wf,
  resamples = animal_folds,
  grid = crossing(mtry = c(5,10,15),
                  trees = 1000),
  metrics = animal_metrics
)  

xg_rs
autoplot(xg_rs)

show_best(xg_rs, "mn_log_loss")

best_ll <- select_best(xg_rs, "mn_log_loss")

final_xg <- finalize_workflow(xg_wf, best_ll)
final_xg


# submission

doParallel::registerDoParallel()
set.seed(08032021)
sub <- fit(final_xg, train) |>
  predict(new_data = test, type = "prob")

animal_pred <- cbind(test$id, sub) |>
  rename("id" = `test$id`,
         "adoption" = .pred_adoption,
         "no outcome" = `.pred_no outcome`,
         "transfer" = .pred_transfer)

write_csv(animal_pred, here::here("data/week_10/kath_submission_02.csv"))
