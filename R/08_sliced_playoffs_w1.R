# 08_sliced_playoffs_w1
# Tue Jul 27 20:24:39 2021 ------------------------------


# Libraries ---------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(xgboost)
library(baguette)
library(stacks)

tidymodels_prefer()

# Read in Data ------------------------------------------------------------

train <- read_csv(here::here("data/week_09/train.csv"))

test <- read_csv(here::here("data/week_09/test.csv"))

dimensions <- read_csv(here::here("data/week_09/park_dimensions.csv"))


# Preview data ------------------------------------------------------------

train_data <- train |>
  left_join(dimensions, by = "park") |>
  mutate(is_home_run = ifelse(is_home_run, "yup","nope"),
         is_batter_lefty = ifelse(is_batter_lefty, "yup","nope"),
         is_pitcher_lefty = ifelse(is_pitcher_lefty, "yup","nope")
         )

test_data <- test  |>
  left_join(dimensions, by = "park") |>
  mutate(
    is_batter_lefty = ifelse(is_batter_lefty, "yup","nope"),
    is_pitcher_lefty = ifelse(is_pitcher_lefty, "yup","nope")
  )

skimr::skim(train_data)

# First model -------------------------------------------------------------


# Recipe ------------------------------------------------------------------

xg_recipe <- recipe(is_home_run ~., data = train_data) |>
  update_role(batter_name, pitcher_name, NAME, bip_id, batter_id, pitcher_id, home_team,
              away_team, batter_team, new_role = "id") |>
  step_rm(game_date) |>
  step_impute_median(launch_speed, launch_angle) |>
  step_novel(all_nominal_predictors()) |>
  step_unknown(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors())


check_recipe <- xg_recipe |>
  prep() |>
  juice() |>
  skimr::skim()

# Resamples ---------------------------------------------------------------

set.seed(07272021)
bb_folds <- vfold_cv(train_data, v = 5, strata = is_home_run)
bb_folds

bb_metrics <- metric_set(mn_log_loss, accuracy, sensitivity, specificity)


# Workflow ----------------------------------------------------------------

xg_wf <- workflow() |>
  add_recipe(xg_recipe) |>
  add_model(boost_tree("classification",
                       mtry = tune(),
                       trees = 500,
                       min_n = tune(),
                       learn_rate = .01) |>set_engine("xgboost"))

doParallel::registerDoParallel()
xg_rs <- tune_grid(
  xg_wf,
  resamples = bb_folds,
  grid = crossing(mtry = c(5,10,15),
                  trees = c(500,800,1000)),
  metrics = bb_metrics
)  

xg_rs
autoplot(xg_rs)

show_best(xg_rs, "mn_log_loss")

best_ll <- select_best(xg_rs, "mn_log_loss")


# Fit Best Workflow -------------------------------------------------------

final_xg <- finalize_workflow(xg_wf, best_ll)
final_xg

doParallel::registerDoParallel()
set.seed(07272021)
 
xg_rs_final <-
  fit_resamples(
    final_xg,
    resamples = bb_folds,
    metrics = bb_metrics
  )

collect_metrics(xg_rs_final)
  
# submission
sub <- fit(final_xg, train_data) |>
  predict(new_data = test_data, type = "prob")



# Model Stacking ----------------------------------------------------------

doParallel::registerDoParallel()
xg_stack <- tune_grid(
  xg_wf,
  resamples = bb_folds,
  grid = 10,
  metrics = bb_metrics,
  control = control_stack_grid()
)

bag_spec <-
  bag_tree(min_n = tune()) %>%
  set_engine("rpart", times = 25) %>%
  set_mode("classification")

bg_wf <-
  workflow() %>%
  add_recipe(xg_recipe) %>%
  add_model(bag_spec)


doParallel::registerDoParallel()
set.seed(07272021)

bag_stack <-
  tune_grid(bg_wf,
            resamples = bb_folds,
            metrics = bb_metrics,
            control = control_stack_grid())

bb_stack <- stacks() |>
  add_candidates(xg_stack) |>
  add_candidates(bag_stack) |>
  blend_predictions(metric = metric_set(mn_log_loss)) |>
  fit_members()

stacks::autoplot(bb_stack)

bb_pred <-
  test_data %>%
  bind_cols(predict(bb_stack, ., type = "prob")) %>%
  select(bip_id, .pred_yup) %>%
  rename("is_home_run" = .pred_yup)

write_csv(bb_pred,
          here::here("data/week_09/kath_submission_03.csv"))

# Submission --------------------------------------------------------------

submission <- sub |> cbind("bip_id" = test_data$bip_id) |>
  mutate(is_home_run = .pred_yup) |>
  select(-c(.pred_yup, .pred_nope))

write_csv(submission, here::here("data/week_09/kath_submission_02.csv"))
