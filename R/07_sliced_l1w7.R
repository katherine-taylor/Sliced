# 07_sliced_l1w7.R
# Tue Jul 13 20:50:10 2021 ------------------------------


# Libraries ---------------------------------------------------------------
library(tidyverse)
library(tidymodels)
library(xgboost)
library(baguette)
library(stacks)
tidymodels_prefer()

# Import data -------------------------------------------------------------

train <- read_csv(here::here("data/week_07/train.csv"))
test <- read_csv(here::here("data/week_07/test.csv"))

skimr::skim(train)
skimr::skim(test)
glimpse(train)

# no missing data

# Data Vis ----------------------------------------------------------------

train |>
  pivot_longer(cols = c(
    customer_age,
    total_relationship_count:avg_utilization_ratio
  )) |>
  mutate(
    attrition_flag = as.factor(attrition_flag),
    education_level = parse_factor(education_level),
    income_category = parse_factor(income_category)
  ) |>
  ggplot(aes(y = value, fill = attrition_flag)) +
  geom_boxplot() +
  facet_wrap( ~ name * education_level, scales = "free")


# Initial Recipe ----------------------------------------------------------

train <-
  train |> mutate(attrition_flag = ifelse(attrition_flag, "bye", "still_here"))

churn_rec <- recipe(attrition_flag ~ ., data = train) |>
  update_role(id, new_role = "id_variable") |>
  step_log(credit_limit, base = 2) |>
  step_other(education_level, income_category, threshold = 0.01) |>
  step_dummy(all_nominal_predictors()) |>
  step_interact(terms = ~education_level : gender) |>
  step_novel(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_unknown(all_nominal_predictors())

# potentially would add interaction terms in the future

# Initial Workflow --------------------------------------------------------

set.seed(07132021)
churn_folds <- vfold_cv(train, v = 5, strata = attrition_flag)
churn_folds
churn_metrics <- metric_set(mn_log_loss, accuracy, sensitivity, specificity)

xg_wf <- workflow() |>
  add_recipe(churn_rec) |>
  add_model(boost_tree("classification",
                       mtry = tune(),
                       trees = tune(),
                       learn_rate = .01) |>set_engine("xgboost"))

xg_rs <- tune_grid(
  xg_wf,
  resamples = churn_folds,
  grid = 10,
  metrics = churn_metrics,
  control = control_stack_grid()
)

# don't need for stacking
xg_rs
autoplot(xg_rs)

show_best(xg_rs, "mn_log_loss")

best_ll <- select_best(xg_rs, "mn_log_loss")

final_xg <- finalize_workflow(xg_wf, best_ll)
final_xg

doParallel::registerDoParallel()
set.seed(07132021)
# xg_rs <-
#   fit_resamples(
#     final_xg,
#     resamples = churn_folds,
#     metrics = churn_metrics
#   )

collect_metrics(xg_rs)


# Bagged Model ------------------------------------------------------------

bag_spec <-
  bag_tree(min_n = tune()) %>%
  set_engine("rpart", times = 25) %>%
  set_mode("classification")

bg_wf <-
  workflow() %>%
  add_recipe(churn_rec) %>%
  add_model(bag_spec)


doParallel::registerDoParallel()
set.seed(07132021)

bag_rs <-
  tune_grid(bg_wf,
            resamples = churn_folds,
            metrics = churn_metrics,
            control = control_stack_grid())

collect_metrics(bag_rs)


# Stacked Models?? --------------------------------------------------------

churn_stack <- stacks() |>
  add_candidates(xg_rs) |>
  add_candidates(bag_rs) |>
  blend_predictions(metric = metric_set(mn_log_loss)) |>
  fit_members()

stacks::autoplot(churn_stack, type = "weights")

churn_pred <-
  test %>%
  bind_cols(predict(churn_stack, ., type = "prob")) %>%
  select(id, .pred_bye) %>%
  rename("attrition_flag" = .pred_bye)

write_csv(churn_pred,
          here::here("data/week_07/kath_submission_06.csv"))
  
# Submission --------------------------------------------------------------

sub <- fit(bg_wf, data = train) |>
  predict(new_data = test, type = "prob")

submission <- sub |> cbind("id" = test$id) |>
  mutate(attrition_flag = .pred_bye) |>
  select(-c(.pred_bye, .pred_still_here))

write_csv(submission,
          here::here("data/week_07/kath_submission_03.csv"))
