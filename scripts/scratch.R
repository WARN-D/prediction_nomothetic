```{r}
## Save test data
df_ml_lstm_week_train$external_id %>%
  as.matrix() %>%
  py_save_object(here(owncloud_file, 'data/pickles/week_train_subs_new.pickle'))

feature_mat(df_ml_lstm_week_train, vars = ema_vars) %>%
  r_to_py() %>%
  py_save_object( here(owncloud_file, 'data/pickles/week_train_ema_new.pickle'))

feature_mat(df_ml_lstm_week_train, vars = epa_vars) %>% 
  r_to_py() %>%
  py_save_object( here(owncloud_file, 'data/pickles/week_train_epa_new.pickle'))

feature_mat(df_ml_lstm_week_train, vars = c(ema_vars, epa_vars)) %>% 
  r_to_py() %>% 
  py_save_object( here(owncloud_file, 'data/pickles/week_train_all_new.pickle'))

df_ml_lstm_week_train %>% 
  select(phq9_sum_w) %>% 
  as.matrix %>%
  py_save_object(here(owncloud_file, 'data/pickles/week_train_phq_new.pickle'))

## Save validation data
df_ml_lstm_week_test$external_id %>% 
  as.matrix() %>%
  py_save_object(here(owncloud_file, 'data/pickles/week_val_subs_new.pickle'))

feature_mat(df_ml_lstm_week_test, vars = ema_vars) %>% 
  r_to_py() %>%
  py_save_object(here(owncloud_file, 'data/pickles/week_val_ema_new.pickle'))

feature_mat(df_ml_lstm_week_test, vars = epa_vars) %>%
  r_to_py() %>%
  py_save_object(here(owncloud_file, 'data/pickles/week_val_epa_new.pickle'))

feature_mat(df_ml_lstm_week_test, vars = c(ema_vars, epa_vars)) %>%
  r_to_py() %>%
  py_save_object(here(owncloud_file, 'data/pickles/week_val_all_new.pickle'))

df_ml_lstm_week_test %>% 
  select(phq9_sum_w) %>% 
  as.matrix %>%
  py_save_object(here(owncloud_file, 'data/pickles/week_val_phq_new.pickle'))

# Variable List
ema_vars %>% 
  as.matrix() %>%
  py_save_object(here(owncloud_file, 'data/pickles/vars_ema_new.pickle'))
epa_vars %>%
  as.matrix() %>%
  py_save_object(here(owncloud_file, 'data/pickles/vars_epa_new.pickle'))
features %>%  
  as.matrix() %>% 
  py_save_object(here(owncloud_file, 'data/pickles/vars_all_new.pickle'))
```


## Daily Data



```{r}

# Daily Data
## Save test data
df_ml_lstm_day_train$external_id %>% 
  as.matrix() %>%
  py_save_object(here(owncloud_file, 'data/pickles/day_train_subs_new.pickle'))

feature_mat(df_ml_lstm_day_train, vars = ema_vars) %>%
  reticulate::r_to_py() %>% 
  py_save_object( here(owncloud_file, 'data/pickles/day_train_ema_new.pickle'))

feature_mat(df_ml_lstm_day_train, vars = epa_vars) %>% 
  reticulate::r_to_py() %>% 
  py_save_object( here(owncloud_file, 'data/pickles/day_train_epa_new.pickle'))

feature_mat(df_ml_lstm_day_train, vars = c(ema_vars, epa_vars) ) %>% 
  reticulate::r_to_py() %>% 
  py_save_object( here(owncloud_file, 'data/pickles/day_train_all_new.pickle'))

df_ml_lstm_day_train %>% 
  select(phq2_sum_e) %>%
  as.matrix %>% 
  py_save_object(here(owncloud_file, 'data/pickles/day_train_phq_new.pickle'))

## Save validation data
df_ml_lstm_day_test$external_id %>%
  as.matrix() %>%
  py_save_object(here(owncloud_file, 'data/pickles/day_val_subs_new.pickle'))

feature_mat(df_ml_lstm_day_test, vars = ema_vars) %>%
  reticulate::r_to_py() %>% 
  py_save_object(here(owncloud_file, 'data/pickles/day_val_ema_new.pickle'))

feature_mat(df_ml_lstm_day_test, vars = epa_vars) %>%
  reticulate::r_to_py() %>% 
  py_save_object(here(owncloud_file, 'data/pickles/day_val_epa_new.pickle'))

feature_mat(df_ml_lstm_day_test, vars = c(ema_vars, epa_vars)) %>% 
  reticulate::r_to_py() %>% 
  py_save_object(here(owncloud_file, 'data/pickles/day_val_all_new.pickle'))

df_ml_lstm_day_test %>%
  select(phq2_sum_e) %>%
  as.matrix %>% 
  py_save_object(here(owncloud_file, 'data/pickles/day_val_phq_new.pickle'))

```


## PHQ Data
```{r}
# Day PHQ

## Train
df_ml_lstm_day_train %>% 
  group_by(external_id) %>% 
  mutate(phq2_sum_e_lag1 = lag(phq2_sum_e), 
         phq2_sum_e_lag1 = if_else(is.na(phq2_sum_e_lag1), round(median(phq2_sum_e_lag1, na.rm=T)), phq2_sum_e_lag1)) %>%
  fill(phq2_sum_e_lag1, .direction = 'up', ) %>%
  select(1:4, phq2_sum_e_lag1) %>% 
  ungroup() %>%
  select(phq2_sum_e_lag1) %>%
  as.matrix %>% 
  py_save_object(here(owncloud_file, 'data/day_train_phq_lag1_new.pickle'))

## Test
df_ml_lstm_day_test %>% 
  group_by(external_id) %>% 
  mutate(phq2_sum_e_lag1 = lag(phq2_sum_e), 
         phq2_sum_e_lag1 = if_else(is.na(phq2_sum_e_lag1), round(median(phq2_sum_e_lag1, na.rm=T)), phq2_sum_e_lag1)) %>%
  fill(phq2_sum_e_lag1, .direction = 'up', ) %>%
  select(1:4, phq2_sum_e_lag1) %>% 
  ungroup() %>%
  select(phq2_sum_e_lag1) %>%
  as.matrix %>% 
  py_save_object(here(owncloud_file, 'data/day_test_phq_lag1_new.pickle'))

# Pivot to wide and make matrix per variable
df_ml_lstm_week_train %>% 
  group_by(external_id) %>% 
  mutate(phq9_sum_w_lag1 = lag(phq9_sum_w), 
         phq9_sum_w_lag1 = if_else(is.na(phq9_sum_w_lag1), round(median(phq9_sum_w_lag1, na.rm=T)), phq9_sum_w_lag1)) %>%
  fill(phq9_sum_w_lag1, .direction = 'up', ) %>%
  select(1:4, phq9_sum_w_lag1) %>% 
  ungroup() %>%
  select(phq9_sum_w_lag1) %>%
  as.matrix %>% 
  py_save_object(here(owncloud_file, 'data/week_train_phq_lag1_new.pickle'))


df_ml_lstm_week_test %>% 
  group_by(external_id) %>% 
  mutate(phq9_sum_w_lag1 = lag(phq9_sum_w), 
         phq9_sum_w_lag1 = if_else(is.na(phq9_sum_w_lag1), round(median(phq9_sum_w_lag1, na.rm=T)), phq9_sum_w_lag1)) %>%
  fill(phq9_sum_w_lag1, .direction = 'up', ) %>%
  select(1:4, phq9_sum_w_lag1) %>% 
  ungroup() %>%
  select(phq9_sum_w_lag1) %>%
  as.matrix %>% 
  py_save_object(here(owncloud_file, 'data/week_test_phq_lag1_new.pickle'))


```





