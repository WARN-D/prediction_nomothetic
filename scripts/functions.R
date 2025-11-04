
# CARET FUNCTIONS ------

# Function to engineer features in the same manner for both test and training sets
ml_feat_engineering = function(df_features_imputed, vars, within = FALSE, feat_group = 'week_nr'){
  #' Feature Engineering Function
  #' 
  #' Function to wrap around feature engineering for 
  #' consistent processing between training and test sets.
  #' 
  #' @param df_features_imputed A dataframe with imputation already done.
  #' @param vars Character vector of the variables which we estimate the features from.
  #' @param within Boolean determining if centering is done at the within or between subject level
  #' @param feat_group Character vector. How to group feature. Within a week, or day.
  #'

  # Message
  message(paste("Estimating features at level of variable", feat_group, '...'))
  
  # Check if within or between and center accordingly
  if (within){
    df_ml_wide = df_features_imputed %>%
      dplyr::group_by(all_of(c('external_id'))) %>% 
      dplyr::mutate(across(all_of(vars), ~ (.x - mean(.x, na.rm = T))/sd(.x, na.rm = T), .names = "{.col}")) 
  } else{
    df_ml_wide = df_features_imputed %>%
      dplyr::mutate(across(all_of(vars), ~ (.x - mean(.x, na.rm = T))/sd(.x, na.rm = T), .names = "{.col}")) 
  }
  

  # Estimate the means, SDs, and slopes
  
  ## First make a formula to use depending on the grouping level
  if (length(feat_group) == 1){
    ln_form = as.formula(paste(".x",  "day_prompt", sep = " ~ "))
  } else{
    ln_form = as.formula(paste(".x",  "prompt_num",  sep = " ~ "))
  }
  
  ## Now we estimate
  df_ml_wide = df_ml_wide %>%
    # Weekly means and SDs
    dplyr::group_by( across(all_of(c('external_id', feat_group)))) %>%
    dplyr::mutate(across(all_of(vars), ~ mean(.x, na.rm = T), .names = "{.col}_m"), 
           across(all_of(vars), ~ sd(.x, na.rm = T), .names = "{.col}_sd")) %>% 
    # Now we do linear weekly or daily trends
    mutate(across(all_of(vars),
                  ~ { tryCatch({
                        if(length(feat_group) == 1){
                          lmod = lm(.x ~ day_prompt)
                          coef(lmod)[[2]] 
                        } else{
                          lmod = lm(.x ~ prompt_num)
                          coef(lmod)[[2]]
                        }
                      }, 
                      error = function(n) {
                            ln = 0
                            ln
                      })
                    }, 
                  .names = "{.col}_lns")) %>% 
    ungroup()
  
  # Now get the unique rows
  df_ml_wide = df_ml_wide %>% distinct(across(all_of(c('external_id', feat_group))), .keep_all = T)
  return(df_ml_wide)
}


# Create feature matrix for keras
feature_mat = function(df, vars){
  #' Feature Matrix for Keras
  #' 
  #' Generates a 4D matrix from data in extra wide 
  #' fromat for given variable list
  #' 
  #' @param df A dataframe in "extra wide" format, where each data is in temporally arranged columns
  #' @param vars Character vector of column names to be used as variables
  #' 
  
  # Get feature names and number of features
  features = vars
  n_features = length(features)
  
  # iterate across all features
  mat_features = c()
  for (i in features){
    mat_temp = df %>% select(starts_with(i)) %>% as.matrix()
    mat_features = c(mat_features, mat_temp)
  }
  mat_features = array(mat_features, dim = c(dim(mat_temp)[1], dim(mat_temp)[2], n_features))
  
  return(mat_features)
}


# Data imputation -------
# Function to make a predictor matrix
make_pred_matrix <- function(df, impute_vars) {
  
  # Set-up the matrix
  predict_matrix <- matrix(1, nrow = length(impute_vars), ncol = ncol(df))
  rownames(predict_matrix) <- impute_vars
  colnames(predict_matrix) <- colnames(df)
  
  # Diagonal to disable self-prediction
  diag(predict_matrix[, impute_vars]) <- 0
  # Set ID and PHQ as non-predictors
  predict_matrix[, "external_id"] <- 0
  predict_matrix[, 'phq9_sum_w'] <- 0
  predict_matrix[, 'phq2_sum_e'] <- 0
  
  # Check if any variable is constant
  constant_vars <- c()
  empty_vars <- c()
  for (var_name in impute_vars) {
    # Make a vector out of column and check for unique values
    col_data <- df[[var_name]]
    if (length(unique(na.omit(col_data))) == 1) {
      message(paste("Variable '", 
                    var_name, 
                    "' is constant. It will be ignored by MICE and filled manually.",
                    sep = ""))
      # Don't impute this variable with a model
      predict_matrix[var_name, ] <- 0
      # Don't use this variable as a predictor
      predict_matrix[, var_name] <- 0
      constant_vars <- c(constant_vars, var_name)
    }
    # Check if any variable is empty
    if (length(na.omit(col_data)) == 0) {
      message(paste("Variable '", var_name, "' is empty", sep = ""))
      # Don't impute this variable with a model
      predict_matrix[var_name, ] <- 0
      # Don't use this variable as a predictor
      predict_matrix[, var_name] <- 0
      empty_vars <- c(empty_vars, var_name)
    }
  }
  # Return a list and matrix to use in the MICE process
  return(list(pred_matrix = predict_matrix, constant_vars = constant_vars, empty_vars=empty_vars))
}

impute_subject <- function(df_features, sub_id, impute_vars) {
  message(paste("\n--- Processing subject:", sub_id, "---"))
  df_temp <- df_features %>% filter(external_id == sub_id)
  
  # First make the predictor matrix and get constant values
  matrix_info <- make_pred_matrix(df = df_temp, impute_vars = impute_vars)
  current_pred_matrix <- matrix_info$pred_matrix
  constant_vars_to_fill <- matrix_info$constant_vars
  empty_vars_to_ignore <- matrix_info$empty_vars
  
  # Manually fill NA values in the constant columns identified earlier
  if (length(constant_vars_to_fill) > 0) {
    for (const_var in constant_vars_to_fill) {
      # Check if the column actually has missing values to fill
      if (any(is.na(df_temp[[const_var]]))) {
        constant_value <- unique(na.omit(df_temp[[const_var]]))
        # Fill the NAs in the result data frame
        df_temp[[const_var]][is.na(df_temp[[const_var]])] <- constant_value
        message(paste("Manually imputed constant variable '", const_var, "' with value: ", constant_value, sep = ""))
      }
    }
  }
  
  # Run MICE Imputation
  data_imputed <- mice(
    data = df_temp,
    m = 5,
    maxit = 5,
    method = 'rf',
    predictorMatrix = current_pred_matrix,
    blocks = as.list(rownames(current_pred_matrix)),
    seed = 666,
    printFlag = F
  )
  
  # Make a complete DF
  df_temp_imp <- complete(data_imputed)
  return(df_temp_imp)
}


