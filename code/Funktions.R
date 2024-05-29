# categories: Names of categories
# N: Total sample size
# class_ratios: A proportional array of categories, which needs to be summed up to 1
# num_continuous: Number of continuous variables to generate
# num_categorical: Number of categorical variables to generate
# noise_level: Noise level


generate_imbalanced_multiclass_data <- function(categories = c("aa","bb","cc","dd"), N = 1000, class_ratios = c(0.05, 0.15, 0.5, 0.3), num_continuous = 3, num_categorical = 2, noise_level = 0.1) {

  if(sum(class_ratios) != 1) {
    stop("The sum of class ratios must be 1.")
  }
  if (length(categories) != length(class_ratios)) {
    stop("Length of 'categories' must match length of 'class_ratios'.")
  }
  class_counts <- ceiling(N * class_ratios)
  class_counts[length(class_counts)] <- N - sum(class_counts[1:(length(class_counts) - 1)])
  targets <- as.factor(sample(categories, N, replace = TRUE, prob = class_ratios))
  data_indices <- sample(1:N)
  targets <- targets[data_indices]
  df <- data.frame(targets)
  
  for (i in 1:num_continuous) {
    df[[paste("cont_var", i, sep = "_")]] <- rnorm(N, mean = 50, sd = 10) +
      rnorm(N, mean = 0, sd = noise_level)
    df[[paste("cont_var", i, sep = "_")]] <- df[[paste("cont_var", i, sep = "_")]][data_indices]
  }
  
  for (i in 1:num_categorical) {
    levels <- sample(2:9, 1)  
    df[[paste("cat_var", i, sep = "_")]] <- sample(LETTERS[1: levels], N, replace = TRUE)
    df[[paste("cat_var", i, sep = "_")]] <- df[[paste("cat_var", i, sep = "_")]][data_indices]
    num_noise <- floor(N * noise_level)
    noise_indices <- sample(1:N, num_noise, replace = FALSE)
    df[[paste("cat_var", i, sep = "_")]][noise_indices] <- sapply(df[[paste("cat_var", i, sep = "_")]][noise_indices], function(old_label) {
      sample(setdiff(LETTERS[1: levels], old_label), 1)})
    df[[paste("cat_var", i, sep = "_")]] <- as.factor(df[[paste("cat_var", i, sep = "_")]])
  }
  
  df
}


# -------------------------------------------------------------------------
F1_Score <- function(true_labels, predictions) {
  conf_matrix <- table(true_labels, predictions)
  precision <- diag(conf_matrix) / rowSums(conf_matrix)
  recall <- diag(conf_matrix) / colSums(conf_matrix)
  f1 <- 2 * (precision * recall) / (precision + recall)
  return(mean(f1, na.rm = TRUE))
}


smote_random_forest <- function(data, target_column, n_trees = 100) {
  data[[target_column]] <- as.factor(data[[target_column]])
  dummies <- dummyVars(as.formula(paste(target_column, "~ .")), data = data)
  data_transformed <- data.frame(predict(dummies, newdata = data))
  data_transformed[[target_column]] <- data[[target_column]]
  smote_data <- SmoteClassif(as.formula(paste(target_column, "~ .")), data_transformed, C.perc = C.perc)
  rf_model <- randomForest(as.formula(paste(target_column, "~ .")), data = smote_data, ntree = n_trees)
  return(rf_model)
}


smote_adaboost <- function(data, target_column, n_iter = 100) {
  data[[target_column]] <- as.factor(data[[target_column]])
  dummies <- dummyVars(as.formula(paste(target_column, "~ .")), data = data)
  data_transformed <- data.frame(predict(dummies, newdata = data))
  data_transformed[[target_column]] <- data[[target_column]]
  smote_data <- SmoteClassif(as.formula(paste(target_column, "~ .")), data_transformed, C.perc = C.perc)
  ada_model <- boosting(as.formula(paste(target_column, "~ .")), data = smote_data, boos = TRUE, mfinal = n_iter)
  return(ada_model)
}


smote_svm <- function(data, target_column, cost = 1, gamma = 0.1) {
  data[[target_column]] <- as.factor(data[[target_column]])
  dummies <- dummyVars(as.formula(paste(target_column, "~ .")), data = data)
  data_transformed <- data.frame(predict(dummies, newdata = data))
  data_transformed[[target_column]] <- data[[target_column]]
  smote_data <- SmoteClassif(as.formula(paste(target_column, "~ .")), data_transformed, C.perc = C.perc)
  svm_model <- svm(as.formula(paste(target_column, "~ .")), data = smote_data, cost = cost, gamma = gamma)
  return(svm_model)
}


predict_model <- function(model, new_data, model_type = "else") {
  dummies <- dummyVars(~ ., data = new_data)
  new_data_transformed <- data.frame(predict(dummies, newdata = new_data))
  
  if (model_type == "adaboost") {
    predictions <- predict(model, newdata = new_data_transformed)$class
  } else {
    predictions <- predict(model, newdata = new_data_transformed)
  }
  return(predictions)
}


# -------------------------------------------------------------------------
test_function <- function(data, target_column, split_ratio = 0.7, n_iter = 100, n_trees = 100) {
  if (!is.factor(data[[target_column]])) {
    data[[target_column]] <- as.factor(data[[target_column]])
  }
  
  if (any(is.na(data))) {
    data <- na.omit(data)
  }
  
  train_indices <- sample(nrow(data), floor(nrow(data) * split_ratio))
  train_set <- data[train_indices, ]
  test_set <- data[-train_indices, ]
  
  smote_rf_model <- smote_random_forest(train_set, target_column, n_trees)
  smote_ada_model <- smote_adaboost(train_set, target_column, n_iter)
  smote_svm_model <- smote_svm(train_set, target_column)
  rf_model <- randomForest(as.formula(paste(target_column, "~ .")), data = train_set, ntree = n_trees)
  balanced_rf_model <- randomForest(as.formula(paste(target_column, "~ .")), data = train_set, ntree = n_trees, strata = train_set[[target_column]], sampsize = rep(min(table(train_set[[target_column]])), length(unique(train_set[[target_column]]))))
  ada_model <- boosting(as.formula(paste(target_column, "~ .")), data = train_set, boos = TRUE, mfinal = n_iter)
  svm_model <- svm(as.formula(paste(target_column, "~ .")), data = train_set, cost = 1, gamma = 0.1)
  
  
  test_labels <- test_set[[target_column]]
  smote_rf_predictions <- predict_model(smote_rf_model, select(test_set , -target_column), model_type = "random_forest")
  smote_ada_predictions <- predict_model(smote_ada_model, select(test_set , -target_column), model_type = "adaboost")
  rf_predictions <- predict(rf_model, newdata = select(test_set , -target_column))
  balanced_rf_predictions <- predict(balanced_rf_model, newdata = select(test_set , -target_column))
  ada_predictions <- predict(ada_model, newdata = select(test_set , -target_column))$class
  svm_predictions <- predict(svm_model, select(test_set , -target_column))
  smote_svm_predictions <- predict_model(smote_svm_model, select(test_set , -target_column))
  
  f1_scores <- c(
    RF = F1_Score(test_labels, rf_predictions),
    Balanced_RF = F1_Score(test_labels, balanced_rf_predictions),
    SMOTE_RF = F1_Score(test_labels, smote_rf_predictions),
    SVM = F1_Score(test_labels, svm_predictions),
    SMOTE_SVM = F1_Score(test_labels, smote_svm_predictions),
    Ada = F1_Score(test_labels, ada_predictions),
    SMOTE_Ada = F1_Score(test_labels, smote_ada_predictions)
  )
  
  return(f1_scores)
}




