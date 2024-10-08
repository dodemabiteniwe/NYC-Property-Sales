# Load necessary libraries
library(data.table)
library(tidyverse)
library(lubridate)
library(caret)
library(e1071)
library(ggplot2)
library(corrplot)
library(scales)
library(ggpubr)
library(xgboost)
library(gbm)
library(janitor)

# Step 1: Data Loading
temp <- tempfile()
download.file("https://raw.githubusercontent.com/dodemabiteniwe/NYC-Property-Sales/main/nyc-rolling-sales.csv.zip", temp)
unzip(temp, files = "nyc-rolling-sales.csv", exdir = "data")
nyc_data <- fread("data/nyc-rolling-sales.csv")



# Step 2: Data Cleaning
nyc_data <- nyc_data %>%
  mutate(`RESIDENTIAL UNITS` = as.numeric(gsub(",|-", "", `RESIDENTIAL UNITS`)),
         `COMMERCIAL UNITS` = as.numeric(gsub(",|-", "", `COMMERCIAL UNITS`)),
         `TOTAL UNITS` = as.numeric(gsub(",|-", "", `TOTAL UNITS`)),
         `LAND SQUARE FEET` = as.numeric(gsub(",|-", "", `LAND SQUARE FEET`)),
         `GROSS SQUARE FEET` = as.numeric(gsub(",|-", "", `GROSS SQUARE FEET`)),
         `SALE PRICE` = as.numeric(gsub(",|-", "", `SALE PRICE`)),
         BLOCK = as.numeric(gsub(",|-", "", BLOCK)),
         `YEAR BUILT` = as.numeric(gsub(",|-", "", `YEAR BUILT`)),
         LOT = as.numeric(gsub(",|-", "", LOT)),
         `SALE DATE` = as.Date(`SALE DATE`, format = "%m/%d/%Y"),
         sale_year = year(`SALE DATE`),
         sale_month = month(`SALE DATE`, label = TRUE),
         building_age = year(`SALE DATE`) - `YEAR BUILT`,
         BOROUGH = as.factor(c('1' = 'Manhattan', '2' = 'Bronx', '3' = 'Brooklyn', 
                               '4' = 'Queens', '5' = 'Staten Island')[BOROUGH]),
         across(c("BOROUGH", "NEIGHBORHOOD", "BUILDING CLASS CATEGORY", 
                  "TAX CLASS AT PRESENT", "BUILDING CLASS AT TIME OF SALE", 
                  "TAX CLASS AT TIME OF SALE"), as.factor)) %>%
  select(-c(`BUILDING CLASS AT PRESENT`, `ZIP CODE`, ADDRESS, `APARTMENT NUMBER`,`SALE DATE`, V1, `EASE-MENT`))

str(nyc_data)
summary(nyc_data)

# Default correlations before any cleaning
corr_default2 <- nyc_data %>%
  select(where(is.numeric)) %>%  # Select only numeric variables
  cor(use = "complete.obs")

#-------------------------------------------Handle  missing values------------------------
# Remove rows where sale price is missing or zero
nyc_data <- nyc_data %>%
  filter(`SALE PRICE` > 0)%>%
  mutate(building_age = ifelse(building_age < 0 | is.na(building_age), NA, building_age))

# Replace empty values with NA
nyc_data2 <- nyc_data%>%
  mutate(across(where(is.factor), as.character)) %>%
  mutate(across(where(is.character), ~na_if(.x, "")))

# Calculate the missing value rate for each column
na_rate2 <- nyc_data2 %>%
  summarise(across(everything(), ~mean(is.na(.)) * 100))%>%t()
colnames(na_rate2) <- "NA_Percentage"
na_rate2 <- as.data.frame(na_rate2)
print(na_rate2)

# Drop rows with any missing values (NA) and Remove duplicate rows
nyc_data <- nyc_data %>%
  drop_na() %>%
  distinct() %>%
  filter(`COMMERCIAL UNITS` + `RESIDENTIAL UNITS` == `TOTAL UNITS`) %>%
  filter(`TOTAL UNITS` != 0 &`YEAR BUILT` != 0 & `LAND SQUARE FEET` != 0 & `GROSS SQUARE FEET` != 0)

corr_after_missing2 <- cor(nyc_data %>% select_if(is.numeric), use = "complete.obs")

#-----------------------------outlier removal-----------------------------------------
nyc_data <- nyc_data %>%
  filter(`TOTAL UNITS` != 2261 & `TOTAL UNITS` != 1866) %>%
  filter(between(`GROSS SQUARE FEET`, mean(`GROSS SQUARE FEET`) - 3*sd(`GROSS SQUARE FEET`), 
                 mean(`GROSS SQUARE FEET`) + 3*sd(`GROSS SQUARE FEET`))) %>%
  filter(between(`LAND SQUARE FEET`, mean(`LAND SQUARE FEET`) - 3*sd(`LAND SQUARE FEET`), 
                 mean(`LAND SQUARE FEET`) + 3*sd(`LAND SQUARE FEET`)))%>%
  filter(between(`SALE PRICE`, mean(`SALE PRICE`) - 3*sd(`SALE PRICE`), 
                 mean(`SALE PRICE`) + 3*sd(`SALE PRICE`)))

corr_after_outliers2 <- cor(nyc_data %>% select_if(is.numeric), use = "complete.obs")

#Combine all correlations into one table
correlation_table2 <- data.frame(
  Variable = rownames(corr_default2),
  Corr_Default = round(corr_default2[, "SALE PRICE"], 5),
  Corr_After_Missing = round(corr_after_missing2[, "SALE PRICE"], 5),
  Corr_After_Outliers = round(corr_after_outliers2[, "SALE PRICE"], 5)
)
print(correlation_table2)

shapiro.test(nyc_data$`SALE PRICE`)
ggqqplot(nyc_data_encoded1$log_sale_price)
#######################################################################
#------Exploratory Data Analysis (EDA)----------------------------
######################################################

nyc_data_filtered <- nyc_data %>% filter(`SALE PRICE` >= 100 & `SALE PRICE` <= 5000000)

# Sale Price Distribution with Density, Mean, and Median
ggplot(nyc_data_filtered, aes(x = `SALE PRICE`)) +
  geom_histogram(aes(y = after_stat(density)), bins = 50, fill = "skyblue", color = "black", alpha = 0.7) +
  geom_density(color = "red", linewidth = 1) +
  geom_vline(aes(xintercept = mean(`SALE PRICE`, na.rm = TRUE)), color = "Violet", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = median(`SALE PRICE`, na.rm = TRUE)), color = "blue", linetype = "dashed", size = 1) +
  labs(title = "Distribution of Sale Price", x = "Sale Price", y = "Density")+
  theme_minimal() +
  annotate("text", x = mean(nyc_data_filtered$`SALE PRICE`, na.rm = TRUE), y = 0.0000016, label = "Mean", color ="Violet" , angle = 90, vjust = 1.5) +
  annotate("text", x = median(nyc_data_filtered$`SALE PRICE`, na.rm = TRUE), y = 0.0000016, label ="Median", color = "blue", angle = 90, vjust = -1)
ggsave("figs/densitySale.png")

# Box Plot of Sale Price by Borough

# Filter data to include only SALE PRICE values between 100 000 and 5,000,000
nyc_data_filtered2 <- nyc_data %>% filter(`SALE PRICE` >= 100000 & `SALE PRICE` <= 5000000)
ggplot(nyc_data_filtered2, aes(x = BOROUGH, y = `SALE PRICE`, fill = BOROUGH)) +
  geom_boxplot() +labs(title = "Box Plot of Sale Price by Boroughs",x = "Borough",y = "Sale Price")+
  scale_fill_brewer(palette = "Set3")+ 
  theme_minimal()+theme(legend.position = "none") 
ggsave("figs/BxPlotSaleBor.png")

str(nyc_data_filtered2)
# Box Plot of Sale Price by Month
ggplot(nyc_data_filtered2, aes(x = sale_month, y = `SALE PRICE`, fill = sale_month)) +
  geom_boxplot() +
  labs(title = "Box Plot of Sale Price by Month",
       x = "Month",
       y = "Sale Price") +
  scale_fill_brewer(palette = "Set3") +  
  theme_minimal() +
  theme(legend.position = "none")  
ggsave("figs/BxPlotByMonth.png")

# 6. Sale Count by Month
sales_count_by_month2 <- nyc_data%>%
  group_by(sale_month) %>%
  summarize(Sale_Count = n(), .groups = 'drop')

ggplot(sales_count_by_month2, aes(x = sale_month, y = Sale_Count, fill = sale_month)) +
  geom_bar(stat = "identity") +
  labs(title = "Sale Count by Month",
       x = "Month",
       y = "Number of Sales") +
  scale_fill_brewer(palette = "Set3") +  
  theme_minimal() +
  theme(legend.position = "none")
ggsave("figs/SaleByMonth.png")

# 5. Plot Average Land Square Feet of Properties in Each Borough
nyc_data_filtered2 %>%
  group_by(BOROUGH) %>%
  summarise(avg_land_sqft = mean(`LAND SQUARE FEET`, na.rm = TRUE), .groups = 'drop') %>%
  ggplot(aes(x = BOROUGH, y = avg_land_sqft, size = avg_land_sqft, color = BOROUGH)) +
  geom_point(alpha = 0.7) + scale_size_continuous(range = c(8, 15))+
  labs(title = "Average Land Square Feet by Borough", x = "Borough", y = "Average Land Square Feet")+
  theme_minimal() + theme(legend.position = "right")
ggsave("figs/avg_land_sqft.png")

# 6. Bar Plot for House Price by Top 10 Neighborhoods
# Group by neighborhood and calculate the number of sales and average sale price
nyc_data_filtered2%>%
  group_by(NEIGHBORHOOD) %>%
  summarize(Number_of_Sales = n(),
            Avg_Sale_Price = mean(`SALE PRICE`, na.rm = TRUE), .groups = 'drop')%>%
  arrange(desc(Number_of_Sales)) %>%
  head(10)%>%
  ggplot(aes(x = reorder(NEIGHBORHOOD, Number_of_Sales), y = Avg_Sale_Price, fill = NEIGHBORHOOD)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  coord_flip() +  # Flip coordinates for better readability
  labs(title = "Top 10 Neighborhoods by Number of Sales and Average Sale Price",
       x = "Neighborhood",
       y = "Average Sale Price") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



# Correlation Matrix Heatmap
nyc_data_reduce <- nyc_data %>%
  select(-c(LOT, `YEAR BUILT`, sale_year, building_age,`TOTAL UNITS`,`RESIDENTIAL UNITS`))

nyc_data_reduce2 <- nyc_data %>%
  select(-c(LOT, `YEAR BUILT`, sale_year, building_age,`TOTAL UNITS`,`RESIDENTIAL UNITS`))



numeric_columns <- nyc_data_reduce2 %>%
  select_if(is.numeric) %>%
  cor(use = "complete.obs")
corrplot(numeric_columns, method = "color", tl.cex = 0.7, number.cex = 0.7, addCoef.col = "black", round = 2)

# Step 4: Machine Learning
# Clean column names to be lowercase and without spaces
nyc_data_reduce2 <- nyc_data_reduce2 %>%
  clean_names()
colnames(nyc_data_reduce2)
nyc_data_reduce2 <- nyc_data_reduce2 %>%
  select(-c(tax_class_at_present, neighborhood, building_class_at_time_of_sale, sale_month))



# One Hot Encoding
nyc_data_encoded <- nyc_data_reduce2 %>%
  model.matrix(~.-1, .) %>%
  as.data.frame()
colnames(nyc_data_encoded)

# Split data into training and testing
set.seed(123)
trainIndex <- createDataPartition(nyc_data_encoded$sale_price, p = .8, list = FALSE)
train_data <- nyc_data_encoded[trainIndex, ]
test_data <- nyc_data_encoded[-trainIndex, ]

# Model Training with Various Algorithms
# Define the control for training
train_control <- trainControl(method = "cv", number = 2)

# Train a linear regression model
linear_model <- train(sale_price ~ ., data = train_data, method = "lm", trControl = train_control)
summary(linear_model)

# Predict on test data
linear_predictions <- predict(linear_model, newdata = test_data)

# Evaluate model performance
linear_rmse <- RMSE(linear_predictions, test_data$sale_price)
cat("Linear Regression RMSE:", linear_rmse, "\n")

# Train a Random Forest model
rf_model <- train(sale_price ~ ., data = train_data, method = "rf", trControl = train_control)
rf_predictions <- predict(rf_model, newdata = test_data)
rf_rmse <- RMSE(rf_predictions, test_data$sale_price)
cat("Random Forest RMSE:", rf_rmse, "\n")

# Train a SVM model
svm_model <- train(sale_price ~ ., data = train_data, method = "svmRadial", trControl = train_control)
 svm_predictions <- predict(svm_model, newdata = test_data)
svm_rmse <- RMSE(svm_predictions, test_data$sale_price)
cat("SVM RMSE:", svm_rmse, "\n")























# Two Approaches: Original Sale Price & Log Sale Price
#nyc_data_encoded$log_sale_price <- log(nyc_data_encoded$sale_price)
nyc_data_encoded1 <- nyc_data_encoded%>% mutate(log_sale_price = log(sale_price))
#nyc_data_encoded1 <- nyc_data_encoded1 %>%
 # select(-sale_price)

# Split into training and testing sets for both original and log Sale Price
set.seed(123)
trainIndex <- createDataPartition(nyc_data_encoded$sale_price, p = .8, list = FALSE)
train_data_orig <- nyc_data_encoded[trainIndex, ]
test_data_orig <- nyc_data_encoded[-trainIndex, ]

train_data_log <- nyc_data_encoded1[trainIndex, ]
test_data_log <- nyc_data_encoded1[-trainIndex, ]


#train_data_log <- train_data_orig %>% mutate(sale_price = log_sale_price)
#test_data_log <- test_data_orig %>% mutate(sale_price = log_sale_price)

train_control <- trainControl(method = "cv", number = 2)

# Function to Evaluate Models
evaluate_model <- function(model, train_data, test_data, response_var) {
  if (response_var == "log_sale_price") {
    train_saleprice <- train_data$sale_price
    test_saleprice <- test_data$sale_price # Reverse log for comparison
    train_data <- train_data%>%select(-sale_price)
    test_data <- test_data%>%select(-sale_price)
  }
  start_time <- Sys.time()
  model_fit <- train(as.formula(paste(response_var, "~ .")), data = train_data, method = model, trControl = train_control, preProcess = c("center", "scale","nzv"))
  # Predictions for train and test datasets
  train_predictions <- predict(model_fit, newdata = train_data)
  test_predictions <- predict(model_fit, newdata = test_data)
  #Execution time
  end_time <- Sys.time()
  exec_time <- round(difftime(end_time, start_time, units = "secs"), 2)
  
  if (response_var == "log_sale_price") {
    train_predictions <- exp(train_predictions)
    test_predictions <- exp(test_predictions) # Reverse log for comparison
    # Train metrics
    train_rmse <- RMSE(train_predictions, train_saleprice)
    train_mae <- MAE(train_predictions, train_saleprice)
    train_score <- cor(train_predictions, train_saleprice)^2
    
    # Test metrics
    test_rmse <- RMSE(test_predictions, test_saleprice)
    test_mae <- MAE(test_predictions, test_saleprice)
    test_score <- cor(test_predictions, test_saleprice)^2
  }else{
  # Train metrics
  train_rmse <- RMSE(train_predictions, train_data$sale_price)
  train_mae <- MAE(train_predictions, train_data$sale_price)
  train_score <- cor(train_predictions, train_data$sale_price)^2
  
  # Test metrics
  test_rmse <- RMSE(test_predictions, test_data$sale_price)
  test_mae <- MAE(test_predictions, test_data$sale_price)
  test_score <- cor(test_predictions, test_data$sale_price)^2
  }
  # Return a list of all metrics including execution time
  return(data.frame(
    Model = model,
    Train_Score = round(train_score, 5),
    Test_Score = round(test_score, 5),
    Train_RMSE = round(train_rmse, 5),
    Test_RMSE = round(test_rmse, 5),
    Train_MAE = round(train_mae, 5),
    Test_MAE = round(test_mae, 5),
    Execution_Time_Secs = exec_time
  ))
}

# List of Models
models <- c("lm", "ranger", "svmRadial", "gbm", "xgbTree")

# Evaluate models for original sale price
results_orig <- lapply(models, evaluate_model, train_data = train_data_orig, test_data = test_data_orig, response_var = "sale_price")
 
 # Combine results into data frames
results_orig_df <- do.call(rbind, results_orig)

# Display results
cat("Results for Original Sale Price:\n")
print(results_orig_df)

# Comparing Results
cat("\nBest Model for Original Sale Price Approach:\n")
best_model_orig <- results_orig_df[which.min(results_orig_df$Test_RMSE),]
print(best_model_orig)






# Set up training control with cross-validation
train_control <- trainControl(
  method = "cv",              # Cross-validation method
  number = 2                # 5-fold cross-validation
)

# Set up the grid for hyperparameter tuning
tune_grid <- expand.grid(
  mtry = c(3, 5, 7, 9),       # Number of variables randomly sampled at each split
  ntree = c(100, 300, 500),    # Number of trees
  maxnodes = c(30, 50, 100)    # Maximum terminal nodes in trees
)

tune_grid

# Train Random Forest model with hyperparameter tuning
rf_optimized1 <- train(
  sale_price ~ .,              # Formula, dependent variable: SALE.PRICE
  data = train_data_orig,      # Training data
  method = "rf",               # Random Forest method
  trControl = train_control,   # Training control
  metric = "RMSE"              # Metric to optimize (Root Mean Squared Error)
)

# Print the best model's hyperparameters
print(rf_optimized1$bestTune)

# Print the results for each hyperparameter combination
print(rf_optimized1$results)

# Predict on the test data using the best tuned model
rf_predictions <- predict(rf_optimized1, newdata = test_data_orig)

# Calculate the final metrics on the test data
final_rmse <- RMSE(rf_predictions, test_data_orig$sale_price)
final_mae <- MAE(rf_predictions, test_data_orig$sale_price)

# Print the final performance metrics
cat("Final Model Performance:\n")
cat("RMSE: ", round(final_rmse, 5), "\n")
cat("MAE: ", round(final_mae, 5), "\n")

# Plot variable importance
varImpPlot(rf_optimized1)
















library(ranger)

# Set up training control with cross-validation
train_control <- trainControl(
  method = "cv",              # Cross-validation method
  number = 5,                 # 5-fold cross-validation
  verboseIter = TRUE          # Show progress
)


# Set up the grid for hyperparameter tuning including ntree, mtry, and maxnodes
tune_grid <- expand.grid(
  mtry = c(3, 5, 7, 9),          # Number of variables randomly sampled at each split
  splitrule = "variance",        # Split rule for regression
  min.node.size = c(50, 100, 150) # Equivalent to maxnodes
)

# Train Ranger Random Forest model with hyperparameter tuning
rf_optimized <- train(
  sale_price ~ .,                # Formula, dependent variable: SALE.PRICE
  data = train_data_orig,        # Training data
  method = "ranger",             # Ranger Random Forest method
  tuneGrid = tune_grid,          # Grid of hyperparameters to search
  trControl = train_control,     # Training control
  num.trees = 300,               # Number of trees
  importance = 'impurity'        # Variable importance measure
)

# Print the best model's hyperparameters
print(rf_optimized$bestTune)
# Predict on the test data using the best tuned model
rf_predictions <- predict(rf_optimized, newdata = test_data_orig)

# Calculate the final metrics on the test data
final_rmse <- RMSE(rf_predictions, test_data_orig$sale_price)
final_mae <- MAE(rf_predictions, test_data_orig$sale_price)

# Print the final performance metrics
cat("Final Model Performance:\n")
cat("RMSE: ", round(final_rmse, 5), "\n")
cat("MAE: ", round(final_mae, 5), "\n")

# Plot variable importance
varImpPlot(rf_optimized$finalModel)

var_importance <- varImp(rf_optimized)

# Print variable importance
print(var_importance)

# Plot variable importance
plot(var_importance)




























# Evaluate models for log sale price
results_log <- lapply(models, evaluate_model, train_data = train_data_log, test_data = test_data_log, response_var = "log_sale_price")

# Combine results into data frames
results_log_df <- do.call(rbind, results_log)

# Display results
cat("\nResults for Log Sale Price:\n")
print(results_log_df)

# Comparing Results
cat("\nBest Model for Log Sale Price Approach:\n")
best_model_log <- results_log_df[which.min(results_log_df$Test_RMSE),]
print(best_model_log)

# Final comparison
if (min(results_orig_df$Test_RMSE) < min(results_log_df$Test_RMSE)) {
  cat("\nOriginal Sale Price approach is better with model:\n")
  print(best_model_orig)
} else {
  cat("\nLog Sale Price approach is better with model:\n")
  print(best_model_log)
}

















