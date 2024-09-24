library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(data.table)
# Install necessary packages
install.packages(c("xgboost", "e1071"))

# Load additional libraries
library(xgboost)
library(e1071)

# Unzip the downloaded file
unzip("nyc-property-sales.zip")

# Step 2: Load the data
nyc_sales <- fread("nyc-rolling-sales.csv")

# View the structure of the dataset
str(nyc_sales)

# Step 3: Data Cleaning
# Remove columns that are irrelevant for modeling (e.g., 'EASE-MENT' and 'Unnamed')
nyc_sales_clean <- nyc_sales %>%
  select(-`EASE-MENT`, -`Unnamed: 0`)

# Convert the SALE PRICE column to numeric
nyc_sales_clean$`SALE PRICE` <- as.numeric(gsub(",", "", nyc_sales_clean$`SALE PRICE`))

# Remove rows where sale price is missing or zero
nyc_sales_clean <- nyc_sales_clean %>%
  filter(`SALE PRICE` > 0)

# Convert categorical variables to factors
nyc_sales_clean$`NEIGHBORHOOD` <- as.factor(nyc_sales_clean$`NEIGHBORHOOD`)
nyc_sales_clean$`BUILDING CLASS CATEGORY` <- as.factor(nyc_sales_clean$`BUILDING CLASS CATEGORY`)

# Step 4: Exploratory Data Analysis
# Summary statistics
summary(nyc_sales_clean)

# Plot distribution of sale prices
ggplot(nyc_sales_clean, aes(x = `SALE PRICE`)) +
  geom_histogram(bins = 30, fill = "blue", color = "black") +
  ggtitle("Distribution of Sale Prices") +
  xlab("Sale Price") +
  ylab("Frequency")

# Boxplot of sale price by borough
ggplot(nyc_sales_clean, aes(x = `BOROUGH`, y = `SALE PRICE`)) +
  geom_boxplot() +
  ggtitle("Sale Price by Borough") +
  xlab("Borough") +
  ylab("Sale Price")





# Log transformation of sale price to normalize it
nyc_sales_clean$log_sale_price <- log(nyc_sales_clean$`SALE PRICE`)

# Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(nyc_sales_clean$log_sale_price, p = 0.8, list = FALSE)
train_data <- nyc_sales_clean[train_index, ]
test_data <- nyc_sales_clean[-train_index, ]

# Prepare features and target variable
train_x <- train_data %>% select(`GROSS SQUARE FEET`, `LAND SQUARE FEET`, `NEIGHBORHOOD`, `BUILDING CLASS CATEGORY`)
train_y <- train_data$log_sale_price

test_x <- test_data %>% select(`GROSS SQUARE FEET`, `LAND SQUARE FEET`, `NEIGHBORHOOD`, `BUILDING CLASS CATEGORY`)
test_y <- test_data$log_sale_price

# One-hot encoding categorical variables for xgboost
train_matrix <- model.matrix(~.-1, data=train_x)
test_matrix <- model.matrix(~.-1, data=test_x)

# Step 6: Multiple Model Training

# 1. Linear Regression
lm_model <- lm(log_sale_price ~ `GROSS SQUARE FEET` + `LAND SQUARE FEET` + `NEIGHBORHOOD` + `BUILDING CLASS CATEGORY`, data=train_data)
lm_pred <- predict(lm_model, test_data)
lm_rmse <- sqrt(mean((test_y - lm_pred)^2))

# 2. Random Forest
rf_model <- randomForest(log_sale_price ~ `GROSS SQUARE FEET` + `LAND SQUARE FEET` + `NEIGHBORHOOD` + 
                           `BUILDING CLASS CATEGORY`, data = train_data, ntree = 100)
rf_pred <- predict(rf_model, test_data)
rf_rmse <- sqrt(mean((test_y - rf_pred)^2))

# 3. Gradient Boosting (xgboost)
xgb_model <- xgboost(data = train_matrix, label = train_y, nrounds = 100, objective = "reg:squarederror", verbose = 0)
xgb_pred <- predict(xgb_model, test_matrix)
xgb_rmse <- sqrt(mean((test_y - xgb_pred)^2))

# 4. Support Vector Machine (SVM)
svm_model <- svm(log_sale_price ~ ., data = train_data)
svm_pred <- predict(svm_model, test_data)
svm_rmse <- sqrt(mean((test_y - svm_pred)^2))

# Convert predicted log prices to original sale prices
lm_price_pred <- exp(lm_pred)
rf_price_pred <- exp(rf_pred)
xgb_price_pred <- exp(xgb_pred)
svm_price_pred <- exp(svm_pred)

# Step 7: Model Comparison
model_comparison <- data.frame(
  Model = c("Linear Regression", "Random Forest", "XGBoost", "SVM"),
  RMSE = c(lm_rmse, rf_rmse, xgb_rmse, svm_rmse)
)

print(model_comparison)

# Step 8: Choose the best model based on RMSE and predict house prices
best_model <- model_comparison[which.min(model_comparison$RMSE), "Model"]
print(paste("Best Model:", best_model))

# Visualize Actual vs Predicted Prices for the best model
if(best_model == "Linear Regression") {
  test_data$predicted_price <- lm_price_pred
} else if(best_model == "Random Forest") {
  test_data$predicted_price <- rf_price_pred
} else if(best_model == "XGBoost") {
  test_data$predicted_price <- xgb_price_pred
} else if(best_model == "SVM") {
  test_data$predicted_price <- svm_price_pred
}

# Plot Actual vs Predicted Prices
ggplot(test_data, aes(x = `SALE PRICE`, y = predicted_price)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  ggtitle(paste("Actual vs Predicted Sale Prices using", best_model)) +
  xlab("Actual Sale Price") +
  ylab("Predicted Sale Price")






