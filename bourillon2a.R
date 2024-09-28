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
                 mean(`LAND SQUARE FEET`) + 3*sd(`LAND SQUARE FEET`)))

corr_after_outliers2 <- cor(nyc_data %>% select_if(is.numeric), use = "complete.obs")

#Combine all correlations into one table
correlation_table2 <- data.frame(
  Variable = rownames(corr_default2),
  Corr_Default = round(corr_default2[, "SALE PRICE"], 5),
  Corr_After_Missing = round(corr_after_missing2[, "SALE PRICE"], 5),
  Corr_After_Outliers = round(corr_after_outliers2[, "SALE PRICE"], 5)
)
print(correlation_table2)



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
nyc_data_encoded$log_sale_price <- log(nyc_data_encoded$sale_price)

# Split into training and testing sets for both original and log Sale Price
set.seed(123)
trainIndex <- createDataPartition(nyc_data_encoded$sale_price, p = .8, list = FALSE)
train_data_orig <- nyc_data_encoded[trainIndex, ]
test_data_orig <- nyc_data_encoded[-trainIndex, ]

train_data_log <- train_data_orig %>% mutate(sale_price = log_sale_price)
test_data_log <- test_data_orig %>% mutate(sale_price = log_sale_price)

train_control <- trainControl(method = "cv", number = 2)

# Function to Evaluate Models
evaluate_model <- function(model, train_data, test_data, response_var) {
  model_fit <- train(as.formula(paste(response_var, "~ .")), data = train_data, method = model, trControl = train_control)
  predictions <- predict(model_fit, newdata = test_data)
  if (response_var == "log_sale_price") {
    predictions <- exp(predictions) # Reverse log for comparison
  }
  return(RMSE(predictions, test_data$sale_price))
}

# List of Models
models <- c("lm", "rf", "svmRadial", "gbm", "xgbTree")

# Evaluate models for original sale price
rmse_results_orig <- sapply(models, evaluate_model, train_data = train_data_orig, test_data = test_data_orig, response_var = "sale_price")
names(rmse_results_orig) <- models
cat("RMSE for Original Sale Price:\n")
print(rmse_results_orig)

# Evaluate models for log sale price
rmse_results_log <- sapply(models, evaluate_model, train_data = train_data_log, test_data = test_data_log, response_var = "log_sale_price")
 names(rmse_results_log) <- models
cat("RMSE for Log Sale Price:\n")
print(rmse_results_log)

# Compare and choose the best approach
best_model_orig <- names(rmse_results_orig)[which.min(rmse_results_orig)]
best_model_log <- names(rmse_results_log)[which.min(rmse_results_log)]

cat("Best Model for Original Sale Price:", best_model_orig, "\n")
cat("Best Model for Log Sale Price:", best_model_log, "\n")

if (min(rmse_results_orig) < min(rmse_results_log)) {
  cat("Original Sale Price approach is better with model:", best_model_orig, "\n")
} else {
  cat("Log Sale Price approach is better with model:", best_model_log, "\n")
}
