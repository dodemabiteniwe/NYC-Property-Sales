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


# Step 3: Exploratory Data Analysis (EDA)
nyc_data_filtered <- nyc_data %>% filter(SALE.PRICE >= 100 & SALE.PRICE <= 5000000)

# Sale Price Distribution with Density, Mean, and Median
ggplot(nyc_data_filtered, aes(x = SALE.PRICE)) +
  geom_histogram(aes(y = ..density..), bins = 50, fill = "blue", alpha = 0.5) +
  geom_density(color = "red", size = 1) +
  geom_vline(aes(xintercept = mean(SALE.PRICE, na.rm = TRUE)), color = "green", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = median(SALE.PRICE, na.rm = TRUE)), color = "blue", linetype = "dashed", size = 1) +
  labs(title = "Distribution of Sale Price", x = "Sale Price", y = "Density")

# Box Plot of Sale Price by Borough
ggplot(nyc_data_filtered, aes(x = BOROUGH, y = SALE.PRICE)) +
  geom_boxplot(fill = "lightblue") +
  coord_cartesian(ylim = c(100, 5000000)) + 
  labs(title = "Sale Price by Borough", x = "Borough", y = "Sale Price")

# Bar Plot for House Price by Top 10 Neighborhoods
nyc_data_filtered %>%
  group_by(NEIGHBORHOOD) %>%
  summarise(avg_price = mean(SALE.PRICE, na.rm = TRUE)) %>%
  arrange(desc(avg_price)) %>%
  head(10) %>%
  ggplot(aes(x = reorder(NEIGHBORHOOD, -avg_price), y = avg_price)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Top 10 Neighborhoods by House Price", x = "Neighborhood", y = "Average Sale Price") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Correlation Matrix Heatmap
numeric_columns <- nyc_data_filtered %>%
  select_if(is.numeric) %>%
  cor(use = "complete.obs")
corrplot(numeric_columns, method = "color", tl.cex = 0.7, number.cex = 0.7, addCoef.col = "black", round = 2)

# Step 4: Machine Learning

# One Hot Encoding
nyc_data_encoded <- nyc_data_filtered %>%
  model.matrix(~.-1, .) %>%
  as.data.frame()

# Two Approaches: Original Sale Price & Log Sale Price
nyc_data_encoded$LOG.SALE.PRICE <- log(nyc_data_encoded$SALE.PRICE)

# Split into training and testing sets for both original and log Sale Price
set.seed(123)
trainIndex <- createDataPartition(nyc_data_encoded$SALE.PRICE, p = .8, list = FALSE)
train_data_orig <- nyc_data_encoded[trainIndex, ]
test_data_orig <- nyc_data_encoded[-trainIndex, ]

train_data_log <- train_data_orig %>% mutate(SALE.PRICE = LOG.SALE.PRICE)
test_data_log <- test_data_orig %>% mutate(SALE.PRICE = LOG.SALE.PRICE)

train_control <- trainControl(method = "cv", number = 10)

# Function to Evaluate Models
evaluate_model <- function(model, train_data, test_data, response_var) {
  model_fit <- train(as.formula(paste(response_var, "~ .")), data = train_data, method = model, trControl = train_control)
  predictions <- predict(model_fit, newdata = test_data)
  if (response_var == "LOG.SALE.PRICE") {
    predictions <- exp(predictions) # Reverse log for comparison
  }
  return(RMSE(predictions, test_data$SALE.PRICE))
}

# List of Models
models <- c("lm", "rf", "svmRadial", "gbm", "xgbTree")

# Evaluate models for original sale price
rmse_results_orig <- sapply(models, evaluate_model, train_data = train_data_orig, test_data = test_data_orig, response_var = "SALE.PRICE")
names(rmse_results_orig) <- models
cat("RMSE for Original Sale Price:\n")
print(rmse_results_orig)

# Evaluate models for log sale price
rmse_results_log <- sapply(models, evaluate_model, train_data = train_data_log, test_data = test_data_log, response_var = "LOG.SALE.PRICE")
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
