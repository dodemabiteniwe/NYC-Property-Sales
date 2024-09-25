library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(data.table)
library(lubridate)
# Install necessary packages
install.packages(c("xgboost", "e1071"))

# Load additional libraries
library(xgboost)
library(e1071)
library(tidyr)
library(reshape2)
library(janitor)

zip_url <- "https://raw.githubusercontent.com/dodemabiteniwe/NYC-Property-Sales/main/nyc-rolling-sales.csv.zip"
zip_file <- "nyc-rolling-sales2.zip"  # Local name for the ZIP file
csv_file <- "nyc-rolling-sales.csv"  # Name of the CSV inside the ZIP

# Download the ZIP file from GitHub
download.file(zip_url, destfile = zip_file, mode = "wb")

# Step 2: Extract the ZIP file
unzip(zip_file, files = csv_file)

# Step 3: Load the CSV data into R
nyc_sales <- fread(csv_file)
# View the structure of the dataset
str(nyc_sales)

#------------------------------------------------------Data Cleaning------------------------------------------------

# Remove columns that are irrelevant for modeling (e.g., 'EASE-MENT' and 'Unnamed')
nyc_sales_clean <- nyc_sales %>%
  select(-`EASE-MENT`,-`V1`,-`BUILDING CLASS AT PRESENT`, -`ZIP CODE`, -`ADDRESS`, -`APARTMENT NUMBER`)

#----------------------Clean and convert specified columns to numeric-----------------------------------

nyc_sales_clean <- nyc_sales_clean %>%
  mutate(
    `RESIDENTIAL UNITS` = as.numeric(gsub(",|-", "", `RESIDENTIAL UNITS`)),
    `COMMERCIAL UNITS` = as.numeric(gsub(",|-", "", `COMMERCIAL UNITS`)),
    `TOTAL UNITS` = as.numeric(gsub(",|-", "", `TOTAL UNITS`)),
    `LAND SQUARE FEET` = as.numeric(gsub(",|-", "", `LAND SQUARE FEET`)),
    `GROSS SQUARE FEET` = as.numeric(gsub(",|-", "", `GROSS SQUARE FEET`)),
    `SALE PRICE` = as.numeric(gsub(",|-", "", `SALE PRICE`)),
    `YEAR BUILT` = as.numeric(`YEAR BUILT`),
    `SALE DATE` = as.Date(`SALE DATE`, format = "%m/%d/%Y")
  )

# Extract year from `SALE DATE` and calculate building age
nyc_sales_clean <- nyc_sales_clean %>%
  mutate(SALE_YEAR = year(`SALE DATE`),
         SALE_MONTH  = month(`SALE DATE`,label = TRUE),
         BUILDING_AGE = SALE_YEAR - `YEAR BUILT`)


#------- Convert categorical variables to factors--------------------------

#  Replace numeric values in the BOROUGH column with borough names
nyc_sales_clean <- nyc_sales_clean %>%
  mutate(BOROUGH = case_when(
    BOROUGH == 1 ~ 'Manhattan',
    BOROUGH == 2 ~ 'Bronx',
    BOROUGH == 3 ~ 'Brooklyn',
    BOROUGH == 4 ~ 'Queens',
    BOROUGH == 5 ~ 'Staten Island',
    TRUE ~ as.character(BOROUGH)  # Handle unexpected values if any
  ))

#  Specify categorical variables and convert them to factors
categorical_columns <- c("BOROUGH","TAX CLASS AT PRESENT", "BUILDING CLASS AT TIME OF SALE", "TAX CLASS AT TIME OF SALE","NEIGHBORHOOD","BUILDING CLASS CATEGORY")

# Convert these columns to factors (categorical variables)
nyc_sales_clean <- nyc_sales_clean %>%
  mutate(across(all_of(categorical_columns), as.factor))
summary(nyc_sales_clean %>% select(all_of(categorical_columns)))

# Default correlations before any cleaning
corr_default <- nyc_sales_clean %>%
  select(where(is.numeric)) %>%  # Select only numeric variables
  cor(use = "complete.obs")





#-------------------------------------------Handle  missing values------------------------

# Convert numeric variable to numeric
#nyc_sales_clean$`SALE PRICE` <- as.numeric(gsub(",|-", "", nyc_sales_clean$`SALE PRICE`))

# Remove rows where sale price is missing or zero
nyc_sales_clean <- nyc_sales_clean %>%
  filter(`SALE PRICE` > 0)

# Replace negative or NA values with NA (or any other value depending on your requirement)
nyc_sales_clean <- nyc_sales_clean %>%
  mutate(BUILDING_AGE = ifelse(BUILDING_AGE < 0 | is.na(BUILDING_AGE), NA, BUILDING_AGE))

# View the updated dataset
#head(nyc_sales_clean %>% select(`YEAR BUILT`, `SALE DATE`, SALE_YEAR,SALE_MONTH, BUILDING_AGE))
#str(nyc_sales_clean)

# Replace empty values with NA
nyc_sales_clean2 <- nyc_sales_clean %>%
  mutate(across(where(is.factor), as.character)) %>%
  mutate(across(where(is.character), ~na_if(.x, "")))  # Replace empty strings with NA across all columns

# Calculate the missing value rate for each column
na_rate <- nyc_sales_clean2 %>%
  summarise(across(everything(), ~mean(is.na(.)) * 100))  # Calculate % of NAs in each column

# Display NA rate for each column
na_rate <- t(na_rate)  # Transpose to make it more readable
colnames(na_rate) <- "NA_Percentage"
na_rate <- as.data.frame(na_rate)

# Print NA rate for each variable
print(na_rate)

nyc_sales_clean2 <- nyc_sales_clean2 %>%
  select( -`SALE DATE`)

# Verify the remaining columns
str(nyc_sales_clean2)

# Drop rows with any missing values (NA)
nyc_sales_clean2 <- nyc_sales_clean2 %>%
  drop_na()

#  Remove duplicate rows
nyc_sales_clean2 <- nyc_sales_clean2 %>%
  distinct()

#  Verify the dataset after dropping NAs and duplicates
str(nyc_sales_clean2)  # Check the structure
summary(nyc_sales_clean2)  


#  Filter out rows where commercial + residential does not equal total units
nyc_sales_clean2 <- nyc_sales_clean2 %>%
  filter(`RESIDENTIAL UNITS` + `COMMERCIAL UNITS` == `TOTAL UNITS`)

#  Group by 'TOTAL UNITS' and sort by 'TOTAL UNITS' and 'SALE PRICE'

sales_grouped_counted <- nyc_sales_clean2 %>%
  group_by(`TOTAL UNITS`) %>%
  summarise(count = n(), avg_sale_price = mean(`SALE PRICE`, na.rm = TRUE)) %>%  # Count and calculate avg sale price
  arrange(desc(`avg_sale_price`))  # Sort by average sale price in ascending order
#  Print the result
#print(sales_grouped_counted, n=152)
#Remove rows where 'TOTAL UNITS' == 0, 'YEAR BUILT' == 0, and outliers for 'TOTAL UNITS'
nyc_sales_clean2 <- nyc_sales_clean2 %>%
  filter(`TOTAL UNITS` != 0 & `TOTAL UNITS` != 2261 & `TOTAL UNITS` != 1866&
           `YEAR BUILT` != 0 & `LAND SQUARE FEET` != 0 & `GROSS SQUARE FEET` != 0)

#  Verify the remaining data
str(nyc_sales_clean2) 
summary(nyc_sales_clean2) 


#Correlations after removing missing values (but before removing outliers)
corr_after_missing <- nyc_sales_clean2 %>%
  select(where(is.numeric)) %>%  # Select only numeric variables
  cor(use = "complete.obs")

#-----------------------------outlier removal-----------------------------------------

#Calculate mean and standard deviation for 'GROSS SQUARE FEET' and 'LAND SQUARE FEET'
mean_gross <- mean(nyc_sales_clean2$`GROSS SQUARE FEET`, na.rm = TRUE)
sd_gross <- sd(nyc_sales_clean2$`GROSS SQUARE FEET`, na.rm = TRUE)

mean_land <- mean(nyc_sales_clean2$`LAND SQUARE FEET`, na.rm = TRUE)
sd_land <- sd(nyc_sales_clean2$`LAND SQUARE FEET`, na.rm = TRUE)

#Filter out rows with values more than 2 standard deviations from the mean
nyc_sales_clean2 <- nyc_sales_clean2 %>%
  filter(`GROSS SQUARE FEET` >= (mean_gross - 3 * sd_gross) & `GROSS SQUARE FEET` <= (mean_gross + 3 * sd_gross)) %>%
  filter(`LAND SQUARE FEET` >= (mean_land - 3 * sd_land) & `LAND SQUARE FEET` <= (mean_land + 3 * sd_land))

#Correlations after outlier removal
corr_after_outliers <- nyc_sales_clean2 %>%
  select(where(is.numeric)) %>%  # Select only numeric variables
  cor(use = "complete.obs")

#Combine all correlations into one table
correlation_table <- data.frame(
  Variable = rownames(corr_default),
  Corr_Default = round(corr_default[, "SALE PRICE"], 5),
  Corr_After_Missing = round(corr_after_missing[, "SALE PRICE"], 5),
  Corr_After_Outliers = round(corr_after_outliers[, "SALE PRICE"], 5)
)

#Print the combined correlation table
print(correlation_table)

nyc_sales_clean2 <- nyc_sales_clean2 %>%
  select(-`YEAR BUILT`,-`LOT`)














#######################################################################
#-------------Exploratory Data Analysis----------------------------
######################################################


# Filter data to include only SALE PRICE values between 100 and 5,000,000
nyc_sales_filtered <- nyc_sales_clean2 %>%
  filter(`SALE PRICE` >= 100 & `SALE PRICE` <= 5000000)

# Calculate mean and median of filtered 'SALE PRICE'
mean_sale_price <- mean(nyc_sales_filtered$`SALE PRICE`, na.rm = TRUE)
median_sale_price <- median(nyc_sales_filtered$`SALE PRICE`, na.rm = TRUE)

# Create the plot
ggplot(nyc_sales_filtered, aes(x = `SALE PRICE`)) +
  geom_histogram(aes(y = ..density..), bins = 50, fill = "skyblue", color = "black", alpha = 0.7) +
  geom_density(color = "red", size = 1) +
  geom_vline(aes(xintercept = mean_sale_price), color = "blue", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = median_sale_price), color = "Violet", linetype = "dashed", size = 1) +
  labs(title = "Distribution of Sale Price (Filtered) with Mean and Median",
       x = "Sale Price",
       y = "Density") +
  theme_minimal() +
  annotate("text", x = mean_sale_price, y = 0.000002, label = paste("Mean:", round(mean_sale_price, 2)), color = "blue", angle = 90, vjust = 1.5) +
  annotate("text", x = median_sale_price, y = 0.000002, label = paste("Median:", round(median_sale_price, 2)), color = "Violet", angle = 90, vjust = 1.5)





# Filter data to include only SALE PRICE values between 100 000 and 5,000,000
nyc_sales_filtered <- nyc_sales_clean2 %>%
  filter(`SALE PRICE` >= 100000 & `SALE PRICE` <= 5000000)

# Create the box plot
ggplot(nyc_sales_filtered, aes(x = factor(BOROUGH), y = `SALE PRICE`, fill = factor(BOROUGH))) +
  geom_boxplot() +
  labs(title = "Box Plot of Sale Price by Boroughs",
       x = "Borough",
       y = "Sale Price") +
  scale_fill_brewer(palette = "Set3") +  # Optional: Use a color palette for better visualization
  theme_minimal() +
  theme(legend.position = "none")  # Hide legend for better clarity





# Remove missing values from 'SALE PRICE' before plotting
nyc_sales_clean2 <- nyc_sales_clean2 %>%
  filter(!is.na(`SALE PRICE`))

# Filter data to include only SALE PRICE values between 100 and 5,000,000
nyc_sales_filtered <- nyc_sales_clean2 %>%
  filter(`SALE PRICE` >= 100000 & `SALE PRICE` <= 5000000)

# Create the box plot
ggplot(nyc_sales_filtered, aes(x = SALE_MONTH, y = `SALE PRICE`, fill = SALE_MONTH)) +
  geom_boxplot() +
  labs(title = "Box Plot of Sale Price by Month",
       x = "Month",
       y = "Sale Price") +
  scale_fill_brewer(palette = "Set3") +  # Optional: Use a color palette for better visualization
  theme_minimal() +
  theme(legend.position = "none")  # Hide legend for better clarity


# Summarize sale prices by year
sales_by_year <- nyc_sales_filtered %>%
  group_by(SALE_YEAR) %>%
  summarize(Average_Sale_Price = mean(`SALE PRICE`), .groups = 'drop')

# Create the histogram (bar plot) with years on the x-axis
ggplot(sales_by_year, aes(x = factor(SALE_YEAR), y = Average_Sale_Price)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(title = "Average Sale Price by Year",
       x = "Year",
       y = "Average Sale Price") +
  theme_minimal()




# Count sales by month
sales_count_by_month <- nyc_sales_clean2 %>%
  group_by(SALE_MONTH) %>%
  summarize(Sale_Count = n(), .groups = 'drop')

# Create the bar plot
ggplot(sales_count_by_month, aes(x = SALE_MONTH, y = Sale_Count, fill = SALE_MONTH)) +
  geom_bar(stat = "identity") +
  labs(title = "Sale Count by Month",
       x = "Month",
       y = "Number of Sales") +
  scale_fill_brewer(palette = "Set3") +  # Optional: Use a color palette for better visualization
  theme_minimal() +
  theme(legend.position = "none")  # Hide legend for better clarity








nyc_sales_clean3 <- nyc_sales_clean2 %>%
  filter(`SALE PRICE` >= 100000 & `SALE PRICE` <= 5000000)
# Filter out rows with missing or zero values in LAND SQUARE FEET
nyc_sales_clean3 <- nyc_sales_clean3 %>%
  filter(!is.na(`LAND SQUARE FEET`) & `LAND SQUARE FEET` > 0)

# Group by Borough and calculate the average land square feet
avg_land_sqft_by_borough <- nyc_sales_clean3 %>%
  group_by(BOROUGH) %>%
  summarize(Avg_Land_SqFt = mean(`LAND SQUARE FEET`, na.rm = TRUE), .groups = 'drop')

# Create the scatter plot
ggplot(avg_land_sqft_by_borough, aes(x = BOROUGH, y = Avg_Land_SqFt, size = Avg_Land_SqFt, color = BOROUGH)) +
  geom_point(alpha = 0.7) +
  labs(title = "Average Land Square Feet by Borough",
       x = "Borough",
       y = "Average Land Square Feet",
       size = "Avg Land SqFt") +
  scale_size_continuous(range = c(8, 15)) +  # Adjust point size range
  theme_minimal() +
  theme(legend.position = "right")



#---------------Top 10 Neighborhoods by Average Sale Price-------------------------------------------


# Group by neighborhood and calculate the average sale price
avg_price_by_neighborhood <- nyc_sales_clean3 %>%
  group_by(NEIGHBORHOOD) %>%
  summarize(Avg_Sale_Price = mean(`SALE PRICE`, na.rm = TRUE), .groups = 'drop')

# Select the top 10 neighborhoods by average sale price
top_10_neighborhoods <- avg_price_by_neighborhood %>%
  arrange(desc(Avg_Sale_Price)) %>%
  head(10)

# Create the bar plot
ggplot(top_10_neighborhoods, aes(x = reorder(NEIGHBORHOOD, Avg_Sale_Price), y = Avg_Sale_Price, fill = NEIGHBORHOOD)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  coord_flip() +  # Flip coordinates for better readability
  labs(title = "Top 10 Neighborhoods by Average Sale Price",
       x = "Neighborhood",
       y = "Average Sale Price") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#----------Top 10 Neighborhoods by Number of Sales and Average Sale Price---------------------------------

# Group by neighborhood and calculate the number of sales and average sale price
sales_by_neighborhood <- nyc_sales_clean3 %>%
  group_by(NEIGHBORHOOD) %>%
  summarize(Number_of_Sales = n(),
            Avg_Sale_Price = mean(`SALE PRICE`, na.rm = TRUE), .groups = 'drop')

# Select the top 10 neighborhoods by number of sales
top_10_sales_neighborhoods <- sales_by_neighborhood %>%
  arrange(desc(Number_of_Sales)) %>%
  head(10)

# Create the bar plot for average sale price in top 10 neighborhoods by number of sales
ggplot(top_10_sales_neighborhoods, aes(x = reorder(NEIGHBORHOOD, Number_of_Sales), y = Avg_Sale_Price, fill = NEIGHBORHOOD)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  coord_flip() +  # Flip coordinates for better readability
  labs(title = "Top 10 Neighborhoods by Number of Sales and Average Sale Price",
       x = "Neighborhood",
       y = "Average Sale Price") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))






###########################################################################################################
#----------------Model Development-------------------------------------------------------------
####################################################################################

#----Feature Selection : Multicolinarity Control ( High correlation of independent variables with each other )------

#-----Correlation Matrix Heatmap--------------


# Clean column names to be lowercase and without spaces
nyc_sales_clean2 <- nyc_sales_clean2 %>%
  clean_names()

# Check the updated column names
colnames(nyc_sales_clean2)

# Select numeric columns to include in the correlation matrix
# Assuming nyc_sales_clean is your cleaned dataset
numeric_columns <- nyc_sales_clean2 %>%
  select(`sale_price`, `residential_units`, `commercial_units`, `total_units`, 
         `land_square_feet`, `gross_square_feet`, `building_age`)

# Compute the correlation matrix
cor_matrix <- cor(numeric_columns, use = "complete.obs")

# Melt the correlation matrix for ggplot
melted_cor_matrix <- melt(cor_matrix)
# Round the correlation values to 2 digits
melted_cor_matrix$value <- round(melted_cor_matrix$value, 2)

# Create the heatmap
ggplot(melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = value), color = "black", size = 4) +  # Add labels with rounded values
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name="Correlation") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  labs(title = "Correlation Matrix Heatmap",
       x = "Variable",
       y = "Variable")


# Drop the highly correlated variables from the dataset
nyc_sales_clean_reduced <- nyc_sales_clean2 %>%
  select(-residential_units, -total_units)

# Check the remaining columns
colnames(nyc_sales_clean_reduced)


#----Feature Selection :Low Correlations with Dependent Variable

# Select numeric variables including the dependent variable 'SALE PRICE'
numeric_columns <- nyc_sales_clean2 %>%
  select(sale_price, commercial_units, 
         land_square_feet, gross_square_feet, building_age,block)

# Compute the correlation matrix between all numeric variables
cor_matrix <- cor(numeric_columns, use = "complete.obs")

# Extract and print the correlation of each independent variable with 'SALE PRICE'
cor_with_sale_price <- cor_matrix[, "sale_price"]

# Print the correlations, rounded to 2 decimal places
cor_with_sale_price_rounded <- round(cor_with_sale_price, 2)
print(cor_with_sale_price_rounded)

nyc_sales_clean_reduced <- nyc_sales_clean_reduced %>%
  select(-building_age)


#---------------One Hot Encoding For Categorical Columns------------------------------












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






