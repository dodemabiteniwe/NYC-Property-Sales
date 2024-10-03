# Load the necessary library and dataset
# Function to install and load a single package
install_and_load <- function(package) {
  if (!require(package, character.only = TRUE)) {
    install.packages(package, dependencies = TRUE)
    library(package, character.only = TRUE)
  }
}

# List of required packages
required_packages <- c("data.table", "tidyverse", "lubridate","caret",
                       "e1071","ggplot2","corrplot","scales","ggpubr",
                       "xgboost","gbm","janitor","grDevices" ,"knitr","kableExtra", "randomForest", "ranger")

# Install and load all required packages
for (pkg in required_packages) {
  install_and_load(pkg)
}

#  Data Loading
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

# Default correlations before any cleaning
corr_default2 <- nyc_data %>%
  select(where(is.numeric)) %>%  
  cor(use = "complete.obs")


nyc_data2<- nyc_data%>%
  select(c(`BOROUGH`, `BUILDING CLASS CATEGORY`,`BLOCK`, LOT, `TOTAL UNITS`,`LAND SQUARE FEET`, `GROSS SQUARE FEET`, `SALE PRICE`, building_age))
kable(
  head(nyc_data2,10), booktabs = TRUE, caption = "first lines of some relevant columns of data")%>%
  kable_styling(latex_options = c("HOLD_position","scale_down"))



#-------------------------------------------Handle  missing values------------------------
# Remove rows where sale price is missing or zero
nyc_data <- nyc_data %>%
  filter(`SALE PRICE` >100)%>%
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
kable(na_rate2, booktabs = TRUE, caption = "Percentage of missing values by variable.")%>%
  kable_styling(latex_options = c("HOLD_position","scale_down"))

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
correlation_table <- data.frame(
  Variable = rownames(corr_default2),
  Corr_Default = round(corr_default2[, "SALE PRICE"], 5),
  Corr_After_Missing = round(corr_after_missing2[, "SALE PRICE"], 5),
  Corr_After_Outliers = round(corr_after_outliers2[, "SALE PRICE"], 5)
)
kable(
  correlation_table, booktabs = TRUE, caption = "Correlation with target variable after each cleaning step.")%>%
  kable_styling(latex_options = c("HOLD_position","scale_down"))



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

# Box Plot of Sale Price by Borough

# Filter data to include only SALE PRICE values between 100 000 and 5,000,000
nyc_data_filtered2 <- nyc_data %>% filter(`SALE PRICE` >= 100000 & `SALE PRICE` <= 5000000)
ggplot(nyc_data_filtered2, aes(x = BOROUGH, y = `SALE PRICE`, fill = BOROUGH)) +
  geom_boxplot() +labs(x = "Borough",y = "Sale Price")+
  scale_fill_brewer(palette = "Set3")+ 
  theme_minimal()+theme(legend.position = "none") 

# Box Plot of Sale Price by Month
ggplot(nyc_data_filtered2, aes(x = sale_month, y = `SALE PRICE`, fill = sale_month)) +
  geom_boxplot() +
  labs( x = "Month", y = "Sale Price") +
  scale_fill_brewer(palette = "Set3") +  
  theme_minimal() +
  theme(legend.position = "none")  

# Sale Count by Month
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

# Plot Average Land Square Feet of Properties in Each Borough

nyc_data_filtered2 %>%
  group_by(BOROUGH) %>%
  summarise(avg_land_sqft = mean(`LAND SQUARE FEET`, na.rm = TRUE), .groups = 'drop') %>%
  ggplot(aes(x = BOROUGH, y = avg_land_sqft, size = avg_land_sqft, color = BOROUGH)) +
  geom_point(alpha = 0.7) + scale_size_continuous(range = c(8, 15))+
  labs( x = "Borough", y = "Average Land Square Feet")+
  theme_minimal() + theme(legend.position = "right")

#  Bar Plot for House Price by Top 10 Neighborhoods

nyc_data_filtered2%>%
  group_by(NEIGHBORHOOD) %>%
  summarize(Number_of_Sales = n(),
            Avg_Sale_Price = mean(`SALE PRICE`, na.rm = TRUE), .groups = 'drop')%>%
  arrange(desc(Number_of_Sales)) %>%
  head(10)%>%
  ggplot(aes(x = reorder(NEIGHBORHOOD, Number_of_Sales), y = Avg_Sale_Price, fill = NEIGHBORHOOD)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  coord_flip() +  # Flip coordinates for better readability
  labs(x = "Neighborhood", y = "Average Sale Price") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#-----------------Multicolinarity Control ( High correlation of independent variables with each other )----------

# Correlation Matrix Heatmap
numeric_columns <- nyc_data%>%
  select_if(is.numeric) %>%
  cor(use = "complete.obs")
corrplot(numeric_columns, method = "color", tl.cex = 0.7, number.cex = 0.7, addCoef.col = "black", round = 2)

nyc_data_reduce <- nyc_data %>%
  select(-c(LOT, `YEAR BUILT`, sale_year, building_age,`TOTAL UNITS`,`RESIDENTIAL UNITS`))
# Clean column names to be lowercase and without spaces
nyc_data_reduce <- nyc_data_reduce %>%
  clean_names()
nyc_data_reduce <- nyc_data_reduce %>%
  select(-c(tax_class_at_present, neighborhood, building_class_at_time_of_sale, sale_month))


###########################################################################################################
#----------------Model Development-------------------------------------------------------------
####################################################################################

nyc_sumary2 <- summary(nyc_data_reduce)
print(nyc_sumary2)

# One Hot Encoding
nyc_data_encoded <- nyc_data_reduce %>%
  model.matrix(~.-1, .) %>%
  as.data.frame()

# Split data into training and testing
set.seed(123)
trainIndex <- createDataPartition(nyc_data_encoded$sale_price, p = .8, list = FALSE)
train_data <- nyc_data_encoded[trainIndex, ]
test_data <- nyc_data_encoded[-trainIndex, ]


#--------------Model selection ------------------------------------------


# Set up training control with cross-validation
train_control <- trainControl(method = "cv", number = 2)

# Function to Evaluate Models
evaluate_model <- function(model, train_data, test_data, response_var) {
  start_time <- Sys.time()
  model_fit <- train(as.formula(paste(response_var, "~ .")), data = train_data, method = model, trControl = train_control,preProcess = c("center", "scale","nzv"))
  # Predictions for train and test datasets
  train_predictions <- predict(model_fit, newdata = train_data)
  test_predictions <- predict(model_fit, newdata = test_data)
  #Execution time
  end_time <- Sys.time()
  exec_time <- round(difftime(end_time, start_time, units = "secs"), 2)
  # Train metrics
  train_rmse <- RMSE(train_predictions, train_data$sale_price)
  train_mae <- MAE(train_predictions, train_data$sale_price)
  train_score <- cor(train_predictions, train_data$sale_price)^2
  
  # Test metrics
  test_rmse <- RMSE(test_predictions, test_data$sale_price)
  test_mae <- MAE(test_predictions, test_data$sale_price)
  test_score <- cor(test_predictions, test_data$sale_price)^2
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

# Evaluate models
results<- lapply(models, evaluate_model, train_data = train_data, test_data = test_data, response_var = "sale_price")

# Combine results into data frames
results_df <- do.call(rbind, results)

# Display results
kable(
  results_df, booktabs = TRUE, caption = "Performance of each algorithm based on different metrics.") %>%
  kable_styling(latex_options = c("HOLD_position","scale_down"))

#-----Optimize model select using the model parameters------------------------------------------

# Set up training control with cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Train Ranger Random Forest model 
rf_optimized <- train(sale_price ~ ., data = train_data, method = "ranger", trControl = train_control,  
                      num.trees = 500,            
                      importance = 'impurity',
                      preProcess = c("center", "scale","nzv")
)

# Predict on the test data using the best tuned model
rf_predictions <- predict(rf_optimized, newdata = test_data)

# Calculate the final metrics on the test data
final_rmse <- RMSE(rf_predictions, test_data$sale_price)
final_mae <- MAE(rf_predictions, test_data$sale_price)
final_score <- cor(rf_predictions, test_data$sale_price)^2

final_result <- data.frame(
  Model = "Ranger Random Forest model",
  Test_Score = round(final_score, 5),
  Test_RMSE = round(final_rmse, 5),
  Test_MAE = round(final_mae, 5)
)

# Display results
kable(
  final_result, booktabs = TRUE, caption = "Performance of the final algorithm.")%>%
  kable_styling(latex_options = c("HOLD_position","scale_down"))



#-----------Variable Importance (Random Forest)----------------------------------------

# Extract variable importance using caret's varImp function
var_importance <- varImp(rf_optimized)$importance

# Convert to a data frame and add row names as a column
var_importance_df <- var_importance %>%
  rownames_to_column(var = "Variable") %>%
  mutate(Importance = Overall / sum(Overall) * 100)  # Calculate percentages

# Plot variable importance using ggplot2
ggplot(var_importance_df, aes(x = reorder(Variable, Importance), y = Importance, fill = Importance)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  geom_text(aes(label = paste0(round(Importance, 1), "%")), hjust = -0.2) +
  coord_flip() +  # Flip to make it horizontal
  labs(x = "Variables", y = "Importance (%)") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") + theme_minimal()+
  scale_y_continuous(expand = expansion(mult = c(0, 0.3)))





