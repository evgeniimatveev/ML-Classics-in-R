# 🛠️ Data Preprocessing Template 🛠️
# This script includes essential steps for preparing data in R before applying machine learning models.

# ------------------------------------------
# 📌 1. Understanding Correlation
# ------------------------------------------
# Correlation measures the relationship between two variables, ranging from -1 to 1.
# - 1 → Perfect positive correlation (both variables increase together).
# - 0 → No correlation (no relationship).
# - -1 → Perfect negative correlation (one variable increases, the other decreases).
# Pearson, Spearman, and Kendall methods are commonly used to calculate correlation.

# ------------------------------------------
# 📌 2. Importing the dataset
# ------------------------------------------

# 📥 Load the dataset from a CSV file into a dataframe
# `read.csv("filename.csv")` is used to load tabular data from a CSV file.
dataset = read.csv('Data.csv', stringsAsFactors = FALSE)

# Ensure the dataset is correctly loaded
# str(dataset)  # 📊 Shows the structure of the dataset
# head(dataset)  # 🔍 Displays the first few rows


# ------------------------------------------
# 📌 3. Handling Missing Data
# ------------------------------------------
# In real-world datasets, some values might be missing (NA).
# Instead of removing these rows, we can replace missing values
# with the mean of the respective column to maintain data consistency.

# Handling missing values in the 'Age' column:
# - is.na(dataset$Age) checks for missing (NA) values.
# - ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)) computes the mean of 'Age' while ignoring NA values.
# - ifelse() replaces NA values with the calculated mean.
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

# Handling missing values in the 'Salary' column:
# - The same process is applied to 'Salary' to replace NA values with the column mean.
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Salary)


# ------------------------------------------
# 📌 4. Encoding Categorical Data
# ------------------------------------------
# Machine learning models typically do not work with categorical (text) data directly.
# We need to convert categorical values (e.g., 'France', 'Spain', 'Germany') into numerical form.
# This is called **Label Encoding**.

# Encoding the 'Country' variable:
# - factor() function is used to convert the text labels into numeric codes.
# - levels = c('France', 'Spain', 'Germany') defines the categories in the correct order.
# - labels = c(1, 2, 3) assigns numeric values to each category:
#    - France → 1
#    - Spain → 2
#    - Germany → 3
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3))

# Encoding the 'Purchased' variable:
# - This is a binary categorical variable ('Yes' or 'No').
# - We convert it into numerical values:
#    - 'No' → 0
#    - 'Yes' → 1
dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No', 'Yes'),
                           labels = c(0, 1))


# ------------------------------------------
# 📌 5. Splitting the dataset into Training and Test sets
# ------------------------------------------

# Install the necessary package if not already installed
# install.packages('caTools')
library(caTools)

set.seed(123)  # 🎯 Setting a seed ensures reproducibility

# 📌 Why split the dataset?
# In machine learning, we **train** the model on one subset (training set) 
# and evaluate its performance on another subset (test set) to avoid overfitting.
# The **training set (80%)** is used to learn patterns, while the **test set (20%)** is used for validation.

# Splitting the dataset (80% training, 20% testing)
# sample.split() ensures that the distribution of 'Purchased' classes is similar in both sets.
split = sample.split(dataset$Purchased, SplitRatio = 0.8)

# Creating training and test sets based on the split
training_set = subset(dataset, split == TRUE)  # Training data (80%)
test_set = subset(dataset, split == FALSE)  # Test data (20%)

# Printing the dimensions of training and test sets
print(dim(training_set))  # Example output: (8, 4) if dataset has 10 rows
print(dim(test_set))  # Example output: (2, 4)


# ------------------------------------------
# 📌 6. Feature Scaling
# ------------------------------------------

# 📌 Why do we need feature scaling?
# Many machine learning models perform better when numerical features 
# are scaled to the same range, especially for distance-based algorithms (e.g., KNN, SVM).
# It prevents one feature (e.g., Salary) from dominating others (e.g., Age).

# Example: Suppose we have:
# - Age = [27, 38, 48, 50]  (small values)
# - Salary = [48000, 61000, 79000, 83000]  (large values)
# The model might give too much importance to Salary simply because of its magnitude.

# Feature scaling transforms values to a standard range (mean = 0, std = 1).

# Standardizing 'Age' and 'Salary' (optional, uncomment if needed)
# scale() normalizes data to have a mean of 0 and standard deviation of 1.
# Scaling is **not required for categorical variables**.
# Uncomment these lines if your model requires scaling:

# training_set[, c("Age", "Salary")] = scale(training_set[, c("Age", "Salary")])
# test_set[, c("Age", "Salary")] = scale(test_set[, c("Age", "Salary")])

# Example before and after scaling:
# Before: Age = [27, 38, 48, 50], Salary = [48000, 61000, 79000, 83000]
# After: Age = [-1.1, 0.2, 1.5, 1.8], Salary = [-1.3, 0.1, 1.4, 1.7]

# Scaling is especially useful for:
# ✅ Logistic Regression
# ✅ K-Nearest Neighbors (KNN)
# ✅ Support Vector Machines (SVM)
# ✅ Principal Component Analysis (PCA)
# ❌ Not needed for Decision Trees or Random Forests


# ------------------------------------------
# 📅 Additional Data Summary and Insights
# ------------------------------------------

# Checking for missing values
missing_values = colSums(is.na(dataset))
print(missing_values)

# Summary statistics for numerical features
summary(dataset)

# Checking correlations between numerical variables
correlation_matrix = cor(dataset[, sapply(dataset, is.numeric)], use = "complete.obs")
print(correlation_matrix)

# ------------------------------------------
# ✅ Data Preprocessing Complete!
# ------------------------------------------
# The dataset is now ready for use in machine learning models!