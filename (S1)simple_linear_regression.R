# 📌 Simple Linear Regression - Full Implementation in R

# 🔹 What is Linear Regression?
# Linear Regression is a fundamental machine learning algorithm used for predicting numerical values.
# It assumes a **linear relationship** between an independent variable (X) and a dependent variable (y).
#
# 🔹 Formula for Simple Linear Regression:
#
#     ŷ = b₀ + b₁ * X
#
# Where:
# - ŷ  → Predicted value (dependent variable)
# - X  → Independent variable (feature)
# - b₀ → Intercept (bias term)
# - b₁ → Slope (coefficient that determines the impact of X on y)
#
# 🔹 Key Insights:
# - The **intercept (b₀)** represents the salary when experience is **zero**.
# - The **slope (b₁)** shows how much the salary increases per additional year of experience.
# - If **b₁ is positive**, there’s a positive correlation (more experience → higher salary).
# - If **b₁ is negative**, there’s an inverse relationship.

# 📌 Step 1: Import Required Libraries
# Required libraries for data manipulation and visualization
library(caTools)
library(ggplot2)
library(Metrics)
library(caret)
library(reshape2)
library(RColorBrewer)

# 📌 Step 2: Load the Dataset
# Load Salary dataset containing experience and salary data
dataset <- read.csv('Salary_Data.csv')

# Display the first few rows of the dataset to understand its structure
head(dataset)

# 📌 Step 3: Split the Dataset into Training and Test Sets
# Set a seed to ensure reproducibility
set.seed(123)
split <- sample.split(dataset$Salary, SplitRatio = 2/3)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# 📌 Step 4: Train the Simple Linear Regression Model
# Fit a simple linear regression model to predict Salary based on Years of Experience
regressor <- lm(Salary ~ YearsExperience, data = training_set)

# Print model summary to analyze coefficients and significance levels
summary(regressor)

# 📌 Step 5: Predict the Test Set Results
# Generate predictions on the test set
y_pred <- predict(regressor, newdata = test_set)

# Create a dataframe to compare actual vs predicted salaries
results <- data.frame(Actual = test_set$Salary, Predicted = y_pred)
print(results)

# 📌 Step 6: Visualizing the Training Set Results
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), colour = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of experience') +
  ylab('Salary')

# 📌 Step 7: Visualizing the Test Set Results
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), colour = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of experience') +
  ylab('Salary')

# 📌 Step 8: Evaluating Model Performance
rmse_value <- rmse(test_set$Salary, y_pred)
r2_value <- cor(test_set$Salary, y_pred)^2
cat("RMSE:", rmse_value, "\n")
cat("R-squared:", r2_value, "\n")

# 📌 Step 9: Residuals Analysis
residuals <- test_set$Salary - y_pred

# Histogram of residuals
ggplot(data.frame(residuals), aes(x = residuals)) +
  geom_histogram(bins = 10, fill = "blue", alpha = 0.7) +
  ggtitle("Residuals Histogram") +
  xlab("Residuals") +
  ylab("Frequency")

# Residuals Plot
ggplot(data.frame(YearsExperience = test_set$YearsExperience, residuals), aes(x = YearsExperience, y = residuals)) +
  geom_point(color = "red") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  ggtitle("Residuals Plot") +
  xlab("Years of Experience") +
  ylab("Residuals")

# 📌 Step 10: Contour Map of Regression Error
error_data <- data.frame(YearsExperience = test_set$YearsExperience, Salary = test_set$Salary, Error = residuals)
error_matrix <- acast(error_data, YearsExperience ~ Salary, value.var = "Error", fun.aggregate = mean)

ggplot(melt(error_matrix), aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradientn(colors = brewer.pal(9, "RdBu")) +
  ggtitle("Contour Map of Regression Error") +
  xlab("Experience (Years)") +
  ylab("Salary ($)")

# 📌 Step 11: Alternative Approach - Using caret for Linear Regression
model <- train(Salary ~ YearsExperience, data = training_set, method = "lm")
y_pred_caret <- predict(model, newdata = test_set)
results_caret <- data.frame(Actual = test_set$Salary, Predicted = y_pred_caret)
print(results_caret)

# 📌 Step 12: Comparison of Actual vs Predicted Salaries
ggplot(results_caret, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  ggtitle("Actual vs Predicted Salaries") +
  xlab("Actual Salary") +
  ylab("Predicted Salary")
# 📌 Step 13: Evaluating Model Performance

# Model Summary
# The following statistics help us understand the strength of our regression model:
summary(regressor)

# Call:
lm(formula = Salary ~ YearsExperience, data = training_set)

# Residuals:
#    Min      1Q  Median      3Q     Max 
# -7325.1 -3814.4   427.7  3559.7  8884.6 

# Coefficients:
#                Estimate Std. Error t value Pr(>|t|)    
# (Intercept)        25592       2646   9.672 1.49e-08 ***
# YearsExperience     9365        421  22.245 1.52e-14 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Residual standard error: 5391 on 18 degrees of freedom
# Multiple R-squared:  0.9649, Adjusted R-squared:  0.963 
# F-statistic: 494.8 on 1 and 18 DF,  p-value: 1.524e-14

# Interpretation of Results:
# - **Residuals:** Shows the distribution of errors (min, max, quartiles).
# - **Coefficients:**
#   - Intercept (b₀) = 25592 → This is the estimated salary when experience is 0 years.
#   - YearsExperience (b₁) = 9365 → Each additional year of experience increases the salary by $9365 on average.
# - **R-squared (0.9649):** Indicates that 96.49% of salary variation is explained by experience.
# - **F-statistic (494.8, p-value: 1.524e-14):** Strong evidence that the model is statistically significant.

# Compute RMSE and R² for model evaluation
rmse_value <- rmse(test_set$Salary, y_pred)
r2_value <- cor(test_set$Salary, y_pred)^2
cat("RMSE:", rmse_value, "\n")
cat("R-squared:", r2_value, "\n")
# 📌 Conclusion
# - Implemented a simple linear regression model using `lm()` and `caret`.
# - Visualized training and test set results with scatter plots.
# - Evaluated model performance using RMSE and R².
# - Analyzed residuals and added a Contour Map for regression error.
# - Used caret for an automated regression approach.
# - Further improvements can be made using feature engineering and additional preprocessing.