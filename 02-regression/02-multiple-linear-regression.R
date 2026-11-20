# 📌 Multiple Linear Regression in R (Enhanced Version)

# 📖 Lecture: Understanding Multiple Linear Regression
#
# Multiple linear regression is used to model the relationship between a dependent variable (Y) and multiple independent variables (X1, X2, ..., Xn).
# The equation for multiple linear regression is:
#
#   Y = β0 + β1*X1 + β2*X2 + ... + βn*Xn + ε
#
# Where:
# - Y = Dependent variable (target to predict)
# - X1, X2, ..., Xn = Independent variables (predictors)
# - β0 = Intercept (the base value of Y when all Xs are zero)
# - β1, β2, ..., βn = Regression coefficients (weights for each predictor)
# - ε = Error term (residuals, the difference between actual and predicted Y)
#
# 📌 The goal is to estimate the coefficients (β) to minimize the difference between predicted and actual values of Y, using the least squares method.
# This method minimizes the sum of squared residuals:
#
#   Minimize ∑(Yi - Ŷi)²
#
# where:
# - Yi = Actual value
# - Ŷi = Predicted value
#
# 📢 Key Insights:
# - If β coefficients are **positive**, increasing X increases Y (positive correlation).
# - If β coefficients are **negative**, increasing X decreases Y (negative correlation).
# - If the error term (ε) is **randomly distributed**, the model is likely well-fitted.
# - High **multicollinearity** (when independent variables are highly correlated) can lead to instability in the model.

# 🔍 **Lecture: Statistical Significance in Regression Analysis**
#
# 📌 **What is Statistical Significance?**
# Statistical significance helps us determine whether the relationships observed in our regression model are real or just occurred due to chance.
# 
# **Key concepts:**
# - **p-value**: Probability that the observed effect is due to randomness.
# - **t-value**: Measures how strongly a predictor is associated with the dependent variable.
# - **F-statistic**: Measures the overall significance of the model.
#
# **Decision Rules for p-value:**
# - If `p < 0.05`, the predictor is statistically significant.
# - If `p > 0.05`, the predictor is likely not significant and should be reconsidered.
#
# 🚀 **Automatic Feature Selection: Backward Elimination**
# To ensure that only significant predictors remain, we use **backward elimination**:

# 📥 Importing Required Libraries
library(caTools)  # For dataset splitting
library(ggplot2)  # For visualization
library(car)  # For VIF analysis

# 📂 Loading the Dataset
dataset <- read.csv('data/50-startups.csv')

# 🔄 Encoding Categorical Variable 'State' (One-Hot Encoding Alternative)
dataset$State <- factor(dataset$State,
                        levels = c('New York', 'California', 'Florida'),
                        labels = c(1, 2, 3))

# ✂️ Splitting the Dataset into Training and Test Sets
set.seed(123)  # Ensuring reproducibility
split <- sample.split(dataset$Profit, SplitRatio = 0.8)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# 📊 Checking Correlation Matrix (Helps in Feature Selection)
cor_matrix <- cor(dataset[, sapply(dataset, is.numeric)])
print(cor_matrix)

# 📈 Visualizing Feature Correlations
heatmap(cor_matrix, main="Feature Correlation Heatmap", col=heat.colors(10), symm=TRUE)

# 🤖 Training the Multiple Linear Regression Model
regressor <- lm(formula = Profit ~ ., data = training_set)

# 📋 Summary of the Model (Coefficients, p-values, R²)
summary(regressor)

# 🔍 Predicting Test Set Results
y_pred <- predict(regressor, newdata = test_set)

# 📊 Visualizing Predictions vs. Actual Profit
comparison <- data.frame(Actual = test_set$Profit, Predicted = y_pred)
ggplot(comparison, aes(x = Actual, y = Predicted)) +
  geom_point(color='blue', alpha=0.6) +
  geom_abline(intercept=0, slope=1, color='red', linetype='dashed') +
  ggtitle("Actual vs. Predicted Profit") +
  xlab("Actual Profit ($)") + ylab("Predicted Profit ($)")

# 📏 Evaluating Model Performance
MAE <- mean(abs(test_set$Profit - y_pred))  # Mean Absolute Error
R2 <- summary(regressor)$r.squared  # R² Score
cat("Mean Absolute Error:", MAE, "\n")
cat("R² Score:", R2, "\n")

# 📊 Confidence Intervals for Predictions
predictions_with_ci <- predict(regressor, newdata = test_set, interval = "confidence")

# Convert predictions to a data frame
pred_df <- data.frame(Actual = test_set$Profit, Predicted = predictions_with_ci[,1],
                      Lower = predictions_with_ci[,2], Upper = predictions_with_ci[,3])

ggplot(pred_df, aes(x = Actual, y = Predicted)) +
  geom_point(color='blue', alpha=0.6) +
  geom_errorbar(aes(ymin=Lower, ymax=Upper), width=0.2, color='gray') +
  geom_abline(intercept=0, slope=1, color='red', linetype='dashed') +
  ggtitle("Actual vs. Predicted Profit with Confidence Intervals") +
  xlab("Actual Profit ($)") + ylab("Predicted Profit ($)")

# 📉 Residual Analysis (Checking Model Assumptions)
residuals_df <- data.frame(Residuals = residuals(regressor), Fitted = fitted(regressor))

ggplot(residuals_df, aes(x = Fitted, y = Residuals)) +
  geom_point(color="blue", alpha=0.6) +
  geom_hline(yintercept=0, linetype="dashed", color="red") +
  ggtitle("Residual Analysis: Checking Model Assumptions") +
  xlab("Fitted Values") + ylab("Residuals")

# 📏 Checking Multicollinearity with VIF (After Feature Selection)
vif_values_final <- vif(stepwise_model_final)
print(vif_values_final)


# 🚀 Performing Backward Elimination (Feature Selection)
stepwise_model_final <- step(lm(Profit ~ ., data = training_set), direction = "backward")

# 🔄 Ensure test_set contains only the necessary columns after Backward Elimination
selected_vars <- names(coef(stepwise_model_final))
test_set_reduced <- test_set[, selected_vars[selected_vars != "(Intercept)"], drop = FALSE]

# 🔍 Predicting Test Set Results Using the Final Model
y_pred_stepwise <- predict(stepwise_model_final, newdata = test_set_reduced)


summary(stepwise_model_final)
