# 📌 **Support Vector Regression (SVR) & Linear Regression in R**

# 🔥 What is SVR?
# Support Vector Regression (SVR) in R is implemented using the `e1071` package, which allows for
# flexible non-linear regression using Support Vector Machines (SVM). Unlike linear regression,
# SVR aims to fit most data points within a margin (ε-tube), ignoring small deviations to improve robustness.

# 📌 **Key Applications of SVR in R**
# ✔ **Financial Forecasting** → Stock price predictions 📈
# ✔ **Healthcare** → Predicting patient outcomes 🏥
# ✔ **Real Estate** → Estimating house prices 🏡

# 📌 **The Regression Formula in SVR**
# SVR approximates the function as:
#   ŷ = w1*x1 + w2*x2 + ... + wn*xn + b
# Where:
#   - ŷ (**Predicted Value**) → The estimated outcome (e.g., price 💰, salary 💵)
#   - x1, x2, ..., xn (**Independent Variables**) → Input features affecting the prediction 📊
#   - w1, w2, ..., wn (**Weights**) → Coefficients that define the importance of each feature ⚖
#   - b (**Bias**) → The base prediction when all inputs are zero 🎯

# 🔑 **Key Insights**
# - The **bias (b)** represents the base prediction when all input features are zero.
# - Each **weight coefficient (w1, w2, w3, …, wn)** determines how much the predicted value changes per unit increase.
# - **SVR finds a function that fits most data points within a margin (ε-tube) rather than minimizing absolute errors.**
# - **If the kernel function is linear, SVR behaves like Linear Regression with margin constraints.**
# - **Non-linear kernels (RBF, Polynomial, Sigmoid) allow SVR to capture complex relationships.**

# 📌 **SVR & Linear Regression Implementation in R**
library(e1071)   # Support Vector Machine (SVM) package
library(ggplot2)  # Visualization library
library(dplyr)    # Data manipulation

# 📌 **Step 1: Loading the Dataset**
dataset <- read.csv('Position_Salaries.csv')  # Load CSV file
dataset <- dataset[2:3]  # Selecting only 'Level' and 'Salary' columns

# 📌 **Step 2: Fitting Linear Regression Model**
lin_reg <- lm(Salary ~ Level, data = dataset)  # Creating a Linear Regression Model

# 📌 **Step 3: Feature Scaling for SVR**
scale_x <- scale(dataset$Level)  # Standardizing 'Level' column
scale_y <- scale(dataset$Salary) # Standardizing 'Salary' column

dataset$Level_scaled <- as.numeric(scale_x)  # Saving scaled Level
dataset$Salary_scaled <- as.numeric(scale_y)  # Saving scaled Salary

# 📌 **Step 4: Fitting SVR Model**
svr_reg <- svm(formula = Salary_scaled ~ Level_scaled, 
               data = dataset, 
               type = 'eps-regression', 
               kernel = 'radial')  # Using RBF kernel for better non-linear predictions

# 📌 **Step 5: Predictions**
x_grid <- seq(min(dataset$Level), max(dataset$Level), by = 0.1)  # Generating levels for visualization
lin_pred <- predict(lin_reg, newdata = data.frame(Level = x_grid))  # Linear regression predictions
y_grid_scaled <- predict(svr_reg, data.frame(Level_scaled = as.numeric(scale(x_grid, center = attr(scale_x, "scaled:center"), scale = attr(scale_x, "scaled:scale")))))
y_grid_svr <- y_grid_scaled * attr(scale_y, "scaled:scale") + attr(scale_y, "scaled:center")  # Rescaling SVR predictions

# 📌 **Step 6: Visualizing Linear & SVR Predictions**
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red', size = 3) +  # Actual salary data
  geom_line(aes(x = x_grid, y = lin_pred), colour = 'blue', linewidth = 1.5) +  # Linear Regression predictions
  geom_line(aes(x = x_grid, y = y_grid_svr), colour = 'green', linewidth = 1.5) +  # SVR predictions
  ggtitle('Comparison of SVR and Linear Regression') +
  xlab('Level') +
  ylab('Salary') +
  theme_minimal()

# 📌 **Step 7: Model Performance Comparison**
actuals <- dataset$Salary  # True salary values
lin_predictions <- predict(lin_reg, newdata = dataset)  # Linear regression predictions
svr_predictions_scaled <- predict(svr_reg, data.frame(Level_scaled = scale(dataset$Level, center = attr(scale_x, "scaled:center"), scale = attr(scale_x, "scaled:scale"))))
svr_predictions <- svr_predictions_scaled * attr(scale_y, "scaled:scale") + attr(scale_y, "scaled:center")  # Rescale predictions

# 📌 **Step 8: Visualizing Residuals for Both Models**
residuals_df <- data.frame(
  Level = dataset$Level,
  Residuals_Lin = actuals - lin_predictions,
  Residuals_SVR = actuals - svr_predictions
)

ggplot(residuals_df, aes(x = Level)) +
  geom_point(aes(y = Residuals_Lin), colour = 'blue', size = 2) +
  geom_point(aes(y = Residuals_SVR), colour = 'green', size = 2) +
  ggtitle('Residuals of SVR vs Linear Regression') +
  xlab('Level') +
  ylab('Residuals') +
  theme_minimal()

# 📌 **Step 9: Heatmap of Model Errors**
errors_df <- data.frame(
  Model = rep(c("Linear Regression", "SVR"), each = length(dataset$Level)),
  Level = rep(dataset$Level, times = 2),
  Error = c(abs(actuals - lin_predictions), abs(actuals - svr_predictions))
)

ggplot(errors_df, aes(x = as.factor(Level), y = Error, fill = Model)) +
  geom_col(position = "dodge") +  
  ggtitle('Prediction Errors by Model') +
  xlab("Level") +
  ylab("Absolute Error") +
  scale_fill_manual(values = c("blue", "green")) +
  theme_minimal()

# 📌 **Step 10: Boxplot of Model Performance Metrics**
ggplot(errors_df, aes(x = Model, y = Error, fill = Model)) +
  geom_boxplot() +
  ggtitle('Boxplot of Model Errors') +
  ylab('Error') +
  theme_minimal()

# 📌 **Step 11: Performance Metrics Calculation**
performance_df <- data.frame(
  Model = c("Linear Regression", "SVR"),  # Model names
  R2_Score = c(
    summary(lin_reg)$r.squared,  # R² for Linear Regression
    1 - sum((dataset$Salary - svr_predictions)^2) / sum((dataset$Salary - mean(dataset$Salary))^2)  # R² for SVR
  ),
  MAE = c(
    mean(abs(dataset$Salary - lin_predictions)),  # Mean Absolute Error (MAE) for Linear Regression
    mean(abs(dataset$Salary - svr_predictions))   # MAE for SVR
  ),
  MSE = c(
    mean((dataset$Salary - lin_predictions)^2),  # Mean Squared Error (MSE) for Linear Regression
    mean((dataset$Salary - svr_predictions)^2)   # MSE for SVR
  ),
  RMSE = c(
    sqrt(mean((dataset$Salary - lin_predictions)^2)),  # Root Mean Squared Error (RMSE) for Linear Regression
    sqrt(mean((dataset$Salary - svr_predictions)^2))   # RMSE for SVR
  )
)

# 📌 **Step 12: Displaying Performance Metrics Table**
print("📌 Model Performance Metrics:")  # Print header for the table
print(performance_df)  # Print the calculated performance metrics

# 📌 **Step 13: Summary of Model Performance**
cat("\n📌 Summary of Model Performance:\n")

# Compare R² scores: Higher R² means the model explains more variance in the data
if (performance_df$R2_Score[2] > performance_df$R2_Score[1]) {
  cat("✔ SVR performed better in explaining variance compared to Linear Regression.\n")
} else {
  cat("✔ Linear Regression performed better in explaining variance.\n")
}

# Print additional evaluation metrics
cat("✔ MAE and RMSE indicate how well models approximate real values.\n")
cat("✔ Lower MAE and RMSE values indicate a better fit.\n")

# 📌 **Final Visualization: Performance Comparison**
ggplot(performance_df, aes(x = Model)) +  
  geom_bar(aes(y = R2_Score, fill = Model), stat = "identity", alpha = 0.7) +  # R² score comparison
  geom_bar(aes(y = MAE, fill = Model), stat = "identity", alpha = 0.5, color = "black") +  # MAE comparison
  geom_bar(aes(y = RMSE, fill = Model), stat = "identity", alpha = 0.3, color = "gray") +  # RMSE comparison
  ggtitle("📊 Model Performance Comparison") +  # Title of the plot
  ylab("Score") +  # Y-axis label
  theme_minimal()  # Apply minimal theme for cleaner visualization