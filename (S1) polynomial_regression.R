# 🎓 The Regression Formula:
# ŷ = b0 + b1*X + b2*X² + ... + bn*Xⁿ + ϵ

# 📌 Where:
# ✅ ŷ (Predicted Value) → The estimated outcome (e.g., salary 💰)
# ✅ X, X², ..., Xⁿ (Polynomial Features) → Transformed input features capturing non-linearity 📊
# ✅ b0 (Intercept) ⏳ → The base salary when X = 0
# ✅ b1, b2, ..., bn (Coefficients) 📈 → The rate of change for each polynomial term of X
# ✅ ϵ (Error Term) ⚠️ → Represents the difference between actual and predicted values due to noise

# 💡 Key Insights:
# - The intercept (b0) represents the base salary when Level = 0.
# - Each coefficient (b1, b2, ..., bn) determines how much the predicted salary changes.
# - If b1, b2, ..., bn are **positive**, the polynomial curve trends **upward** 📈.
# - If b1, b2, ..., bn are **negative**, the polynomial curve trends **downward** 📉.
# - Unlike **Multiple Linear Regression**, Polynomial Regression captures **non-linear** relationships.

# -----------------------------------
# 📦 Step 1: Load necessary libraries
# -----------------------------------
library(ggplot2)  # 📊 For visualization

# -----------------------------------
# 📂 Step 2: Load the dataset
# -----------------------------------
dataset <- read.csv("Position_Salaries.csv")  # Load data from CSV file
dataset <- dataset[2:3]  # Select relevant columns: Level (X) and Salary (y)

# -----------------------------------
# 📈 Step 3: Train Linear Regression Model
# -----------------------------------
# Formula:  Salary = β0 + β1 * Level  (Simple Linear Regression)
lin_reg <- lm(Salary ~ Level, data = dataset)  # Train a linear regression model

# -----------------------------------
# 🔄 Step 4: Transform dataset for Polynomial Regression (Degree 4)
# -----------------------------------
# Polynomial Regression Formula (Degree 4):
# Salary = β0 + β1 * Level + β2 * Level² + β3 * Level³ + β4 * Level⁴ + ε
dataset$Level2 <- dataset$Level^2  # Generate Level²
dataset$Level3 <- dataset$Level^3  # Generate Level³
dataset$Level4 <- dataset$Level^4  # Generate Level⁴

# Train the polynomial regression model
poly_reg <- lm(Salary ~ Level + Level2 + Level3 + Level4, data = dataset)

# -----------------------------------
# 📊 Step 5: Visualizing the Linear Regression Model
# -----------------------------------
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") +  # Actual data points
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)), color = "blue") + 
  ggtitle("Truth or Bluff (Linear Regression)") +
  xlab("Level") + ylab("Salary")

# -----------------------------------
# 📊 Step 6: Visualizing the Polynomial Regression Model (Smooth Curve)
# -----------------------------------
# 🔹 We use a finer grid of x-values for a smooth curve
x_grid <- seq(min(dataset$Level), max(dataset$Level), 0.1)  # Generate high-resolution X-axis
dataset_grid <- data.frame(Level = x_grid, Level2 = x_grid^2, Level3 = x_grid^3, Level4 = x_grid^4)

ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") +  # Real data points
  geom_line(aes(x = x_grid, y = predict(poly_reg, newdata = dataset_grid)), color = "blue") + 
  ggtitle("Truth or Bluff (Polynomial Regression)") +
  xlab("Level") + ylab("Salary")


# -----------------------------------
# 📊 Step 8: Additional Visualization - Comparison of Linear vs Polynomial Predictions
# -----------------------------------
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") +  # Actual Data
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)), color = "blue", linetype="dashed") +  # Linear Prediction
  geom_line(aes(x = x_grid, y = predict(poly_reg, newdata = dataset_grid)), color = "green") +  # Polynomial Prediction
  ggtitle("Comparison: Linear vs Polynomial Regression") +
  xlab("Level") + ylab("Salary") +
  theme_minimal()

# -----------------------------------
# 🔮 Step 9: Predicting Salary for Level = 6.5
# -----------------------------------
# Prepare new data for prediction
new_data <- data.frame(Level = 6.5, Level2 = 6.5^2, Level3 = 6.5^3, Level4 = 6.5^4)

# 🔹 Predict salary using Polynomial Regression
predicted_salary <- predict(poly_reg, newdata = new_data)
cat("🔮 Predicted Salary for Level 6.5:", predicted_salary, "\n")

# -----------------------------------
# 🏆 Step 10: Model Evaluation - Comparing Linear vs Polynomial Regression
# -----------------------------------
# 📊 We will compute key regression metrics:
# ✅ R² Score (coefficient of determination)
# ✅ MAE (Mean Absolute Error)
# ✅ MSE (Mean Squared Error)
# ✅ RMSE (Root Mean Squared Error)

# 📌 Compute R² Score for both models
r2_lin <- summary(lin_reg)$r.squared
r2_poly <- summary(poly_reg)$r.squared

# 📌 Compute MAE, MSE, and RMSE
mae <- function(actual, predicted) { mean(abs(actual - predicted)) }  # MAE function

y_actual <- dataset$Salary  # True salaries
y_pred_lin <- predict(lin_reg, newdata = dataset)  # Predictions from Linear Regression
y_pred_poly <- predict(poly_reg, newdata = dataset)  # Predictions from Polynomial Regression

mae_lin <- mae(y_actual, y_pred_lin)
mae_poly <- mae(y_actual, y_pred_poly)

mse_lin <- mean((y_actual - y_pred_lin)^2)
mse_poly <- mean((y_actual - y_pred_poly)^2)

rmse_lin <- sqrt(mse_lin)
rmse_poly <- sqrt(mse_poly)

# 📊 Create a dataframe for structured comparison
performance_comparison <- data.frame(
  Model = c("Linear Regression", "Polynomial Regression (Degree=4)"),
  R2_Score = c(r2_lin, r2_poly),
  MAE = c(mae_lin, mae_poly),
  MSE = c(mse_lin, mse_poly),
  RMSE = c(rmse_lin, rmse_poly)
)

# 📊 Display the performance comparison table
print(performance_comparison)

# -----------------------------------
# 📢 Step 11: Summary & Conclusion
# -----------------------------------
cat("\n🏆 **Final Conclusion:** Polynomial Regression (Degree=4) significantly outperforms Linear Regression. 🎯\n")
cat("📌 **R² Score:** Polynomial = High accuracy (", round(r2_poly, 4), ") vs. Linear = Poor fit (", round(r2_lin, 4), ").\n")
cat("📌 **Mean Absolute Error (MAE):** Polynomial is ~", round(mae_lin/mae_poly, 1), "x smaller, meaning more precise predictions.\n")
cat("📌 **RMSE:** Polynomial = ", round(rmse_poly, 2), ", Linear = ", round(rmse_lin, 2), ", showing a much better fit.\n")
cat("\n✅ **Polynomial Regression is the clear winner for this dataset!** 🚀\n")


# -----------------------------------
# 📊 Step 12: Residual Plot for Linear Regression
# -----------------------------------
# 🔹 Residuals = Actual - Predicted
residuals_lin <- dataset$Salary - predict(lin_reg, newdata = dataset)

ggplot(data.frame(Level = dataset$Level, Residuals = residuals_lin)) +
  geom_point(aes(x = Level, y = Residuals), color = "blue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  ggtitle("Residuals Plot: Linear Regression") +
  xlab("Level") + ylab("Residuals") +
  theme_minimal()

# -----------------------------------
# 📊 Step 13: Residual Plot for Polynomial Regression
# -----------------------------------
residuals_poly <- dataset$Salary - predict(poly_reg, newdata = dataset)

ggplot(data.frame(Level = dataset$Level, Residuals = residuals_poly)) +
  geom_point(aes(x = Level, y = Residuals), color = "green") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  ggtitle("Residuals Plot: Polynomial Regression") +
  xlab("Level") + ylab("Residuals") +
  theme_minimal()

# -----------------------------------
# 📊 Step 14: Comparison of Errors Between Models
# -----------------------------------
# 🔹 Create a dataframe to compare model errors
error_comparison <- data.frame(
  Model = c("Linear Regression", "Polynomial Regression (Degree=4)"),
  MAE = c(mae_lin, mae_poly),
  RMSE = c(rmse_lin, rmse_poly)
)

# 🔹 Visualizing the error comparison
library(reshape2)  # Needed for reshaping data
error_comparison_melted <- melt(error_comparison, id.vars = "Model")

ggplot(error_comparison_melted, aes(x = Model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  ggtitle("Comparison of Errors: Linear vs Polynomial Regression") +
  xlab("Model") + ylab("Error Value") +
  scale_fill_manual(values = c("MAE" = "orange", "RMSE" = "blue")) +
  theme_minimal()