# Random Forest Regression - Key Insights
#
# How It Works:
# - Bagging (Bootstrap Aggregating): Each tree is trained on a random subset of the data.
# - Overfitting Reduction: By averaging multiple trees, the model generalizes better.
# - High Accuracy: Works well on both regression and classification tasks.
#
# The Regression Formula:
#   ŷ = (1/T) * Σ f_t(x)   (t=1 to T)
#
# Where:
# - (ŷ) Predicted Value → The estimated outcome (e.g., salary 💰)
# - (T) → The number of decision trees in the forest
# - (f_t(x)) → The prediction from an individual decision tree (t)
# - (1/T Σ f_t(x)) → The average prediction from all decision trees in the ensemble
# - (ε) Error Term → Represents the difference between actual and predicted values due to noise
#
# Key Insights:
# - The Random Forest model is an ensemble learning method that builds multiple Decision Trees and averages their predictions to improve accuracy.
# - Each tree is trained on a random subset of the data (bagging), reducing overfitting.
# - More trees → More robust model, reducing variance and improving generalization.
# - Can capture non-linearity in data while being less sensitive to small changes.
# - Unlike Linear Regression, Random Forest does not require feature scaling.
# - Random Forest Regression is powerful when relationships between variables are complex, and high stability & accuracy are required!
#
# --------------------------------------------------------------------------------------------

# Random Forest vs Decision Tree Regression

# Load necessary libraries
library(ggplot2)
library(rpart)
library(randomForest)

# Load dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]  # Keep only Level and Salary columns

# Set seed for reproducibility
set.seed(1234)

# Train Random Forest with better parameters
rf_regressor = randomForest(x = dataset[-2], y = dataset$Salary, 
                            ntree = 500, nodesize = 1, importance = TRUE)

# Train Decision Tree with optimized settings
dt_regressor = rpart(Salary ~ Level, data = dataset, method = "anova",
                     control = rpart.control(cp = 0.001, maxdepth = 5, minsplit = 2))

# Create a sequence of values for smooth plotting
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)

# Visualizing Random Forest Regression with smooth predictions
rf_plot = ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red', size = 3, alpha = 0.7) + 
  geom_line(aes(x = x_grid, y = predict(rf_regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue', size = 1.2) +  # Smooth curve for Random Forest
  ggtitle('Random Forest Regression (Improved)') +
  xlab('Level') +
  ylab('Salary') +
  theme_minimal()

# Print Random Forest plot
print(rf_plot)

# Visualizing Decision Tree Regression with step function
dt_plot = ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red', size = 3, alpha = 0.7) + 
  geom_step(aes(x = x_grid, y = predict(dt_regressor, newdata = data.frame(Level = x_grid))),
            colour = 'green', size = 1.2) +  # Step plot for Decision Tree
  ggtitle('Decision Tree Regression (Fixed)') +
  xlab('Level') +
  ylab('Salary') +
  theme_minimal()

# Print Decision Tree plot
print(dt_plot)

# Combined visualization
combined_plot = ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red', size = 3, alpha = 0.7) +
  geom_line(aes(x = x_grid, y = predict(rf_regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue', linetype = "solid", size = 1.2) +  # Smooth line for Random Forest
  geom_step(aes(x = x_grid, y = predict(dt_regressor, newdata = data.frame(Level = x_grid))),
            colour = 'green', linetype = "solid", size = 1.2) +  # Step function for Decision Tree
  ggtitle('Random Forest vs Decision Tree Regression (Final)') +
  xlab('Level') +
  ylab('Salary') +
  theme_minimal()

# Print combined plot
print(combined_plot)

# Compute performance metrics for Random Forest
mae_rf = mean(abs(dataset$Salary - predict(rf_regressor, dataset)))
mse_rf = mean((dataset$Salary - predict(rf_regressor, dataset))^2)
rmse_rf = sqrt(mse_rf)
r2_rf = 1 - (sum((dataset$Salary - predict(rf_regressor, dataset))^2) / sum((dataset$Salary - mean(dataset$Salary))^2))

# Compute performance metrics for Decision Tree
mae_dt = mean(abs(dataset$Salary - predict(dt_regressor, dataset)))
mse_dt = mean((dataset$Salary - predict(dt_regressor, dataset))^2)
rmse_dt = sqrt(mse_dt)
r2_dt = 1 - (sum((dataset$Salary - predict(dt_regressor, dataset))^2) / sum((dataset$Salary - mean(dataset$Salary))^2))



# Decision Tree model summary
summary(dt_regressor)

# Random Forest model summary
print(rf_regressor)  # General Random Forest model summary
summary(rf_regressor)  # Detailed summary
importance(rf_regressor)  # Feature importance



# Create comparison table with rounded values for readability
comparison_table = data.frame(
  Model = c("Random Forest", "Decision Tree"),
  MAE = round(c(mae_rf, mae_dt), 2),
  MSE = round(c(mse_rf, mse_dt), 2),
  RMSE = round(c(rmse_rf, rmse_dt), 2),
  R2 = round(c(r2_rf, r2_dt), 4)
)

# Print comparison table
print(comparison_table)


# Create a table with actual vs predicted values
predictions_table = data.frame(
  Level = dataset$Level,  # Levels from dataset
  Actual_Salary = dataset$Salary,  # Actual salaries
  Predicted_RF = predict(rf_regressor, dataset),  # Predicted salaries (Random Forest)
  Predicted_DT = predict(dt_regressor, dataset)  # Predicted salaries (Decision Tree)
)

# Print the table
print(predictions_table)

