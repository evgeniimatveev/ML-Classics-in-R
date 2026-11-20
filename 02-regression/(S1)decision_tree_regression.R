# 🌳 Decision Tree Regression

# 📌 The Regression Formula:
#  ŷ = Σ (c_j * I(x ∈ R_j))

# 🔍 Where:
# ✅ ŷ (Predicted Value) → The estimated outcome (e.g., salary 💰)
# ✅ J → The number of terminal (leaf) regions in the tree
# ✅ c_j → The constant predicted value for all ( x ) in region ( R_j )
# ✅ I(x ∈ R_j) → Indicator function that checks if ( x ) belongs to region ( R_j )
# ⚠️ ϵ (Error Term) → Represents the difference between actual and predicted values due to noise 🛠️

# 💡 Key Insights:
# - The Decision Tree model splits data into regions based on **feature thresholds**.
# - Each **region ( R_j )** has a constant predicted value, minimizing variance.
# - More splits → **More complex model, but risk of overfitting increases.**
# - Decision Trees **capture non-linearity** but can be **sensitive to small changes**.
# - Unlike **Linear Regression**, Decision Trees do not require feature scaling.
# - 🌟 **Decision Tree Regression is powerful when relationships are non-linear and require an interpretable model!**


# 📥 Load required libraries
library(rpart)  # 🌳 Decision Tree Regression model
library(randomForest)  # 🌲 Random Forest Regression model
library(ggplot2)  # 📊 Data visualization
library(caTools)  # ✂️ Splitting dataset into training and testing sets
library(gt)  # 📋 Creating formatted tables

# 📂 Load the dataset
dataset <- read.csv('housing.csv')  # 📥 Load CSV file

# 🔄 Convert categorical variables into numerical (One-Hot Encoding)
dataset <- model.matrix(~ . -1, data = dataset)  # 🔀 Convert categorical features into numeric

# 🛠️ Split the dataset into features (X) and target variable (y)
X <- dataset[, -ncol(dataset)]  # 🎯 Select all columns except the last one as features
y <- dataset[, ncol(dataset)]  # 🏠 The last column is the target variable (house price)

# ✂️ Split the dataset into training (80%) and testing (20%) sets
set.seed(42)  # 🎲 Set a random seed for reproducibility
split <- sample.split(y, SplitRatio = 0.8)  # 🔄 Split data into training and test sets
X_train <- X[split, ]  # 📚 Training features
X_test <- X[!split, ]  # 📝 Testing features
y_train <- y[split]  # 📚 Training target variable
y_test <- y[!split]  # 📝 Testing target variable

# 🌳 Train the Decision Tree model
dt_model <- rpart(y_train ~ ., 
                  data = data.frame(X_train, y_train), 
                  control = rpart.control(minsplit = 5, cp = 0.01))  # ⚙️ Adjust complexity parameter

# 🔄 If the tree has only a root node, show a warning
if (nrow(dt_model$frame) == 1) {
  stop("🚨 Decision Tree did not split! Check your training data.")
}

# 🔮 Make predictions using Decision Tree
y_pred_dt <- predict(dt_model, newdata = data.frame(X_test))

# 📊 Evaluate Decision Tree model performance
mae_dt <- mean(abs(y_test - y_pred_dt))  # 📏 Mean Absolute Error (MAE)
mse_dt <- mean((y_test - y_pred_dt)^2)  # 🔧 Mean Squared Error (MSE)
rmse_dt <- sqrt(mse_dt)  # 🔍 Root Mean Squared Error (RMSE)
r2_dt <- 1 - sum((y_test - y_pred_dt)^2) / sum((y_test - mean(y_test))^2)  # 🎯 R² Score
mdape_dt <- median(abs((y_test - y_pred_dt) / (y_test + 1e-10))) * 100  # 📊 Median Absolute Percentage Error (MdAPE)

# 🌲 Train the Random Forest model
rf_model <- randomForest(y_train ~ ., data = data.frame(X_train, y_train), ntree = 100)  # 🌳 Train with 100 trees

# 🔮 Make predictions using Random Forest
y_pred_rf <- predict(rf_model, newdata = data.frame(X_test))

# 📊 Evaluate Random Forest model performance
mae_rf <- mean(abs(y_test - y_pred_rf))  # 📏 Mean Absolute Error (MAE)
mse_rf <- mean((y_test - y_pred_rf)^2)  # 🔧 Mean Squared Error (MSE)
rmse_rf <- sqrt(mse_rf)  # 🔍 Root Mean Squared Error (RMSE)
r2_rf <- 1 - sum((y_test - y_pred_rf)^2) / sum((y_test - mean(y_test))^2)  # 🎯 R² Score
mdape_rf <- median(abs((y_test - y_pred_rf) / (y_test + 1e-10))) * 100  # 📊 Median Absolute Percentage Error (MdAPE)

# ✅ Create a comparison table with updated metrics
comparison <- data.frame(
  Model = c("Decision Tree", "Random Forest"),  # 🏆 Model names
  R2 = c(r2_dt, r2_rf),  # 🎯 R² Score
  MAE = c(mae_dt, mae_rf),  # 📏 Mean Absolute Error
  MdAPE = c(mdape_dt, mdape_rf),  # 📊 Median Absolute Percentage Error
  RMSE = c(rmse_dt, rmse_rf),  # 🔍 Root Mean Squared Error
  MSE = c(mse_dt, mse_rf)  # 🔧 Mean Squared Error
)

# ✅ Sort by best R² score
comparison <- comparison[order(-comparison$R2), ]

# ✅ Create a formatted table for model metrics
metrics_table <- data.frame(
  Metric = c("R² Score", "Mean Absolute Error (MAE)", "Median Absolute Percentage Error (MdAPE)", 
             "Root Mean Squared Error (RMSE)", "Mean Squared Error (MSE)"),
  Decision_Tree = round(c(r2_dt, mae_dt, mdape_dt, rmse_dt, mse_dt), 6),  
  Random_Forest = round(c(r2_rf, mae_rf, mdape_rf, rmse_rf, mse_rf), 6)  
)

# 📊 Display the table using gt()
gt(metrics_table) %>%
  tab_header(title = "Model Performance Comparison") %>%
  fmt_number(columns = c(Decision_Tree, Random_Forest), decimals = 6) %>%
  tab_options(
    column_labels.font.weight = "bold",  
    table.border.top.color = "black",  
    table.border.bottom.color = "black"  
  )

# ✅ Check for missing values before plotting
if (any(is.na(comparison))) {
  stop("🚨 Error: comparison table contains missing values (NA). Check model calculations.")
}

if (dev.cur() > 1) dev.off()

# 🌳 Visualizing the Decision Tree structure
par(xpd = NA)  # Ensure text is not clipped
plot(dt_model, uniform = TRUE, main = "Decision Tree Structure")  # 📊 Plot Decision Tree
text(dt_model, use.n = TRUE, all = TRUE, cex = 0.8)  # 📝 Add labels to tree nodes

# ✅ Define Colors
colors <- c("Decision Tree" = "green3", "Random Forest" = "blue3")

# 📊 Plot R² Score Comparison
ggplot(data = comparison, aes(x = reorder(Model, -R2), y = R2, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6, show.legend = FALSE) +  
  geom_text(aes(label = round(R2, 4)), vjust = -0.5, color = "black", size = 5) +  
  scale_fill_manual(values = colors) +
  ggtitle("R² Score Comparison") +  
  xlab("Model") +  
  ylab("R² Score") +  
  coord_flip() +  
  theme_light() + theme(text = element_text(size = 14))

# 📊 Plot MAE Comparison
ggplot(data = comparison, aes(x = reorder(Model, -MAE), y = MAE, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6, show.legend = FALSE) +  
  geom_text(aes(label = round(MAE, 6)), vjust = -0.5, color = "black", size = 5) +  
  scale_fill_manual(values = colors) +
  ggtitle("Mean Absolute Error (MAE) Comparison") +  
  xlab("Model") +  
  ylab("Mean Absolute Error (MAE)") +  
  coord_flip() +  
  theme_light() + theme(text = element_text(size = 14))

# 📊 Plot MdAPE Comparison
ggplot(data = comparison, aes(x = reorder(Model, -MdAPE), y = MdAPE, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6, show.legend = FALSE) +  
  geom_text(aes(label = paste0(signif(MdAPE, 3), "%")), vjust = -0.5, color = "black", size = 5) +  
  scale_fill_manual(values = colors) +
  ggtitle("Median Absolute Percentage Error (MdAPE) Comparison") +  
  xlab("Model") +  
  ylab("MdAPE (%)") +  
  coord_flip() +  
  theme_light() + theme(text = element_text(size = 14))

# 📊 Plot RMSE Comparison
ggplot(data = comparison, aes(x = reorder(Model, -RMSE), y = RMSE, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6, show.legend = FALSE) +  
  geom_text(aes(label = round(RMSE, 4)), vjust = -0.5, color = "black", size = 5) +  
  scale_fill_manual(values = colors) +
  ggtitle("Root Mean Squared Error (RMSE) Comparison") +  
  xlab("Model") +  
  ylab("Root Mean Squared Error (RMSE)") +  
  coord_flip() +  
  theme_light() + theme(text = element_text(size = 14))

# 📊 Plot MSE Comparison
ggplot(data = comparison, aes(x = reorder(Model, -MSE), y = MSE, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6, show.legend = FALSE) +  
  geom_text(aes(label = round(MSE, 6)), vjust = -0.5, color = "black", size = 5) +  
  scale_fill_manual(values = colors) +
  ggtitle("Mean Squared Error (MSE) Comparison") +  
  xlab("Model") +  
  ylab("Mean Squared Error (MSE)") +  
  coord_flip() +  
  theme_light() + theme(text = element_text(size = 14))