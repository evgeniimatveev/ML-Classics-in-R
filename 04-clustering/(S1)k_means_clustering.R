# ======================================================
# 📌 K-Means Clustering: Unsupervised Machine Learning
# ======================================================
# K-Means is a clustering algorithm that partitions data into K clusters.
# It assigns each point to the nearest cluster centroid and iteratively refines the clusters.
# The goal is to minimize intra-cluster variance and maximize inter-cluster separation.
# Common applications:
# ✅ Customer Segmentation (Marketing & Business)
# ✅ Image Compression (Reducing colors in images)
# ✅ Anomaly Detection (Finding outliers)
# ✅ Document Clustering (Grouping similar texts)
# ✅ Gene Expression Analysis (Bioinformatics)


# 📌 Step 1: Load necessary libraries
library(ggplot2)  # For creating visualizations
library(cluster)  # For cluster analysis and visualization
library(factoextra)  # For advanced clustering visualization
library(dplyr)  # For data manipulation

# 📌 Step 2: Load the dataset
dataset <- read.csv('mall.csv')  # Load the dataset from CSV file
X <- dataset[, 4:5]  # Select columns "Annual Income" and "Spending Score"

# 📌 Step 3: Normalize the data (scaling for better clustering results)
X_scaled <- scale(X)  # Standardize the features

# 📌 Step 4: Find the optimal number of clusters using the Elbow Method
set.seed(6)  # Set a fixed seed for reproducibility
wcss <- vector()  # Create an empty vector to store Within-Cluster Sum of Squares (WCSS)

# Loop through 1 to 10 clusters and calculate WCSS for each
for (i in 1:10) {
  wcss[i] <- sum(kmeans(X_scaled, centers = i, nstart = 25)$withinss)
}

# 📌 Step 5: Plot the Elbow Method graph
plot(1:10, wcss, type = "b",  # Type 'b' for both points and lines
     main = "The Elbow Method",  # Title of the plot
     xlab = "Number of clusters",  # Label for x-axis
     ylab = "WCSS",  # Label for y-axis
     col = "blue", pch = 19)  # Set color and point shape

# 📌 Step 6: Find the optimal number of clusters using Silhouette Score
set.seed(6)  # Ensure reproducibility
sil_scores <- c()  # Create an empty vector for silhouette scores

# Loop through 2 to 10 clusters to calculate the average Silhouette Score
for (k in 2:10) {
  km_model <- kmeans(X_scaled, centers = k, nstart = 25)  # Apply K-Means clustering
  silhouette_avg <- mean(silhouette(km_model$cluster, dist(X_scaled))[, 3])  # Compute silhouette score
  sil_scores <- c(sil_scores, silhouette_avg)  # Store the silhouette score
}

# Find the best K using the maximum silhouette score
best_k <- which.max(sil_scores) + 1  # Adjust index (since we started from 2)
print(paste("🔍 Optimal number of clusters based on Silhouette Score:", best_k))

# 📌 Step 7: Fit K-Means model with optimal clusters
set.seed(29)  # Set seed for reproducibility
kmeans <- kmeans(X_scaled, centers = best_k, iter.max = 300, nstart = 25)  # Train the model

# 📌 Step 8: Compute clustering metrics
silhouette_score <- mean(silhouette(kmeans$cluster, dist(X_scaled))[, 3])  # Compute silhouette score
withinss_total <- sum(kmeans$withinss)  # Compute total Within-Cluster Sum of Squares (WCSS)

# 📌 Step 9: Display clustering performance metrics
metrics_table <- data.frame(
  Algorithm = "K-Means++",  # Name of algorithm
  Optimal_K = best_k,  # Optimal number of clusters
  Silhouette_Score = silhouette_score,  # Silhouette score
  Total_WCSS = withinss_total  # Total WCSS
)

# Print metrics to console
print("\n📊 Clustering Performance Metrics:")
print(metrics_table)

# 📌 Step 10: Visualizing Clusters with Clusplot (Basic)
clusplot(X_scaled, kmeans$cluster,  # Data and cluster labels
         lines = 0,  # No lines between clusters
         shade = TRUE,  # Add shading for clusters
         color = TRUE,  # Use colors for clusters
         labels = 2,  # Add labels for clusters
         plotchar = FALSE,  # No special characters for points
         span = TRUE,  # Expand to fill the space
         main = "Clusters of Customers",  # Title of the plot
         xlab = "Annual Income",  # X-axis label
         ylab = "Spending Score")  # Y-axis label

# 📌 Step 11: Advanced Visualization with ggplot2 + factoextra
fviz_cluster(kmeans, data = X_scaled,  # Data and clustering model
             geom = "point",  # Use points for visualization
             ellipse.type = "convex",  # Use convex hulls to show clusters
             palette = "jco",  # Use a specific color palette
             ggtheme = theme_minimal(),  # Apply a clean theme
             main = paste("K-Means Clustering (k =", best_k, ")"))  # Title with optimal K