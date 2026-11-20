
# 📌 Hierarchical Clustering in R
# Hierarchical clustering is a powerful unsupervised learning technique that builds a hierarchy of clusters.
# Unlike K-Means, which requires predefining the number of clusters (k), hierarchical clustering organizes data into a tree-like structure (dendrogram).
# 
# Two main types of hierarchical clustering:
# 1️⃣ **Agglomerative (Bottom-Up)** - Starts with each data point as its own cluster and merges them iteratively.
# 2️⃣ **Divisive (Top-Down)** - Starts with all data points in one cluster and splits them recursively.
# 
# We will implement both Agglomerativ (`hclust`) and Divisive (`diana`) clustering in R.

# 📥 Load Required Libraries
library(cluster)      # For divisive clustering (DIANA)
library(factoextra)   # For visualization of clustering results
library(NbClust)      # For determining the optimal number of clusters
library(ggplot2)      # For enhanced visualization

# 📂 Load Dataset
# We'll use a sample dataset similar to "Mall_Customers.csv"
data <- read.csv("Mall.csv")

# 📊 Select relevant features (Annual Income & Spending Score)
data_selected <- data[, c(4,5)]  # Selecting numerical columns

# ⚖️ Standardize the data to ensure all variables have equal weight
scaled_data <- scale(data_selected)

# 🏗️ Perform Agglomerative Hierarchical Clustering (HCLUST)
dist_matrix <- dist(scaled_data, method = "euclidean")  # Compute distance matrix
hc_agg <- hclust(dist_matrix, method = "ward.D2")  # Apply Ward's method for clustering

# 🎨 Plot the dendrogram for Agglomerative Clustering
plot(hc_agg, labels = FALSE, main = "Agglomerative Clustering Dendrogram", sub = "Ward's Method")
abline(h = 7, col = "red")  # Draw horizontal line to cut the tree at a certain height

# 🏗️ Perform Divisive Hierarchical Clustering (DIANA)
diana_model <- diana(dist_matrix)  # Compute DIANA clustering

# 🎨 Plot the dendrogram for Divisive Clustering
plot(diana_model, labels = FALSE, main = "Divisive Clustering Dendrogram", sub = "DIANA Method")
abline(h = 7, col = "blue")  # Draw horizontal line to cut the tree

# 📈 Determine the optimal number of clusters
nb_clusters <- NbClust(scaled_data, distance = "euclidean", min.nc = 2, max.nc = 10, method = "ward.D2")

# 🎨 Visualize clusters for Agglomerative Clustering
fviz_cluster(list(data = scaled_data, cluster = cutree(hc_agg, k = 5)), geom = "point", ellipse.type = "convex", ggtheme = theme_minimal())

# 🎨 Visualize clusters for Divisive Clustering
fviz_cluster(list(data = scaled_data, cluster = cutree(as.hclust(diana_model), k = 5)), geom = "point", ellipse.type = "convex", ggtheme = theme_minimal())

# 📊 Compute Cluster Validity Metrics
library(clusterCrit)  # Package for computing cluster validity indices
labels_agg <- cutree(hc_agg, k = 5)
labels_div <- cutree(as.hclust(diana_model), k = 5)

silhouette_agg <- mean(silhouette(labels_agg, dist_matrix)[, 3])  # Compute silhouette score for Agglomerative
silhouette_div <- mean(silhouette(labels_div, dist_matrix)[, 3])  # Compute silhouette score for Divisive

cat("📊 Silhouette Score - Agglomerative: ", silhouette_agg, "\n")
cat("📊 Silhouette Score - Divisive: ", silhouette_div, "\n")

# 📊 Compute Additional Cluster Metrics

# 📌 Compute Davies-Bouldin Index (Lower is better) - Measures cluster compactness & separation
davies_bouldin_agg <- intCriteria(scaled_data, labels_agg, "Davies_Bouldin")$davies_bouldin  # Agglomerative
davies_bouldin_div <- intCriteria(scaled_data, labels_div, "Davies_Bouldin")$davies_bouldin  # Divisive

# 📌 Compute Calinski-Harabasz Index (Higher is better) - Evaluates intra-cluster vs inter-cluster variance
calinski_harabasz_agg <- intCriteria(scaled_data, labels_agg, "Calinski_Harabasz")$calinski_harabasz  # Agglomerative
calinski_harabasz_div <- intCriteria(scaled_data, labels_div, "Calinski_Harabasz")$calinski_harabasz  # Divisive

# 📝 Create Performance Table

# 📌 Construct a DataFrame to store clustering performance metrics
performance_table <- data.frame(
  Method = c("Agglomerative", "Divisive"),  # Names of clustering methods
  Silhouette_Score = c(silhouette_agg, silhouette_div),  # Silhouette Score (Higher is better)
  Davies_Bouldin = c(davies_bouldin_agg, davies_bouldin_div),  # Davies-Bouldin Index (Lower is better)
  Calinski_Harabasz = c(calinski_harabasz_agg, calinski_harabasz_div)  # Calinski-Harabasz Index (Higher is better)
)

# 📊 Print the Performance Table
print(performance_table)  # Display final comparison of clustering models


# 📥 Load Required Libraries
library(ggplot2)       # For enhanced visualization
library(ggforce)       # For ellipse-based clustering visualization

# 🏷️ Prepare Data for Visualization
clustered_data <- data_selected  # Use the dataset with selected features
colnames(clustered_data) <- c("Annual_Income", "Spending_Score")  # Assign correct column names

# Add cluster labels and ID numbers
clustered_data$Cluster_Agg <- as.factor(labels_agg)  # Cluster labels for Agglomerative clustering
clustered_data$Cluster_Div <- as.factor(labels_div)  # Cluster labels for Divisive clustering
clustered_data$ID <- 1:nrow(clustered_data)  # Assign unique IDs to each data point

# 🎨 **Enhanced Visualization for Agglomerative Clustering**
ggplot(clustered_data, aes(x = Annual_Income, y = Spending_Score, color = Cluster_Agg)) +
  geom_point(size = 2) +  # Plot data points
  geom_mark_ellipse(aes(fill = Cluster_Agg), alpha = 0.2) +  # Add cluster ellipses
  geom_text(aes(label = ID), size = 3, vjust = -0.5) +  # Display data point IDs
  theme_minimal() +  # Apply minimal theme
  labs(title = "Agglomerative Clustering Results", x = "Annual Income", y = "Spending Score")  # Set plot title and axis labels

# 🎨 **Enhanced Visualization for Divisive Clustering**
ggplot(clustered_data, aes(x = Annual_Income, y = Spending_Score, color = Cluster_Div)) +
  geom_point(size = 2) +  # Plot data points
  geom_mark_ellipse(aes(fill = Cluster_Div), alpha = 0.2) +  # Add cluster ellipses
  geom_text(aes(label = ID), size = 3, vjust = -0.5) +  # Display data point IDs
  theme_minimal() +  # Apply minimal theme
  labs(title = "Divisive Clustering Results", x = "Annual Income", y = "Spending Score")  # Set plot title and axis labels