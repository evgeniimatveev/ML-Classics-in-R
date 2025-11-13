# ml_classics_intro.R
# Quick overview script for the "ML Classics in R" repository
# All comments are in English for GitHub readability.

# 1. Short message about the project ---------------------------------------

message("📊 ML Classics in R")
message("This repository collects classic machine learning algorithms implemented in R.")
message("It is structured into five main parts: preprocessing, regression, classification, clustering, and association rules.")
message("Source: SuperDataScience Machine Learning A-Z (R).")

# 2. Overview of parts and example algorithms ------------------------------

ml_parts <- data.frame(
  part = c(
    "Part 1 - Data Preprocessing",
    "Part 2 - Regression",
    "Part 3 - Classification",
    "Part 4 - Clustering",
    "Part 5 - Association Rule Learning"
  ),
  examples = c(
    "Missing values, encoding, feature scaling",
    "Linear, Polynomial, SVR, Decision Tree, Random Forest",
    "Logistic, KNN, SVM, Decision Tree, Random Forest",
    "K-Means, Hierarchical Clustering",
    "Apriori, Eclat"
  ),
  stringsAsFactors = FALSE
)

cat("\n📌 Repository structure (high-level):\n")
print(ml_parts, row.names = FALSE)

# 3. Helper function: list supported libraries -----------------------------

ml_required_packages <- c(
  "ggplot2", "dplyr", "e1071", "randomForest", "caret", "arules"
)

describe_dependencies <- function(pkgs) {
  cat("\n🔧 Recommended R packages for this project:\n")
  for (pkg in pkgs) {
    cat("- ", pkg, "\n", sep = "")
  }
  cat("\nTip: run install.packages() for any missing package before executing the scripts.\n")
}

describe_dependencies(ml_required_packages)
