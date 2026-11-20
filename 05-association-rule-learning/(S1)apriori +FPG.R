# Install necessary packages (only if not installed)
# install.packages("arules")
# install.packages("arulesViz")
# install.packages("ggplot2")
# install.packages("dplyr")

# Load required libraries
library(arules)  # Association rules
library(arulesViz)  # Visualization for association rules
library(ggplot2)  # Data visualization
library(dplyr)  # Data manipulation

# 🔹 Step 1: Load and Preprocess Data
cat("\n🔹 Loading dataset...\n")
df <- read.csv("Groceries_dataset.csv", stringsAsFactors = FALSE)  # Read dataset

# Convert transactions into a list format
cat("\n🔹 Transforming transactions...\n")
transactions <- split(df$itemDescription, paste(df$Member_number, df$Date))

# 🔹 Step 2: Identify Top 100 Frequent Items
cat("\n🔹 Identifying top-100 frequent items...\n")
item_counts <- sort(table(unlist(transactions)), decreasing = TRUE)  # Count items
top_items <- names(item_counts[1:100])  # Selecting top 100 items

# 🔹 Step 3: Filter Transactions
filtered_transactions <- lapply(transactions, function(x) x[x %in% top_items])
filtered_transactions <- filtered_transactions[lengths(filtered_transactions) > 1]  # Remove empty transactions

# 🔹 Step 4: Convert to Transactions Format
trans <- as(filtered_transactions, "transactions")  # Convert list to transaction format

# ✅ Check if transactions are available
if (length(trans) == 0) {
  stop("\n❌ Error: Empty transaction list after filtering!")
}

# 🔹 Step 5: Run Apriori Algorithm with optimized parameters
cat("\n🔹 Running Apriori Algorithm...\n")
rules_apriori <- apriori(trans, parameter = list(supp = 0.0005, conf = 0.1, minlen = 2))  # Lowered support & confidence

# ✅ Check if Apriori found rules
if (length(rules_apriori) == 0) {
  cat("\n❌ Apriori found no rules! Try lowering supp/conf again.\n")
} else {
  rules_apriori <- sort(rules_apriori, by = "lift", decreasing = TRUE)  # Sort by lift
}

# 🔹 Step 6: Run FP-Growth (ECLAT) with adjusted support
cat("\n🔹 Running ECLAT Algorithm...\n")
rules_fpgrowth <- eclat(trans, parameter = list(supp = 0.002, minlen = 2))  # Lowered support

# ✅ Check if FP-Growth found itemsets
if (length(rules_fpgrowth) == 0) {
  cat("\n❌ FP-Growth found no itemsets! Try lowering supp.\n")
} else {
  rules_fpgrowth <- sort(rules_fpgrowth, by = "support", decreasing = TRUE)  # Sort by support
}

# 🔹 Step 7: Display Top 5 and Top 10 Rules (Tables)
if (length(rules_apriori) > 0) {
  cat("\n📌 Top 5 Association Rules (Apriori):\n")
  inspect(head(rules_apriori, 5))  # Show top 5 rules
  
  cat("\n📌 Top 10 Association Rules (Apriori):\n")
  inspect(head(rules_apriori, 10))  # Show top 10 rules
}

if (length(rules_fpgrowth) > 0) {
  cat("\n📌 Top 5 Itemsets (FP-Growth):\n")
  inspect(head(rules_fpgrowth, 5))  # Show top 5 itemsets
  
  cat("\n📌 Top 10 Itemsets (FP-Growth):\n")
  inspect(head(rules_fpgrowth, 10))  # Show top 10 itemsets
}

# 🔹 Step 8: Visualizing Frequent Items (Top 20)
top_20_items <- as.data.frame(head(item_counts, 20))  # Convert to DataFrame
colnames(top_20_items) <- c("Item", "Frequency")  # Rename columns

ggplot(top_20_items, aes(x = reorder(Item, -Frequency), y = Frequency)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Top 20 Frequent Items in Transactions", x = "Items", y = "Frequency")

cat("\n✅ Script completed successfully!\n")