# 📂 Datasets

All ten files are shared by the scripts; each script reads what it needs with a path such as `data/mall.csv`.

| File | Rows × columns | Used in | What it is |
|---|:-:|---|---|
| [`data-preprocessing.csv`](data-preprocessing.csv) | 10 × 4 | Section 1 | Country, age, salary, purchased: a 10-row toy table |
| [`salary-data.csv`](salary-data.csv) | 30 × 2 | Section 2 | Years of experience vs salary |
| [`50-startups.csv`](50-startups.csv) | 50 × 5 | Section 2 | R&D, administration, marketing spend, state, profit |
| [`position-salaries.csv`](position-salaries.csv) | 10 × 3 | Section 2 | Position level vs salary |
| [`housing.csv`](housing.csv) | 20,640 × 10 | Section 2 | California housing: location, rooms, income, house value |
| [`social-network-ads.csv`](social-network-ads.csv) | 400 × 5 | Section 3 | Age, estimated salary, purchased (0/1) |
| [`mall.csv`](mall.csv) | 200 × 5 | Section 4 | Annual income and spending score of mall customers |
| [`groceries.csv`](groceries.csv) | 38,765 × 3 | Section 5 | Member, date, item: one row per purchased item |
| [`iris.csv`](iris.csv) | 150 × 6 | not used by an R script | The classic Iris flowers |
| [`breast-cancer-wisconsin.csv`](breast-cancer-wisconsin.csv) | 569 × 33 | not used by an R script | Breast-cancer diagnosis (569 samples, 30 measurements) |

> The old `data` branch contained both `Data.csv` and `data.csv`. On Windows and macOS (case-insensitive file systems) those two cannot exist in one folder, so they now have distinct names: `data-preprocessing.csv` and `breast-cancer-wisconsin.csv`.

---

*Built by [Evgenii Matveev](https://github.com/evgeniimatveev) · Data Analyst / Analytics & MLOps Engineer*
