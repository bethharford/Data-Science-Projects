# Retail sales SQL Analysis 
# Author: Beth Harford
# Tools: Python, SQLite, Pandas

#This project explores a retail dataset using SQL queries
# executed through Python to analyse regional and product sales performance.


# ---------------------------------
# 1. Import libraries
# ---------------------------------

import pandas as pd
import sqlite3

# --------------------------------
# 2. Load dataset 
# --------------------------------
df = pd.read_csv("/Users/bethharford/Library/CloudStorage/OneDrive-TheUniversityofAuckland/Desktop/PROJECTS 2026/data/train.csv")

# ---------------------------------
#3. Create SQlite database 
# ---------------------------------
conn = sqlite3.connect(":memory")
df.to_sql("train", conn, index = False, if_exists="replace")

# ---------------------------------
# 4. Basic exploration 
# ---------------------------------

# First 10 Rows
query = "SELECT * FROM train LIMIT 10;"
result = pd.read_sql_query(query, conn)
print(result)

# ---------------------------------
# 5. Aggregate Analaysis 
# ---------------------------------

# Total revenue 

query = "SELECT SUM(Sales) AS total_sum FROM train;"
df = pd.read_sql_query(query, conn)
print(df)

# Sales by region

query = "SELECT region, SUM(SALES) AS total_sales FROM train GROUP BY Region ORDER by total_sales DESC;"
df_one = pd.read_sql_query(query, conn)
print(df_one)

# Sales by Category 
query = "SELECT category, SUM(Sales) AS total_sales FROM train GROUP BY Category ORDER BY total_sales DESC;"
df_two = pd.read_sql_query(query,conn)
print(df_two)

# Minimum sales per region
query = "SELECT region, MIN(SALES) AS min_sales from train GROUP BY region ORDER by min_sales ASC;"
df_three = pd.read_sql_query(query,conn)
print(df_three)

# Average sales per region
query = "SELECT region, AVG(SALES) AS avg_sales from train GROUP BY region;"
df_four = pd.read_sql_query(query, conn)
print(df_four)

# Region with the highest sales variability 
query = "SELECT region, MAX(Sales) - MIN(Sales) AS sales_range from train GROUP BY region;"
df_five = pd.read_sql_query(query, conn)
print(df_five)

# Top five products by total sales
query = 'SELECT "Product Name", SUM(Sales) as top_sales FROM train GROUP BY "Product Name" ORDER BY top_sales DESC LIMIT 5;'
df_six = pd.read_sql_query(query, conn)
print(df_six)

# Average sales per category 
query = "SELECT category, AVG(Sales) as avg_sales from train GROUP BY category;"
df_seven = pd.read_sql_query(query, conn)
print(df_seven)

# Number of Transactions per region
query = "SELECT region, COUNT(*) AS transactions FROM train GROUP BY region ORDER BY transactions DESC;"
df_eight = pd.read_sql_query(query, conn)
print(df_eight)

# Sales metrics per region 
query ="SELECT region, COUNT(*) AS transactions, SUM(Sales) AS total_sales, AVG(Sales) AS avg_sale FROM train GROUP BY Region ORDER BY total_sales DESC;"
df_nine = pd.read_sql_query(query, conn)
print(df_nine)

# Highest single sale per region 
query = "SELECT region, MAX(Sales) AS highest_sale FROM train GROUP BY Region ORDER BY highest_sale DESC;"
df_ten = pd.read_sql_query(query, conn)
print(df_ten)

# ---------------------------------
# 6. Advanced SQL analysis 
# ---------------------------------

# Regions above average 

query = "SELECT region, SUM(Sales) AS total_sales FROM train GROUP BY region HAVING SUM(Sales) > (SELECT AVG(total_sales) FROM (SELECT region, SUM(Sales) AS total_sales FROM train GROUP BY region));"
df_eleven = pd.read_sql_query(query, conn)
print(df_eleven)

# Regions below average 
query = "SELECT region, SUM(Sales) AS total_sales FROM train GROUP BY region HAVING SUM(Sales) < (SELECT AVG(total_sales) FROM (SELECT region, SUM(Sales) AS total_sales FROM train GROUP BY region));"
df_twelve = pd.read_sql_query(query, conn)
print(df_twelve)

# Regions above maximum regional total - 10%

query = "SELECT region, SUM(Sales) AS total_sales FROM train GROUP BY region HAVING SUM(Sales) > (  SELECT MAX(total_sales) * 0.9 FROM (SELECT region, SUM(Sales) AS total_sales FROM train GROUP BY region));"
df_thirteen = pd.read_sql_query(query, conn)
print(df_thirteen)

# Top performing region per category 

query = "SELECT category, region, SUM(Sales) AS total_sales FROM train GROUP BY category, region ORDER BY category, total_sales DESC;" 
df_fourteen = pd.read_sql_query(query, conn)
print(df_fourteen)

# -------------------------------
# 7. Close database connection 
# -------------------------------

conn = sqlite3.connect(":memory:")

# Save results 
df_one.to_csv("sales_by_region.csv", index=False)
df_two.to_csv("sales_by_category.csv", index=False)
df_nine.to_csv("regional_sales_metrics.csv", index=False)