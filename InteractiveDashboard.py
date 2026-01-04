# =====================================
# INTERACTIVE DASHBOARD WITH DROPDOWN
# =====================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month, sum as _sum
import pandas as pd
import plotly.graph_objects as go

# =====================================
# STEP 1: START SPARK SESSION
# =====================================
spark = SparkSession.builder \
    .appName("Interactive Grocery Sales Dashboard") \
    .getOrCreate()

# =====================================
# STEP 2: LOAD DATASET
# =====================================
df = spark.read.csv(
    "file:///home/hduser/Downloads/archive(1)/grocerydata.csv",
    header=True,
    inferSchema=True
)

# =====================================
# STEP 3: DATA PREPARATION
# =====================================
df = df.withColumn("transaction_date", to_date(col("transaction_date"), "yyyy-MM-dd"))
df = df.dropna()
df = df.withColumn("Year", year(col("transaction_date"))) \
       .withColumn("Month", month(col("transaction_date")))

pdf = df.select(
    "aisle", "store_name", "quantity", "unit_price", "total_amount",
    "discount_amount", "final_amount", "Year", "Month"
).toPandas()

# -------------------------------------
# Prepare aggregated data
# -------------------------------------

# 1️⃣ Monthly Sales Trend
monthly_sales = df.groupBy("Year", "Month") \
    .agg(_sum("final_amount").alias("MonthlySales")) \
    .orderBy("Year", "Month") \
    .toPandas()
monthly_sales["YearMonth"] = monthly_sales["Year"].astype(str) + "-" + monthly_sales["Month"].astype(str)

# 2️⃣ Store-wise Sales Trend
store_sales = df.groupBy("Year", "Month", "store_name") \
    .agg(_sum("final_amount").alias("Sales")) \
    .orderBy("Year", "Month") \
    .toPandas()
pivot_store = store_sales.pivot_table(
    index=["Year", "Month"], columns="store_name", values="Sales", aggfunc="sum"
).fillna(0)
pivot_store["YearMonth"] = pivot_store.index.map(lambda x: f"{x[0]}-{x[1]}")

# =====================================
# STEP 4: CREATE DROPDOWN DASHBOARD
# =====================================

fig = go.Figure()

# --------- Trace 1: Histogram Final Amount ---------
fig.add_trace(go.Histogram(
    x=pdf["final_amount"],
    name="Final Amount Distribution",
    marker_color="#1f77b4",
    visible=True  # Default visible
))

# --------- Trace 2: Scatter Total vs Final ---------
fig.add_trace(go.Scatter(
    x=pdf["total_amount"],
    y=pdf["final_amount"],
    mode="markers",
    marker=dict(size=8, color=pdf["aisle"].astype("category").cat.codes, showscale=True),
    hovertemplate="Store: %{customdata[0]}<br>Discount: %{customdata[1]}<br>Total: %{x}<br>Final: %{y}",
    customdata=pdf[["store_name", "discount_amount"]],
    name="Total vs Final Amount",
    visible=False
))

# --------- Trace 3: Monthly Sales Trend ---------
fig.add_trace(go.Scatter(
    x=monthly_sales["YearMonth"],
    y=monthly_sales["MonthlySales"],
    mode="lines+markers",
    line=dict(color="#ff7f0e", width=3),
    name="Monthly Sales Trend",
    visible=False
))

# --------- Trace 4: Store-wise Sales Trend ---------
for col_name in pivot_store.columns[:-1]:  # exclude YearMonth column
    fig.add_trace(go.Scatter(
        x=pivot_store["YearMonth"],
        y=pivot_store[col_name],
        mode="lines+markers",
        name=col_name,
        visible=False
    ))

# --------- Dropdown Menu ---------
dropdown_buttons = [
    {"label": "Final Amount Distribution", "method": "update",
     "args": [{"visible": [True] + [False]* (len(fig.data)-1)}, {"title": "Final Amount Distribution"}]},
    {"label": "Total vs Final Amount", "method": "update",
     "args": [{"visible": [False, True] + [False]* (len(fig.data)-2)}, {"title": "Total vs Final Amount"}]},
    {"label": "Monthly Sales Trend", "method": "update",
     "args": [{"visible": [False, False, True] + [False]* (len(fig.data)-3)}, {"title": "Monthly Sales Trend"}]},
    {"label": "Store-wise Sales Trend", "method": "update",
     "args": [{"visible": [False]*3 + [True]*(len(fig.data)-3)}, {"title": "Store-wise Sales Trend"}]},
]

fig.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=dropdown_buttons,
        x=0.1,
        y=1.15,
        xanchor="left",
        yanchor="top"
    )],
    template="plotly_white",
    title="Interactive Grocery Sales Dashboard",
    xaxis_title="X-axis",
    yaxis_title="Y-axis",
    hovermode="closest"
)

# Save as HTML
fig.write_html("interactive_dashboard.html")

print("✅ Interactive dashboard created: interactive_dashboard.html")

# =====================================
# STEP 5: STOP SPARK
# =====================================
spark.stop()

