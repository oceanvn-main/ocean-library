"""
spark_phase1_ingest.py
======================
Phase 1: Data Acquisition & Dimensionality Profiling using Apache Spark.

This is the industry-standard equivalent of TransactionalReader.m.
It reads the Instacart relational CSVs, performs JOIN + GROUP BY,
profiles the dimensions, and exports clean baskets.

Usage:
    python spark_phase1_ingest.py

Output:
    Data/InstaCart/baskets_department.parquet
    Data/InstaCart/baskets_aisle.parquet
    Data/InstaCart/baskets_product.parquet
"""

# %% Imports & Configuration
import os
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] += r";C:\hadoop\bin"

import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set, size, countDistinct, col, avg as spark_avg
#collect_set gather unique values into an array per group, size: count elemetns in an aray cols, countDistinct: count distinct values in a column, col: select a column, avg: compute the average of a column

# ============================================================
#  CONFIGURATION
# ============================================================
DATA_DIR = r"C:\Ocean Library\Matlab code practice\Data\InstaCart"
OUTPUT_DIR = DATA_DIR  # Save parquet alongside the CSVs

# %% STEP 0: Start Spark Engine
print("=" * 60)
print("  MARKET BASKET ANALYSIS — PHASE 1 (Apache Spark)")
print("=" * 60)

#local: this machine, [*]: use ALL CPU cores
# 4g: 4GB RAM to work with
# 8: when reshuffling data across partitions, split into 8 chunks
spark = SparkSession.builder \
    .appName("MBA_Phase1_Instacart") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# Suppress noisy logs
spark.sparkContext.setLogLevel("ERROR")

print(f"\n  Spark {spark.version} started")
print(f"  Data directory: {DATA_DIR}\n")

# %% STEP 1: Read raw CSVs
print("-" * 60)
print("  Step 1: Reading raw CSV files...")
print("-" * 60)

t0 = time.time()

order_products = spark.read.csv(
    f"{DATA_DIR}/order_products__prior.csv",
    header=True, inferSchema=True
)

products = spark.read.csv(
    f"{DATA_DIR}/products.csv",
    header=True, inferSchema=True
)

aisles = spark.read.csv(
    f"{DATA_DIR}/aisles.csv",
    header=True, inferSchema=True
)

departments = spark.read.csv(
    f"{DATA_DIR}/departments.csv",
    header=True, inferSchema=True
)

t_read = time.time() - t0
print(f"  CSVs loaded in {t_read:.1f}s")
print(f"  order_products: {order_products.count():,} rows")
print(f"  products:       {products.count():,} rows")
print(f"  aisles:         {aisles.count():,} rows")
print(f"  departments:    {departments.count():,} rows")

# %% STEP 2: JOIN + GROUP BY
#
#  This is the core of Phase 1. In SQL, it would be:
#
#    SELECT order_id, COLLECT_SET(aisle_id) AS basket
#    FROM order_products op
#    JOIN products p ON op.product_id = p.product_id
#    GROUP BY order_id
#
#  Spark does this in parallel across partitions.

# Join products to get aisle_id and department_id
op_enriched = order_products.join(products, "product_id", "inner") \
    .select("order_id", "product_id", "aisle_id", "department_id")

# Cache — this intermediate result is reused 3 times
op_enriched.cache()

# %% STEP 2b: Profile at each granularity level
granularities = {
    "department":  ("department_id", departments, "department"),
    "aisle":       ("aisle_id",      aisles,      "aisle"),
    "product":     ("product_id",    products,    "product_name"),
}

profiles = {}

for gran_name, (id_col, label_df, label_col) in granularities.items():
    print(f"\n{'=' * 60}")
    print(f"  Granularity: {gran_name.upper()}")
    print(f"{'=' * 60}")

    t0 = time.time()

    # GROUP BY order_id, collect unique items per basket
    #.groupBy("order_id") piles rows with the same order_id together ->  [120, 120, 83]
    #agg = combine many values into one
    #.alias = name this new column "basket"
    #id_col = pick this column
    baskets_df = op_enriched.groupBy("order_id") \
        .agg(collect_set(col(id_col)).alias("basket"))

    # Compute profile metrics
    N = baskets_df.count() #How many rows
    p = op_enriched.select(id_col).distinct().count() #How many unique values in this column

    #agg without groupBy = aggregate entire table and produce one value(depend on operation to decide which value)
    basket_stats = baskets_df.select(
        size("basket").alias("basket_size") #size = count elements in an array
    ).agg(
        spark_avg("basket_size").alias("avg_size"), #avg = compute the average of a column
    ).collect()[0]

    avg_basket = basket_stats["avg_size"]
    #[0] first get the row. [0] second get the value inside the row
    total_items = baskets_df.select(size("basket").alias("s")).groupBy().sum("s").collect()[0][0]
    sparsity = 1.0 - total_items / (N * p)

    dense_memory_bytes = N * p
    sparse_recommended = sparsity > 0.90

    t_elapsed = time.time() - t0

    # Print profile
    print(f"  Transactions (N)    : {N:,}")
    print(f"  Unique Items (p)    : {p:,}")
    print(f"  Avg Basket Size     : {avg_basket:.1f} items")
    print(f"  Sparsity Ratio      : {sparsity * 100:.4f}%")
    print(f"  Dense Memory Est.   : {dense_memory_bytes / 1024**3:.1f} GB" if dense_memory_bytes > 1024**3
          else f"  Dense Memory Est.   : {dense_memory_bytes / 1024**2:.1f} MB")
    print(f"  Sparse Recommended  : {sparse_recommended}")
    print(f"  Wall-clock time     : {t_elapsed:.1f}s")

    profiles[gran_name] = {
        "N": N, "p": p, "avg_basket": avg_basket,
        "sparsity": sparsity, "time": t_elapsed
    }

    # --------------------------------------------------------
    #  STEP 3: Save clean baskets as Parquet
    # --------------------------------------------------------
    output_path = f"{OUTPUT_DIR}/baskets_{gran_name}.parquet"
    baskets_df.write.mode("overwrite").parquet(output_path)
    print(f"  Saved to: {output_path}")

# %% STEP 3: Comparison Table
print(f"\n{'=' * 60}")
print("  COMPARISON TABLE")
print(f"{'=' * 60}")
print(f"  {'Granularity':<14} | {'N':>12} | {'p':>8} | {'Sparsity':>12} | {'Time':>8}")
print(f"  {'-'*62}")

for gran_name, prof in profiles.items():
    print(f"  {gran_name:<14} | {prof['N']:>12,} | {prof['p']:>8,} | "
          f"{prof['sparsity']*100:>10.4f}% | {prof['time']:>6.1f}s")

# %% STEP 4: Sample Baskets (sanity check)
print(f"\n{'=' * 60}")
print("  SAMPLE BASKETS (Aisle Level — first 5)")
print(f"{'=' * 60}")

# Read aisle names for decoding
aisle_names = {row.aisle_id: row.aisle for row in aisles.collect()}

#Groupby sorts rows into piles 
#collect_set  = collect into a SET ( unique values only)
#agg with groupBy meaning: For each group, take all the aisle_id values and put them into one array, removing duplicates
sample_baskets = op_enriched.groupBy("order_id") \
    .agg(collect_set("aisle_id").alias("basket")) \
    .limit(5).collect()

for i, row in enumerate(sample_baskets): #Enumerate give both index and value
    items = [aisle_names.get(aid, f"[ID:{aid}]") for aid in sorted(row.basket)] #List comprehension
    print(f"\n  Basket {i+1} ({len(row.basket)} items):")
    print(f"    {', '.join(items)}")

# %% STEP 5: Shutdown
print(f"\n{'=' * 60}")
print("  Phase 1 complete.")
print(f"  Output files in: {OUTPUT_DIR}")
print(f"{'=' * 60}\n")

spark.stop()
