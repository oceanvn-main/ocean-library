"""
spark_market_basket_analysis.py
================================
MARKET BASKET ANALYSIS — FULL PIPELINE (ALL 5 PHASES)

Production-grade equivalent of exp_market_basket_analysis.m
Uses Apache Spark MLlib FPGrowth instead of manual Apriori.

Pipeline:
    Phase 1: Data Acquisition         (Spark SQL: read CSVs, JOIN, GROUP BY)
    Phase 2: Encoding                  (Automatic — FPGrowth handles internally)
    Phase 3: Frequent Itemset Mining   (FP-Growth algorithm via MLlib)
    Phase 4: Rule Generation           (Built into FPGrowth model)
    Phase 5: Filtering & Visualization (Pandas + Plotly)

Usage:
    python spark_market_basket_analysis.py
"""

# %% ============================================================
#  IMPORTS & CONFIGURATION
# ============================================================
import os
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] += r";C:\hadoop\bin"

import time
import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    collect_set, size, col, countDistinct,
    avg as spark_avg, explode, array_sort
)
from pyspark.ml.fpm import FPGrowth

DATA_DIR = r"C:\Ocean Library\Matlab code practice\Data\InstaCart"

# Pipeline parameters (same as MATLAB experiment)
MIN_SUPPORT    = 0.05    # 5% (balance between memory and meaningful rules)
MIN_CONFIDENCE = 0.30    # 30%
MIN_LIFT       = 1.0     # Only rules better than random
MAX_RULES      = 50      # Top 50 rules

print("=" * 65)
print("  MARKET BASKET ANALYSIS — FULL PIPELINE (Apache Spark)")
print(f"  minSupport:    {MIN_SUPPORT * 100:.2f}%")
print(f"  minConfidence: {MIN_CONFIDENCE * 100:.2f}%")
print(f"  minLift:       {MIN_LIFT:.2f}")
print(f"  maxRules:      {MAX_RULES}")
print("=" * 65)

# %% ============================================================
#  SPARK ENGINE
# ============================================================
#local: this machine, [*] all CPU cores
#When shufflring, splits into 8 chunks
#Maxsize of data collected back to driver = 4GB
spark = SparkSession.builder \
    .appName("MBA_FullPipeline") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.maxResultSize", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(f"\n  Spark {spark.version} started on {os.cpu_count()} cores\n")


# %% ============================================================
#  PHASE 1: DATA ACQUISITION
#  Read CSVs → JOIN → GROUP BY → Baskets
# ============================================================
print("=" * 65)
print("  PHASE 1: DATA ACQUISITION")
print("=" * 65)

t_phase1 = time.time()

# 1a. Read raw CSVs
order_products = spark.read.csv(f"{DATA_DIR}/order_products__prior.csv", header=True, inferSchema=True)
products       = spark.read.csv(f"{DATA_DIR}/products.csv", header=True, inferSchema=True)
aisles         = spark.read.csv(f"{DATA_DIR}/aisles.csv", header=True, inferSchema=True)

print(f"  order_products: {order_products.count():,} rows")
print(f"  products:       {products.count():,} rows")
print(f"  aisles:         {aisles.count():,} rows")

# 1b. JOIN: order_products → products (to get aisle_id)
#     Equivalent to: TransactionalReader enrichment
enriched = order_products.join(products, "product_id", "inner") \
    .select("order_id", "aisle_id")

# 1c. GROUP BY order_id, collect unique aisle IDs per basket
#     Equivalent to: TransactionalReader basket construction
baskets = enriched.groupBy("order_id") \
    .agg(collect_set("aisle_id").alias("basket"))

baskets.cache()  # Cache — reused in profiling and mining

# 1d. Profile
N = baskets.count()
p = enriched.select("aisle_id").distinct().count()
avg_basket = baskets.select(size("basket").alias("s")) \
    .agg(spark_avg("s")).collect()[0][0]

t_phase1 = time.time() - t_phase1

print(f"\n  Baskets (N):        {N:,}")
print(f"  Unique Aisles (p):  {p}")
print(f"  Avg Basket Size:    {avg_basket:.1f} items")
print(f"  Phase 1 time:       {t_phase1:.1f}s")


# %% ============================================================
#  PHASE 2 + 3: ENCODING + FP-GROWTH MINING
#  Spark MLlib does encoding internally — no manual sparse matrix
# ============================================================
print(f"\n{'=' * 65}")
print("  PHASE 2+3: FP-GROWTH MINING (replaces TransactionEncoder + Apriori)")
print("=" * 65)

t_phase23 = time.time()

# FPGrowth takes the basket column directly — no manual encoding needed
# Internally, it builds a compressed FP-Tree instead of candidate generation
fpg = FPGrowth(
    itemsCol="basket",
    minSupport=MIN_SUPPORT,
    minConfidence=MIN_CONFIDENCE
)

model = fpg.fit(baskets)

# Frequent itemsets (Phase 3 output)
freq_itemsets = model.freqItemsets
freq_itemsets.cache()

# Show itemset breakdown by size (use groupBy to avoid full count)
freq_with_size = freq_itemsets.withColumn("k", size("items")) #withColumn adds a new column without destroying anything
k_breakdown = freq_with_size.groupBy("k").count().orderBy("k").collect() #collect() : pull form Spark to Python list
n_itemsets = sum(row["count"] for row in k_breakdown)

t_phase23 = time.time() - t_phase23

print(f"\n  Frequent itemsets found: {n_itemsets:,}")
print(f"  Phase 2+3 time:         {t_phase23:.1f}s")

print(f"\n  {'k':<6} | {'Count':>10}")
print(f"  {'-' * 20}")
for k_row in k_breakdown:
    print(f"  {k_row['k']:<6} | {k_row['count']:>10,}")


# %% ============================================================
#  PHASE 4: RULE GENERATION
#  Built into FPGrowth model — automatic
# ============================================================
print(f"\n{'=' * 65}")
print("  PHASE 4: RULE GENERATION (built into FPGrowth)")
print("=" * 65)

t_phase4 = time.time()

# Association rules (Phase 4 output) — Spark computes confidence + lift
# Convert directly to Pandas to avoid double materialization
rules_pd = model.associationRules.toPandas()
n_rules = len(rules_pd)

t_phase4 = time.time() - t_phase4

print(f"\n  Rules generated: {n_rules:,}")
print(f"  Phase 4 time:    {t_phase4:.1f}s")


# %% ============================================================
#  PHASE 5: FILTERING & VISUALIZATION
#  Convert to Pandas for filtering, ranking, and Plotly viz
# ============================================================
print(f"\n{'=' * 65}")
print("  PHASE 5: FILTERING & VISUALIZATION")
print("=" * 65)

t_phase5 = time.time()

# 5a. Already in Pandas from Phase 4
print(f"\n  Total rules:     {len(rules_pd):,}")

# 5b. Filter: lift >= MIN_LIFT
rules_pd = rules_pd[rules_pd["lift"] >= MIN_LIFT].copy()
print(f"  After lift filter: {len(rules_pd):,}")

# 5c. Sort by lift (descending) and take top N
rules_pd = rules_pd.nlargest(MAX_RULES, "lift")
print(f"  After top-{MAX_RULES}:     {len(rules_pd)}")

# 5d. Build aisle name lookup for human-readable output
aisle_lookup = {row.aisle_id: row.aisle for row in aisles.collect()}

def items_to_names(item_list):
    """Convert aisle IDs to human-readable names."""
    return " + ".join(aisle_lookup.get(int(aid), f"ID:{aid}") for aid in sorted(item_list))

rules_pd["antecedent_names"] = rules_pd["antecedent"].apply(items_to_names)
rules_pd["consequent_names"] = rules_pd["consequent"].apply(items_to_names)
rules_pd["itemset_size"] = rules_pd.apply(
    lambda r: len(r["antecedent"]) + len(r["consequent"]), axis=1
)

# 5e. Print final rule table
print(f"\n{'=' * 65}")
print(f"  FINAL RULES TABLE (Top {len(rules_pd)} by Lift)")
print(f"{'=' * 65}\n")

print(f"  {'#':<4} | {'IF (buy this)':<30} | {'THEN (also buy)':<22} | {'Sup':>6} | {'Conf':>6} | {'Lift':>6}")
print(f"  {'-' * 85}")

for i, (_, row) in enumerate(rules_pd.iterrows(), 1):
    ant = row["antecedent_names"]
    con = row["consequent_names"]
    if len(ant) > 28: ant = ant[:25] + "..."
    if len(con) > 20: con = con[:17] + "..."

    print(f"  {i:<4} | {ant:<30} | {con:<22} | "
          f"{row['support']*100:5.2f}% | {row['confidence']*100:5.1f}% | {row['lift']:5.2f}")

# 5f. Visualize with Plotly (if available)
try:
    import plotly.express as px
    import plotly.graph_objects as go

    fig = px.scatter(
        rules_pd,
        x="confidence",
        y="lift",
        size="support",
        color="itemset_size",
        hover_data=["antecedent_names", "consequent_names"],
        title="Association Rules: Confidence vs Lift",
        labels={
            "confidence": "Confidence",
            "lift": "Lift",
            "support": "Support",
            "itemset_size": "Itemset Size"
        },
        color_continuous_scale="Viridis",
        size_max=40,
        template="plotly_dark"
    )

    # Reference line: lift = 1.0 (random baseline)
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="random baseline",
                  annotation_position="bottom right")

    # Annotate top 5 rules
    top5 = rules_pd.head(5)
    for _, row in top5.iterrows():
        label = f"{row['antecedent_names'][:20]} → {row['consequent_names'][:15]}"
        fig.add_annotation(
            x=row["confidence"], y=row["lift"],
            text=label, showarrow=True, arrowhead=2,
            font=dict(size=10, color="white"),
            bgcolor="rgba(0,0,0,0.6)"
        )

    # Save as HTML (viewable in browser)
    output_html = os.path.join(DATA_DIR, "association_rules_plot.html")
    fig.write_html(output_html)
    print(f"\n  Plot saved to: {output_html}")
    fig.show()

except ImportError:
    print("\n  [INFO] Install plotly for interactive visualization:")
    print("         pip install plotly")

t_phase5 = time.time() - t_phase5

# %% ============================================================
#  SUMMARY
# ============================================================
t_total = t_phase1 + t_phase23 + t_phase4 + t_phase5

print(f"\n{'=' * 65}")
print("  PIPELINE COMPLETE")
print(f"{'=' * 65}")
print(f"  Phase 1 (Ingest):           {t_phase1:>8.1f}s")
print(f"  Phase 2+3 (FP-Growth):      {t_phase23:>8.1f}s")
print(f"  Phase 4 (Rules):            {t_phase4:>8.1f}s")
print(f"  Phase 5 (Filter + Viz):     {t_phase5:>8.1f}s")
print(f"  {'─' * 35}")
print(f"  Total:                      {t_total:>8.1f}s")
print(f"\n  {N:,} baskets → {n_itemsets:,} itemsets → {n_rules:,} rules → {len(rules_pd)} filtered")
print(f"{'=' * 65}\n")

# %% ============================================================
#  COMPARISON: MATLAB vs SPARK
# ============================================================
print(f"{'=' * 65}")
print("  MATLAB vs SPARK COMPARISON")
print(f"{'=' * 65}")
print(f"  {'Aspect':<30} | {'MATLAB (manual)':<20} | {'Spark (production)':<20}")
print(f"  {'-' * 75}")
print(f"  {'Algorithm':<30} | {'Apriori':<20} | {'FP-Growth':<20}")
print(f"  {'Encoding':<30} | {'Manual sparse matrix':<20} | {'Automatic':<20}")
print(f"  {'Pair counting':<30} | {'Matrix multiply':<20} | {'FP-Tree traversal':<20}")
print(f"  {'Candidate generation':<30} | {'Yes (level-wise)':<20} | {'No (prefix paths)':<20}")
print(f"  {'Rule generation':<30} | {'Binary enumeration':<20} | {'Built-in':<20}")
print(f"  {'Lines of code':<30} | {'~800':<20} | {'~60':<20}")
print(f"  {'Total time':<30} | {'~108s':<20} | {f'~{t_total:.0f}s':<20}")
print(f"{'=' * 65}\n")

spark.stop()
