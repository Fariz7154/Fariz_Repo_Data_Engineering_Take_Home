"""
NYC Jobs Dataset — Feature Engineering Module
Author: [Your Name]
Description: Creates analytical features from raw NYC Jobs data.

Techniques Applied:
  FE1 - Salary normalization to annual midpoint (domain transformation)
  FE2 - Degree level extraction via regex from qualification text (NLP/text mining)
  FE3 - Time-based features from posting date (temporal feature engineering)
  FE4 - Salary band categorization (binning/discretization)
  FE5 - Employment type encoding (categorical encoding)
"""

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, IntegerType


# ─────────────────────────────────────────────────────────────────
# FE1: Annual Salary Midpoint (Domain Transformation)
# ─────────────────────────────────────────────────────────────────

def add_annual_salary_midpoint(df: DataFrame) -> DataFrame:
    """
    FE1 — Domain Transformation:
    Compute the midpoint of salary range and normalize to annual equivalent.
    
    Logic:
        - Hourly × 2080 = Annual (52 weeks × 40 hrs)
        - Daily  × 260  = Annual (52 weeks × 5 days)
        - Annual stays as-is
    
    New columns:
        salary_annual_from, salary_annual_to, salary_mid_annual
    
    Args:
        df: Spark DataFrame with Salary Range From/To and Salary Frequency
    
    Returns:
        DataFrame with new salary feature columns
    """
    freq = F.lower(F.coalesce(F.col("Salary Frequency"), F.lit("")))

    df = df.withColumn(
        "salary_annual_from",
        F.when(freq.contains("hour"), F.col("Salary Range From") * 2080)
         .when(freq.contains("day"), F.col("Salary Range From") * 260)
         .otherwise(F.col("Salary Range From"))
    )

    df = df.withColumn(
        "salary_annual_to",
        F.when(freq.contains("hour"), F.col("Salary Range To") * 2080)
         .when(freq.contains("day"), F.col("Salary Range To") * 260)
         .otherwise(F.col("Salary Range To"))
    )

    df = df.withColumn(
        "salary_mid_annual",
        (F.col("salary_annual_from") + F.col("salary_annual_to")) / 2
    )

    return df


# ─────────────────────────────────────────────────────────────────
# FE2: Degree Level (Text Mining / NLP)
# ─────────────────────────────────────────────────────────────────

def add_degree_level(df: DataFrame) -> DataFrame:
    """
    FE2 — Text Mining:
    Extract required education level from 'Minimum Qual Requirements' text.
    
    Hierarchy: PhD > Master > Bachelor > Associate > High School > Unspecified
    
    New column: degree_level (string, categorical)
    
    Args:
        df: Spark DataFrame with Minimum Qual Requirements column
    
    Returns:
        DataFrame with degree_level feature
    """
    req_text = F.lower(
        F.coalesce(F.col("Minimum Qual Requirements"), F.lit(""))
    )

    return df.withColumn(
        "degree_level",
        F.when(req_text.rlike(r"ph\.?d|doctorate|doctoral"), F.lit("PhD"))
         .when(req_text.rlike(r"master[\\'']?s|m\.b\.a|m\.s\.|graduate degree"), F.lit("Master"))
         .when(req_text.rlike(r"baccalaureate|bachelor|four.?year college|college degree"), F.lit("Bachelor"))
         .when(req_text.rlike(r"associate|60 semester credit"), F.lit("Associate"))
         .when(req_text.rlike(r"high school|h\.?s\.|ged|equivalent"), F.lit("High School"))
         .otherwise(F.lit("Unspecified"))
    )


# ─────────────────────────────────────────────────────────────────
# FE3: Temporal Features (Date-Based Engineering)
# ─────────────────────────────────────────────────────────────────

def add_temporal_features(df: DataFrame) -> DataFrame:
    """
    FE3 — Temporal Feature Engineering:
    Extract time-based features from Posting Date.
    
    New columns:
        - posting_year     : Year of job posting
        - posting_month    : Month of job posting (1-12)
        - posting_quarter  : Quarter (Q1-Q4)
        - days_since_posted: Days from posting to latest date in dataset
    
    Args:
        df: Spark DataFrame with Posting Date column (DateType)
    
    Returns:
        DataFrame with temporal features added
    """
    # Compute max date for recency calculation
    max_date = df.agg(F.max("Posting Date")).collect()[0][0]

    df = df.withColumn("posting_year", F.year(F.col("Posting Date")))
    df = df.withColumn("posting_month", F.month(F.col("Posting Date")))
    df = df.withColumn(
        "posting_quarter",
        F.concat(
            F.lit("Q"),
            F.quarter(F.col("Posting Date")).cast("string")
        )
    )

    if max_date is not None:
        df = df.withColumn(
            "days_since_posted",
            F.datediff(F.lit(max_date), F.col("Posting Date"))
        )
    else:
        df = df.withColumn("days_since_posted", F.lit(None).cast(IntegerType()))

    return df


# ─────────────────────────────────────────────────────────────────
# FE4: Salary Band (Binning / Discretization)
# ─────────────────────────────────────────────────────────────────

def add_salary_band(df: DataFrame) -> DataFrame:
    """
    FE4 — Binning / Discretization:
    Categorize annual salary midpoint into labeled salary bands.
    
    Bands:
        < $40K         → Entry Level
        $40K – $70K    → Mid Level
        $70K – $100K   → Senior Level
        $100K – $150K  → Director Level
        > $150K        → Executive Level
        NULL           → Unknown
    
    New column: salary_band (string, ordinal categorical)
    
    Args:
        df: Spark DataFrame with salary_mid_annual column
    
    Returns:
        DataFrame with salary_band feature
    """
    mid = F.col("salary_mid_annual")

    return df.withColumn(
        "salary_band",
        F.when(mid.isNull(), F.lit("Unknown"))
         .when(mid < 40000, F.lit("Entry Level"))
         .when((mid >= 40000) & (mid < 70000), F.lit("Mid Level"))
         .when((mid >= 70000) & (mid < 100000), F.lit("Senior Level"))
         .when((mid >= 100000) & (mid < 150000), F.lit("Director Level"))
         .otherwise(F.lit("Executive Level"))
    )


# ─────────────────────────────────────────────────────────────────
# FE5: Employment Type Encoding (Categorical Encoding)
# ─────────────────────────────────────────────────────────────────

def add_employment_type_flag(df: DataFrame) -> DataFrame:
    """
    FE5 — Categorical Encoding:
    Create binary flag for Full-Time vs Part-Time positions.
    
    New column: is_full_time (integer, 1 = Full-Time, 0 = Part-Time/Unknown)
    
    Args:
        df: Spark DataFrame with Full-Time/Part-Time indicator column
    
    Returns:
        DataFrame with is_full_time flag
    """
    col_name = "Full-Time/Part-Time indicator"
    if col_name not in df.columns:
        return df.withColumn("is_full_time", F.lit(None).cast(IntegerType()))

    return df.withColumn(
        "is_full_time",
        F.when(F.upper(F.col(col_name)) == "F", F.lit(1))
         .when(F.upper(F.col(col_name)) == "P", F.lit(0))
         .otherwise(F.lit(None).cast(IntegerType()))
    )


def apply_all_features(df: DataFrame) -> DataFrame:
    """
    Master function: apply all feature engineering steps in sequence.
    
    Args:
        df: Raw or cleaned Spark DataFrame
    
    Returns:
        DataFrame with all engineered features
    """
    df = add_annual_salary_midpoint(df)
    df = add_degree_level(df)
    df = add_temporal_features(df)
    df = add_salary_band(df)
    df = add_employment_type_flag(df)
    return df

