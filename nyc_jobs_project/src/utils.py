"""
NYC Jobs Dataset — Utility Functions
Author: [Your Name]
"""

import os
from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "nyc-jobs-assessment") -> SparkSession:
    """
    Create or get an existing SparkSession.
    
    Configured for local development on Apple M1/M2 Macs
    and also works inside Docker Spark clusters.
    
    Args:
        app_name: Name for the Spark application
    
    Returns:
        SparkSession
    """
    master = os.environ.get("SPARK_MASTER", "local[*]")

    return (
        SparkSession.builder
            .appName(app_name)
            .master(master)
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
    )


def save_as_parquet(df, output_path: str, n_partitions: int = 1) -> None:
    """
    Save a Spark DataFrame as Parquet file(s).
    
    Args:
        df: Spark DataFrame to save
        output_path: Directory path for the Parquet output
        n_partitions: Number of output partitions (default 1 = single file)
    """
    df.coalesce(n_partitions).write.mode("overwrite").parquet(output_path)
    print(f"✅ Saved processed data to: {output_path}")


def get_null_counts(df) -> None:
    """
    Print null/empty count for every column in a DataFrame.
    
    Args:
        df: Spark DataFrame to profile
    """
    import pyspark.sql.functions as F
    print("=== NULL / EMPTY COUNTS ===")
    df.select([
        F.count(F.when(F.col(c).isNull() | (F.trim(F.col(c).cast("string")) == ""), 1))
          .alias(c)
        for c in df.columns
    ]).show(vertical=True, truncate=False)

