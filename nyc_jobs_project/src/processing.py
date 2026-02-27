"""
NYC Jobs Dataset — Data Processing Module
Author: [Your Name]
Description: Functions for cleaning, normalizing, and processing NYC Jobs data.
"""

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, IntegerType, DateType


def clean_string_column(df: DataFrame, col_name: str) -> DataFrame:
    """
    Trim whitespace from a string column and replace empty strings with NULL.
    
    Args:
        df: Spark DataFrame
        col_name: Column to clean
    
    Returns:
        DataFrame with cleaned column
    """
    return df.withColumn(
        col_name,
        F.when(
            F.col(col_name).isNull() | (F.trim(F.col(col_name)) == ""),
            None
        ).otherwise(F.trim(F.col(col_name)))
    )


def cast_salary_columns(df: DataFrame) -> DataFrame:
    """
    Cast Salary Range From and Salary Range To from string to double.
    Removes commas and dollar signs before casting.
    
    Args:
        df: Spark DataFrame
    
    Returns:
        DataFrame with properly typed salary columns
    """
    return df \
        .withColumn(
            "Salary Range From",
            F.regexp_replace(F.col("Salary Range From"), "[$,]", "").cast(DoubleType())
        ) \
        .withColumn(
            "Salary Range To",
            F.regexp_replace(F.col("Salary Range To"), "[$,]", "").cast(DoubleType())
        )


def cast_date_columns(df: DataFrame) -> DataFrame:
    """
    Cast all date columns (Posting Date, Post Until, etc.) to DateType.
    Handles both 'yyyy-MM-dd' and 'yyyy-MM-ddTHH:mm:ss.SSS' formats.
    
    Args:
        df: Spark DataFrame
    
    Returns:
        DataFrame with properly typed date columns
    """
    date_cols = ["Posting Date", "Post Until", "Posting Updated", "Process Date"]
    for col_name in date_cols:
        if col_name in df.columns:
            df = df.withColumn(
                col_name,
                F.coalesce(
                    F.to_date(F.col(col_name), "yyyy-MM-dd"),
                    F.to_date(F.col(col_name), "yyyy-MM-dd'T'HH:mm:ss.SSS")
                )
            )
    return df


def cast_positions_column(df: DataFrame) -> DataFrame:
    """
    Cast '# Of Positions' column to integer, stripping non-numeric characters.
    
    Args:
        df: Spark DataFrame
    
    Returns:
        DataFrame with integer positions column
    """
    return df.withColumn(
        "num_positions",
        F.regexp_replace(F.col("# Of Positions"), "[^0-9]", "").cast(IntegerType())
    )


def normalize_job_category(df: DataFrame) -> DataFrame:
    """
    Replace null or empty Job Category with 'Unspecified'.
    
    Args:
        df: Spark DataFrame
    
    Returns:
        DataFrame with normalized job categories
    """
    return df.withColumn(
        "Job Category",
        F.when(
            F.col("Job Category").isNull() | (F.trim(F.col("Job Category")) == ""),
            F.lit("Unspecified")
        ).otherwise(F.trim(F.col("Job Category")))
    )


def remove_unused_columns(df: DataFrame) -> DataFrame:
    """
    Drop columns that have no predictive or analytical value based on profiling.
    These are primarily long free-text fields or highly sparse columns.
    
    Columns removed:
        - Job Description: long text, sparse signal
        - Minimum Qual Requirements: already extracted into degree_level
        - Preferred Skills: already extracted into skills features
        - Additional Information: mostly null, low signal
        - To Apply: directions text, not analytical
        - Recruitment Contact: mostly null
        - Residency Requirement: mostly null, administrative
        - Work Location 1: duplicates Work Location
        - Division/Work Unit: very high cardinality, redundant with Agency
    
    Args:
        df: Spark DataFrame
    
    Returns:
        DataFrame without low-value columns
    """
    cols_to_drop = [
        "Job Description",
        "Minimum Qual Requirements",
        "Preferred Skills",
        "Additional Information",
        "To Apply",
        "Recruitment Contact",
        "Residency Requirement",
        "Work Location 1",
        "Division/Work Unit"
    ]
    cols_existing = [c for c in cols_to_drop if c in df.columns]
    return df.drop(*cols_existing)


def sanitize_column_names(df: DataFrame) -> DataFrame:
    """
    Replace spaces, slashes, and special characters in column names
    so that the DataFrame can be written to Parquet without issues.
    
    Args:
        df: Spark DataFrame
    
    Returns:
        DataFrame with safe column names
    """
    def sanitize(name: str) -> str:
        import re
        name = name.replace("#", "num")
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        name = re.sub(r"_+", "_", name)
        return name.strip("_")

    for col_name in df.columns:
        new_name = sanitize(col_name)
        if new_name != col_name:
            df = df.withColumnRenamed(col_name, new_name)
    return df


def get_salary_frequency(df: DataFrame) -> list:
    """
    Return distinct values of Salary Frequency column.
    (Original sample function from assessment template)
    
    Args:
        df: Spark DataFrame
    
    Returns:
        List of distinct salary frequency values
    """
    row_list = df.select("Salary Frequency").distinct().collect()
    return [row["Salary Frequency"] for row in row_list]


