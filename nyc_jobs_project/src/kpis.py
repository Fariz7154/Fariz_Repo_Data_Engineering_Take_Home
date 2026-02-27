"""
NYC Jobs Dataset — KPI Calculation Module
Author: [Your Name]
Description: Analytical KPI functions for NYC Jobs data.
"""

from pyspark.sql import DataFrame
from pyspark.sql.window import Window
import pyspark.sql.functions as F


def kpi1_top10_postings_by_category(df: DataFrame) -> DataFrame:
    """
    KPI 1: Number of job postings per category (Top 10).
    
    Args:
        df: Spark DataFrame with Job Category column
    
    Returns:
        DataFrame with columns [Job Category, posting_count]
        sorted descending, limited to 10 rows
    """
    return (
        df.groupBy("Job Category")
          .agg(F.count("*").alias("posting_count"))
          .orderBy(F.desc("posting_count"))
          .limit(10)
    )


def kpi2_salary_distribution_by_category(df: DataFrame, top_categories: list) -> DataFrame:
    """
    KPI 2: Salary distribution (avg, min, max) per job category.
    
    Args:
        df: Spark DataFrame with salary_mid_annual and Job Category
        top_categories: List of category names to include
    
    Returns:
        DataFrame with [Job Category, avg_salary, min_salary, max_salary, count]
    """
    return (
        df.filter(F.col("Job Category").isin(top_categories))
          .filter(F.col("salary_mid_annual").isNotNull())
          .groupBy("Job Category")
          .agg(
              F.avg("salary_mid_annual").alias("avg_salary"),
              F.min("salary_mid_annual").alias("min_salary"),
              F.max("salary_mid_annual").alias("max_salary"),
              F.count("*").alias("count")
          )
          .orderBy(F.desc("avg_salary"))
    )


def kpi3_degree_vs_salary(df: DataFrame) -> DataFrame:
    """
    KPI 3: Correlation between degree level and average salary.
    
    Args:
        df: Spark DataFrame with degree_level and salary_mid_annual
    
    Returns:
        DataFrame with [degree_level, avg_salary, count]
        sorted by avg_salary descending
    """
    return (
        df.filter(F.col("salary_mid_annual").isNotNull())
          .groupBy("degree_level")
          .agg(
              F.avg("salary_mid_annual").alias("avg_salary"),
              F.count("*").alias("count")
          )
          .orderBy(F.desc("avg_salary"))
    )


def kpi4_highest_salary_per_agency(df: DataFrame) -> DataFrame:
    """
    KPI 4: Job posting with the highest salary per agency.
    
    Uses window function to rank postings per agency by salary,
    then selects the top posting for each agency.
    
    Args:
        df: Spark DataFrame with Agency, Business Title, salary_mid_annual
    
    Returns:
        DataFrame with [Agency, Business Title, salary_mid_annual]
        one row per agency, sorted by salary descending
    """
    window_spec = Window.partitionBy("Agency").orderBy(F.desc("salary_mid_annual"))
    return (
        df.filter(F.col("salary_mid_annual").isNotNull())
          .withColumn("rank", F.row_number().over(window_spec))
          .filter(F.col("rank") == 1)
          .select("Agency", "Business Title", "salary_mid_annual")
          .orderBy(F.desc("salary_mid_annual"))
    )


def kpi5_avg_salary_per_agency_last2yrs(df: DataFrame) -> DataFrame:
    """
    KPI 5: Average salary per agency for job postings in the last 2 years.
    
    'Last 2 years' is defined relative to the maximum Posting Date in the dataset.
    
    Args:
        df: Spark DataFrame with Agency, Posting Date, salary_mid_annual
    
    Returns:
        DataFrame with [Agency, avg_salary, posting_count]
        sorted by avg_salary descending
    """
    max_date = df.agg(F.max("Posting Date")).collect()[0][0]
    if max_date is None:
        return df.limit(0)

    cutoff = F.date_sub(F.lit(max_date), 730)  # 2 years ≈ 730 days

    return (
        df.filter(
            F.col("Posting Date").isNotNull() &
            (F.col("Posting Date") >= cutoff) &
            F.col("salary_mid_annual").isNotNull()
        )
          .groupBy("Agency")
          .agg(
              F.avg("salary_mid_annual").alias("avg_salary"),
              F.count("*").alias("posting_count")
          )
          .orderBy(F.desc("avg_salary"))
    )


def kpi6_highest_paid_skills(df: DataFrame, min_count: int = 5) -> DataFrame:
    """
    KPI 6: Highest-paid skills in the NYC job market.
    
    Parses 'Preferred Skills' column by splitting on common delimiters,
    explodes into individual skills, then aggregates average salary per skill.
    
    Args:
        df: Spark DataFrame with Preferred Skills and salary_mid_annual
        min_count: Minimum number of job postings to include a skill (default 5)
    
    Returns:
        DataFrame with [skill, avg_salary, count]
        sorted by avg_salary descending, top 20
    """
    from pyspark.sql.functions import explode, split

    skills_df = (
        df.filter(F.col("salary_mid_annual").isNotNull())
          .withColumn(
              "skill",
              explode(
                  split(
                      F.regexp_replace(
                          F.lower(F.coalesce(F.col("Preferred Skills"), F.lit(""))),
                          r"[,;•·\n\r]",
                          ","
                      ),
                      ","
                  )
              )
          )
          .withColumn("skill", F.trim(F.col("skill")))
          .filter(
              (F.length(F.col("skill")) >= 4) &
              (F.length(F.col("skill")) <= 60)
          )
    )

    return (
        skills_df.groupBy("skill")
                 .agg(
                     F.avg("salary_mid_annual").alias("avg_salary"),
                     F.count("*").alias("count")
                 )
                 .filter(F.col("count") >= min_count)
                 .orderBy(F.desc("avg_salary"))
                 .limit(20)
    )

