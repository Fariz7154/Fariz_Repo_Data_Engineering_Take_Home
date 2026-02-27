"""
Unit tests for src/kpis.py
Run with: pytest tests/
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyspark.sql import SparkSession
from src.kpis import (
    kpi1_top10_postings_by_category,
    kpi3_degree_vs_salary,
    kpi4_highest_salary_per_agency,
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder
            .master("local[*]")
            .appName("test-kpis")
            .getOrCreate()
    )
    yield session
    session.stop()


def test_kpi1_returns_top_categories(spark):
    data = [
        ("Engineering",), ("Engineering",), ("Legal",),
        ("Engineering",), ("Technology",), ("Legal",)
    ]
    df = spark.createDataFrame(data, ["Job Category"])
    result = kpi1_top10_postings_by_category(df).collect()
    assert result[0]["Job Category"] == "Engineering"
    assert result[0]["posting_count"] == 3


def test_kpi1_limit_to_10(spark):
    cats = [("Cat{}".format(i),) for i in range(20) for _ in range(i + 1)]
    df = spark.createDataFrame(cats, ["Job Category"])
    result = kpi1_top10_postings_by_category(df).collect()
    assert len(result) <= 10


def test_kpi3_degree_ordering(spark):
    data = [
        ("Bachelor", 60000.0),
        ("Master",   90000.0),
        ("PhD",      110000.0),
    ]
    df = spark.createDataFrame(data, ["degree_level", "salary_mid_annual"])
    result = kpi3_degree_vs_salary(df).collect()
    salaries = [row["avg_salary"] for row in result]
    assert salaries == sorted(salaries, reverse=True)


def test_kpi4_one_row_per_agency(spark):
    data = [
        ("AgencyA", "Job1", "Senior Analyst", 80000.0),
        ("AgencyA", "Job2", "Director",       120000.0),
        ("AgencyB", "Job3", "Engineer",        90000.0),
    ]
    df = spark.createDataFrame(
        data, ["Agency", "Job ID", "Business Title", "salary_mid_annual"]
    )
    result = kpi4_highest_salary_per_agency(df).collect()
    agencies = [row["Agency"] for row in result]
    assert len(agencies) == len(set(agencies))  # no duplicate agencies

    agency_a_row = [r for r in result if r["Agency"] == "AgencyA"][0]
    assert agency_a_row["salary_mid_annual"] == 120000.0

