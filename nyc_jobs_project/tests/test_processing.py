"""
Unit tests for src/processing.py
Run with: pytest tests/
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyspark.sql import SparkSession
from src.processing import (
    clean_string_column,
    cast_salary_columns,
    cast_date_columns,
    cast_positions_column,
    normalize_job_category,
    get_salary_frequency
)


@pytest.fixture(scope="module")
def spark():
    """Create a shared SparkSession for all tests."""
    session = (
        SparkSession.builder
            .master("local[*]")
            .appName("test-processing")
            .getOrCreate()
    )
    yield session
    session.stop()


# ── clean_string_column ──────────────────────────────────────────

def test_clean_string_column_trims_whitespace(spark):
    df = spark.createDataFrame([("  hello  ",)], ["col"])
    result = clean_string_column(df, "col").first()["col"]
    assert result == "hello"


def test_clean_string_column_empty_to_null(spark):
    df = spark.createDataFrame([("",)], ["col"])
    result = clean_string_column(df, "col").first()["col"]
    assert result is None


def test_clean_string_column_null_stays_null(spark):
    df = spark.createDataFrame([(None,)], ["col"])
    result = clean_string_column(df, "col").first()["col"]
    assert result is None


# ── cast_salary_columns ──────────────────────────────────────────

def test_cast_salary_columns_basic(spark):
    df = spark.createDataFrame([("50000", "80000")], ["Salary Range From", "Salary Range To"])
    casted = cast_salary_columns(df).first()
    assert casted["Salary Range From"] == 50000.0
    assert casted["Salary Range To"] == 80000.0


def test_cast_salary_columns_removes_commas(spark):
    df = spark.createDataFrame([("50,000", "80,000")], ["Salary Range From", "Salary Range To"])
    casted = cast_salary_columns(df).first()
    assert casted["Salary Range From"] == 50000.0
    assert casted["Salary Range To"] == 80000.0


# ── normalize_job_category ───────────────────────────────────────

def test_normalize_job_category_null(spark):
    df = spark.createDataFrame([(None,)], ["Job Category"])
    result = normalize_job_category(df).first()["Job Category"]
    assert result == "Unspecified"


def test_normalize_job_category_empty(spark):
    df = spark.createDataFrame([("",)], ["Job Category"])
    result = normalize_job_category(df).first()["Job Category"]
    assert result == "Unspecified"


def test_normalize_job_category_valid(spark):
    df = spark.createDataFrame([("Technology",)], ["Job Category"])
    result = normalize_job_category(df).first()["Job Category"]
    assert result == "Technology"


# ── cast_positions_column ────────────────────────────────────────

def test_cast_positions_column(spark):
    df = spark.createDataFrame([("5",)], ["# Of Positions"])
    result = cast_positions_column(df).first()["num_positions"]
    assert result == 5


# ── get_salary_frequency ─────────────────────────────────────────

def test_get_salary_frequency(spark):
    data = [("1", "Annual"), ("2", "Hourly")]
    df = spark.createDataFrame(data, ["id", "Salary Frequency"])
    result = get_salary_frequency(df)
    assert set(result) == {"Annual", "Hourly"}

