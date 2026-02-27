"""
Unit tests for src/feature_engineering.py
Run with: pytest tests/
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyspark.sql import SparkSession
from src.feature_engineering import (
    add_annual_salary_midpoint,
    add_degree_level,
    add_salary_band,
    add_employment_type_flag,
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder
            .master("local[*]")
            .appName("test-feature-engineering")
            .getOrCreate()
    )
    yield session
    session.stop()


# ── add_annual_salary_midpoint ───────────────────────────────────

def test_annual_salary_annual_freq(spark):
    df = spark.createDataFrame(
        [(50000.0, 70000.0, "Annual")],
        ["Salary Range From", "Salary Range To", "Salary Frequency"]
    )
    result = add_annual_salary_midpoint(df).first()
    assert result["salary_mid_annual"] == 60000.0


def test_annual_salary_hourly_freq(spark):
    df = spark.createDataFrame(
        [(20.0, 30.0, "Hourly")],
        ["Salary Range From", "Salary Range To", "Salary Frequency"]
    )
    result = add_annual_salary_midpoint(df).first()
    # midpoint = 25, annualized = 25 * 2080 = 52000
    assert result["salary_mid_annual"] == 52000.0


def test_annual_salary_daily_freq(spark):
    df = spark.createDataFrame(
        [(100.0, 200.0, "Daily")],
        ["Salary Range From", "Salary Range To", "Salary Frequency"]
    )
    result = add_annual_salary_midpoint(df).first()
    # midpoint = 150, annualized = 150 * 260 = 39000
    assert result["salary_mid_annual"] == 39000.0


# ── add_degree_level ─────────────────────────────────────────────

def test_degree_level_phd(spark):
    df = spark.createDataFrame(
        [("Applicant must have a Ph.D. or equivalent.",)],
        ["Minimum Qual Requirements"]
    )
    result = add_degree_level(df).first()["degree_level"]
    assert result == "PhD"


def test_degree_level_master(spark):
    df = spark.createDataFrame(
        [("Master's degree required in Computer Science.",)],
        ["Minimum Qual Requirements"]
    )
    result = add_degree_level(df).first()["degree_level"]
    assert result == "Master"


def test_degree_level_bachelor(spark):
    df = spark.createDataFrame(
        [("A baccalaureate degree from an accredited college.",)],
        ["Minimum Qual Requirements"]
    )
    result = add_degree_level(df).first()["degree_level"]
    assert result == "Bachelor"


def test_degree_level_high_school(spark):
    df = spark.createDataFrame(
        [("High school diploma or GED required.",)],
        ["Minimum Qual Requirements"]
    )
    result = add_degree_level(df).first()["degree_level"]
    assert result == "High School"


def test_degree_level_unspecified(spark):
    df = spark.createDataFrame([("",)], ["Minimum Qual Requirements"])
    result = add_degree_level(df).first()["degree_level"]
    assert result == "Unspecified"


# ── add_salary_band ──────────────────────────────────────────────

def test_salary_band_entry(spark):
    df = spark.createDataFrame([(30000.0,)], ["salary_mid_annual"])
    assert add_salary_band(df).first()["salary_band"] == "Entry Level"


def test_salary_band_mid(spark):
    df = spark.createDataFrame([(55000.0,)], ["salary_mid_annual"])
    assert add_salary_band(df).first()["salary_band"] == "Mid Level"


def test_salary_band_executive(spark):
    df = spark.createDataFrame([(200000.0,)], ["salary_mid_annual"])
    assert add_salary_band(df).first()["salary_band"] == "Executive Level"


# ── add_employment_type_flag ─────────────────────────────────────

def test_employment_flag_full_time(spark):
    df = spark.createDataFrame(
        [("F",)], ["Full-Time/Part-Time indicator"]
    )
    assert add_employment_type_flag(df).first()["is_full_time"] == 1


def test_employment_flag_part_time(spark):
    df = spark.createDataFrame(
        [("P",)], ["Full-Time/Part-Time indicator"]
    )
    assert add_employment_type_flag(df).first()["is_full_time"] == 0

