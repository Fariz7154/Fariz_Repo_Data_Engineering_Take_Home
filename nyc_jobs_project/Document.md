# MyDocument — Assumptions, Learnings & Challenges
## Assumptions
1. **Salary Normalization**: Hourly rates multiplied by 2080 (52 wks × 40 hrs).
   Daily rates multiplied by 260 (52 wks × 5 days). Annual rates used as-is.
2. **"Last 2 years"**: Computed relative to the maximum Posting Date in the dataset
   (not today), so the analysis is reproducible on historical data.
3. **Job Category nulls**: Replaced with "Unspecified" to preserve row count.
4. **Degree extraction**: Best-effort regex on unstructured text; some postings
   with unusual phrasing will fall into "Unspecified".
5. **Skills extraction**: Parsed from "Preferred Skills" column only (not Job Description)
   to avoid very noisy text mining on long free-text.
## Challenges
1. **Date column inconsistency**: Some date values contain full timestamps
   (`2019-01-01T00:00:00.000`), others plain dates (`2019-01-01`).
   Solved with `F.coalesce` of multiple format parsers.
2. **Parquet column naming**: Spark Parquet writer rejects column names containing
   spaces, `/`, `#`. Sanitization function `sanitize_column_names()` addresses this.
3. **M1 Mac compatibility**: PySpark 3.4+ runs natively on Apple Silicon; only
   requirement is Java 11 from Homebrew (`openjdk@11`).
4. **Sparse columns**: Many columns (Recruitment Contact, Residency Requirement,
   Hours/Shift) have >30% null rate and were dropped after profiling.
## Learnings
- PySpark Window functions are essential for per-group ranking (KPI 4).
- Regex-based NLP on Minimum Qual Requirements provides a lightweight alternative
  to full NLP models when structured data is unavailable.
- Salary normalization is critical before any cross-frequency comparison.
- Feature engineering decisions should be driven by domain knowledge (NYC government
  pay scales, union contracts, etc.).
## Deployment Approach
- **Development**: Run Jupyter locally with `local[*]` Spark master.
- **Staging/Production**: Docker Compose with Bitnami Spark cluster
  (1 master + 2 workers, matching original assignment setup).
- **Scheduling**: Use Apache Airflow DAG to trigger the processing pipeline
  on a daily schedule; or AWS Glue for fully managed Spark execution.
## Triggering the Pipeline
```bash
# Option 1 — Run notebook headlessly
jupyter nbconvert --to notebook --execute notebooks/assessment_notebook.ipynb
# Option 2 — Run Python script directly
python -c "
from src.processing import *
from src.feature_engineering import *
from src.kpis import *
from src.utils import *
# ... pipeline code
"
# Option 3 — Apache Airflow DAG
# (See airflow_dag_example.py for reference)
