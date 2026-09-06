"""
claim_denial.pipeline — Spark distributed pipeline wiring.

Exports
-------
SparkPipeline
    Main orchestration class that coordinates all pipeline phases through
    PySpark's distributed execution model (with local sequential fallback).

SparkPipelineConfig
    Dataclass holding cluster and Data Lake layout configuration.

DataLakeLayout
    Utility for resolving Data Lake partition paths.
"""

from claim_denial.pipeline.spark_pipeline import (
    DataLakeLayout,
    SparkPipeline,
    SparkPipelineConfig,
    SparkSessionManager,
    THROUGHPUT_TARGET_CLAIMS_PER_DAY,
)

__all__ = [
    "SparkPipeline",
    "SparkPipelineConfig",
    "DataLakeLayout",
    "SparkSessionManager",
    "THROUGHPUT_TARGET_CLAIMS_PER_DAY",
]
