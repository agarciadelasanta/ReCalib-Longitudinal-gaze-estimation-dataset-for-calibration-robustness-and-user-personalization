"""Hierarchical statistical comparison tools for ReCalib-style datasets."""

from .core import AnalysisConfig, AnalysisResult, analyze_frame
from .consistency import (
    ConsistencyConfig,
    ConsistencyResult,
    analyze_consistency_csv,
    analyze_consistency_frame,
)
from .reporting import AnalysisManifest, analyze_csv

__all__ = [
    "AnalysisConfig",
    "AnalysisManifest",
    "AnalysisResult",
    "ConsistencyConfig",
    "ConsistencyResult",
    "analyze_consistency_csv",
    "analyze_consistency_frame",
    "analyze_csv",
    "analyze_frame",
]
