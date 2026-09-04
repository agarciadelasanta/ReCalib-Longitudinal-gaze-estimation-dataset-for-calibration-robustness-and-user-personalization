"""Hierarchical statistical comparison tools for ReCalib-style datasets."""

from .core import AnalysisConfig, AnalysisResult, analyze_frame
from .comparison import (
    DatasetComparisonManifest,
    DatasetComparisonResult,
    compare_csvs,
    compare_frames,
)
from .reporting import AnalysisManifest, analyze_csv

__all__ = [
    "AnalysisConfig",
    "AnalysisManifest",
    "AnalysisResult",
    "DatasetComparisonManifest",
    "DatasetComparisonResult",
    "analyze_csv",
    "analyze_frame",
    "compare_csvs",
    "compare_frames",
]
