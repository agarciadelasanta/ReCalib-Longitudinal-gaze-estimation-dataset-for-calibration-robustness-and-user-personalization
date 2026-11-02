"""Hierarchical statistical comparison tools for ReCalib-style datasets."""

from .core import AnalysisConfig, AnalysisResult, analyze_frame
from .comparison import (
    DatasetComparisonManifest,
    DatasetComparisonResult,
    compare_csvs,
    compare_frames,
    compare_label_sets,
    compare_label_sets_csv,
)
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
    "DatasetComparisonManifest",
    "DatasetComparisonResult",
    "analyze_consistency_csv",
    "analyze_consistency_frame",
    "analyze_csv",
    "analyze_frame",
    "compare_csvs",
    "compare_frames",
    "compare_label_sets",
    "compare_label_sets_csv",
]
