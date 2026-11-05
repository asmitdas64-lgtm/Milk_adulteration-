"""Evaluation module for computing metrics, plotting performance, and interpretability."""
from foodadult.evaluation.metrics import compute_metrics
from foodadult.evaluation.interpret import run_interpretability_analysis

__all__ = ["compute_metrics", "run_interpretability_analysis"]

