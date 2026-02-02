"""
LIME Stability Analysis Package
================================
Team: ModelMiners (Abdul Aahad Qureshi, Khyzar Baig)
Course: iML Winter 2025/26
"""

__version__ = "1.0.0"
__author__ = "ModelMiners"

from .data_loader import DataLoader
from .models import SimpleLogisticModel, DistilBERTModel
from .explainers import LIMEStabilityAnalyzer
from .metrics import StabilityMetrics

__all__ = [
    'DataLoader',
    'SimpleLogisticModel',
    'DistilBERTModel',
    'LIMEStabilityAnalyzer',
    'StabilityMetrics',
]
