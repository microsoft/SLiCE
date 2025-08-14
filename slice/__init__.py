"""
SLiCE: Schema Lineage Calculation and Evaluation

A Python package for evaluating schema lineage extraction by comparing 
model predictions with gold standards. Supports component-wise evaluation 
and weighted averaging for final scores.
"""

from slice.SchemaLineageEvaluator import SchemaLineageEvaluator, SchemaLineage

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "SchemaLineageEvaluator",
    "SchemaLineage",
]