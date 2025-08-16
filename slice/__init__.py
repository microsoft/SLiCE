"""
SLiCE: Schema Lineage Composite Evaluation

A Python package for evaluating schema lineage extraction by comparing
model predictions with gold standards. Supports component-wise evaluation
and weighted averaging for final scores.
"""

# Suppress tree-sitter deprecation warnings before any imports
# This is from tree-sitter-languages library using deprecated API internally
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")
warnings.filterwarnings(
    "ignore", message="Language.*is deprecated", category=FutureWarning
)

from slice.SchemaLineageEvaluator import SchemaLineage, SchemaLineageEvaluator

__version__ = "0.1.0"
__author__ = "Jackie Jiaqi Yin"
__email__ = "jackie.yin@microsoft.com"

__all__ = [
    "SchemaLineageEvaluator",
    "SchemaLineage",
]
