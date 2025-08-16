"""
SLiCE Sample Data Module

Provides easy access to sample datasets for testing and demonstration.
"""

from .loaders import (
    get_lineages,
    get_pipeline_script,
    list_inventory,
    list_pipeline_scripts,
    prepare_batch_evaluation,
)

__all__ = [
    "list_inventory",
    "get_pipeline_script",
    "get_lineages",
    "prepare_batch_evaluation",
    "list_pipeline_scripts",
]
