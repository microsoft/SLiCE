"""
SLiCE Sample Data Module

Provides easy access to sample datasets for testing and demonstration.
"""

from .loaders import (
    list_inventory,
    get_pipeline_script,
    get_lineages,
    prepare_batch_evaluation,
    list_pipeline_scripts
)

__all__ = [
    'list_inventory',
    'get_pipeline_script', 
    'get_lineages',
    'prepare_batch_evaluation',
    'list_pipeline_scripts'
]