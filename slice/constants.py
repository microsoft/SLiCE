"""
Constants for the project

This file centralizes all constant values used throughout the codebase to avoid typos and
make it easier to update values in a single place.
"""

# lineage components name
class LineageComponents:
    SOURCE_SCHEMA = "source_schema"
    SOURCE_TABLE = "source_table"
    TRANSFORMATION = "transformation"
    AGGREGATION = "aggregation"
    METADATA = "metadata"

# Score components name
class ScoreComponents:
    SOURCE_SCHEMA = "source_schema"
    SOURCE_TABLE = "source_table"
    TRANSFORMATION = "transformation"
    AGGREGATION = "aggregation"
    METADATA = "metadata"
    ALL = "overall"

# Evaluation constants
class EvaluationDefaults:
    DEFAULT_WEIGHTS_WITHOUT_METADATA = {
        "source_table": 0.4,
        "transformation": 0.4, 
        "aggregation": 0.2
    }
    DEFAULT_WEIGHTS_WITH_METADATA = {
        "source_table": 0.3,
        "transformation": 0.3,
        "aggregation": 0.2,
        "metadata": 0.2
    }
    DEFAULT_SOURCE_TABLE_WEIGHTS = {"f1": 0.7, "fuzzy": 0.3}
    DEFAULT_TEXT_SIMILARITY_WEIGHTS = {"bleu": 0.5, "weighted_bleu": 0.3, "ast": 0.2}
    
    # BLEU score constants
    DEFAULT_BLEU_WEIGHTS = (0.4, 0.3, 0.2, 0.1)
    WEIGHTED_BLEU_WEIGHTS = (0.4, 0.3, 0.2, 0.1)
    WEIGHTED_BLEU_DEFAULT_WEIGHT = 0.2
    