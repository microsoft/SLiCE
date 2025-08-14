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

# Inventory file column names provied by human
class InventoryColumns:
    GUID = "GUID"
    FOLDER_NAME = "folder_name"
    HAS_META = "has_meta"
    COLUMN_COUNT = "column_count"
    FOLDER_URL = "folder_url" 
    DIFFICULTY = "difficulty"
    STATUS = "status"
    LAST_MODIFIED_DATE = "last_modified_date"

# Lineage file column names provided by human
class LineageFileColumns:
    FINAL_COLUMN = "final_column"
    FINAL_TABLE = "final_table"
    HF_SOURCE_SCHEMA = "hf_source_schema"
    HF_SOURCE_TABLE = "hf_source_table"
    HF_TRANSFORMATION = "hf_transformation"
    HF_AGGREGATION_TYPE = "hf_aggregation_type"

# Output format keys for training data
class OutputKeys:
    PIPELINE_SCRIPT = "pipeline_script"
    FINAL_TABLE = "final_table"
    METADATA = "metadata"
    LINEAGE = "lineage"
    DIFFICULTY = "difficulty"
    EXAMPLES = "examples"

# File extension and naming
class FileExtensions:
    LINEAGE_SUFFIX = "_lineage.csv"
    TEXT_SUFFIX = ".txt"
    META_SUFFIX = "_meta.yaml"
    REASONING_SUFFIX = "_human_reasoning.yaml"
    REASONING_DISTILLATION_SUFFIX = "_reasoning_distillation.json"
    REASONING_DISTILLATION_UNPROCESSED_SUFFIX = "_reasoning_distillation_unprocessed.json"
    REASONING_DISTILLATION_NOT_MEET_THRESHOLD_SUFFIX = "_reasoning_distillation_not_meet_threshold.json"

class HumanReasoningKeys:
    QUERY = "query"
    REASONING = "reasoning"
    SOLUTION = "solution"

# Data handler summary keys
class SummaryDataKeys:
    ID = "id"
    TABLE_NAME = "table_name"
    COLUMN_COUNT = "column_count"
    DIFFICULTY = "difficulty"
    HAS_METADATA = "has_metadata"
    STATUS = "status"
    COMPLETE = "complete"

# Lineage output directory
class LineageOutputFile:
    PROCESSED = "processed_lineage.json"

# lineage llm response after parsing
class LineageParsedResponse:
    REASONING = "reasoning"
    ANSWERS = "answers"
    RAW_RESPONSE = "raw_response"
    IS_VALID = "is_valid"
    
# Human feedback special tokens
class HumanFeedbackSpecialTokens:
    CODE_END = "<CODEEND>"

class LLMResponseSpecialTokens:
    THINK_START = "<think>"
    THINK_END = "</think>"
    ANSWER_START = "<answer>"
    ANSWER_END = "</answer>"
    