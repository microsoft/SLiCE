"""
Data loading utilities for SLiCE sample datasets.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

GOLD = "human_labeled.json"
GENERATED = "generated.json"
INVENTORY_COLUMNS = [
    "id",
    "column_count",
    "difficulty",
    "last_modified_date",
    "language",
]


def list_inventory() -> pd.DataFrame:
    """
    List the metadata
    """
    file_path = Path(__file__).parent / "datasets" / "index.csv"

    result = pd.read_csv(file_path)
    return result


def get_pipeline_script(id: str = "80963c21-378a-4d8d-e948-feeb9d74c049") -> str:
    """Get the pipeline script for a specific dataset ID from gold data.

    Note: Pipeline scripts are only available in gold (human_labeled) data.
    """
    file_path = Path(__file__).parent / "datasets" / GOLD

    with open(file_path, "r") as f:
        data = json.load(f)

    if id in data:
        return str(data[id].get("pipeline_script", ""))

    raise ValueError(f"Dataset ID '{id}' not found in gold data")


def get_lineages(
    id: str = "80963c21-378a-4d8d-e948-feeb9d74c049",
    label_type: str = "gold",
    columns: Optional[str] = None,
) -> Dict[str, Any]:
    """Get lineage information filtered by ID and/or label type.

    Args:
        id: Specific dataset ID to filter by
        label_type: Filter by 'gold' or 'generated'
        columns: Specific column name to get lineage for (optional)

    Returns:
        Dictionary containing lineage information
    """
    lineages = {}

    if label_type == "gold":
        file_path = Path(__file__).parent / "datasets" / GOLD
        lineage_key = "lineage"
    elif label_type == "generated":
        file_path = Path(__file__).parent / "datasets" / GENERATED
        lineage_key = "parsed_lineages"
    else:
        raise ValueError(
            f"Invalid label_type '{label_type}'. Must be 'gold' or 'generated'"
        )

    with open(file_path, "r") as f:
        data = json.load(f)

    if id in data and lineage_key in data[id]:
        lineage_data = data[id][lineage_key]

        if columns:
            if columns in lineage_data:
                lineages[columns] = lineage_data[columns]
            else:
                raise ValueError(
                    f"Column '{columns}' not found in lineage for dataset '{id}'"
                )
        else:
            lineages = lineage_data
    else:
        raise ValueError(f"Dataset ID '{id}' not found or has no lineage data")

    return lineages


def prepare_batch_evaluation(gold: Dict, generated: Dict) -> Tuple:
    """Prepare data for batch evaluation maintaining proper order.

    Args:
        gold: Dictionary of gold lineages from get_lineages
        generated: Dictionary of generated lineages from get_lineages

    Returns:
        Tuple of (gold_ordered, generated_ordered) with matching schema names

    Raises:
        ValueError: If schema names don't match between gold and generated
    """
    gold_schema_names = set(gold.keys())
    generated_schema_names = set(generated.keys())

    if gold_schema_names != generated_schema_names:
        missing_in_generated = gold_schema_names - generated_schema_names
        extra_in_generated = generated_schema_names - gold_schema_names

        error_msg = "Schema names mismatch between gold and generated data."
        if missing_in_generated:
            error_msg += f" Missing in generated: {missing_in_generated}."
        if extra_in_generated:
            error_msg += f" Extra in generated: {extra_in_generated}."

        raise ValueError(error_msg)

    gold_ordered = []
    generated_ordered = []

    for schema_name in sorted(gold_schema_names):
        gold_ordered.append((schema_name, gold[schema_name]))
        generated_ordered.append((schema_name, generated[schema_name]))

    return (gold_ordered, generated_ordered)


def list_pipeline_scripts() -> List[Dict[str, Any]]:
    """List all available pipeline scripts with metadata.

    Note: Pipeline scripts are only available in gold (human_labeled) data.
    """
    scripts = []

    # Get inventory for difficulty information
    inventory_df = list_inventory()
    difficulty_map = dict(zip(inventory_df["id"], inventory_df["difficulty"]))

    file_path = Path(__file__).parent / "datasets" / GOLD
    with open(file_path, "r") as f:
        data = json.load(f)

    for dataset_id, dataset_info in data.items():
        if isinstance(dataset_info, dict):
            scripts.append(
                {
                    "id": dataset_id,
                    "label_type": "gold",
                    "final_table": dataset_info["final_table"],
                    "difficulty": difficulty_map[dataset_id],
                }
            )

    return scripts
