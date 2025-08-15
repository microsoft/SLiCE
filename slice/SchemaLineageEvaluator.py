import json
import logging
from typing import Dict, Optional, List, Set
from fuzzywuzzy import fuzz
from dataclasses import dataclass
from slice.eval import (calc_fuzzy_match_score, 
                            calc_f1_score, 
                            calc_bleu_score,
                            WeightedBleuCalculator,
                            ASTCalculator,
                            clean_code_formatting)
from slice.string_parsers import parse_entity_names
from slice.constants import LineageComponents, ScoreComponents, EvaluationDefaults
import numpy as np
from slice.SYNTAX import PYTHON_SYNTAX, SQL_SYNTAX, KEYWORDS, PYTHON_OPERATORS, SQL_OPERATORS, CSHARP_SYNTAX, CSHARP_OPERATORS


@dataclass
class SchemaLineage:
    source_schema: str
    source_table: str
    transformation: str
    aggregation: str
    metadata: Optional[str] = None

class SchemaLineageEvaluator:
    """
    Evaluates schema lineage extraction by comparing model predictions with gold standards.
    Supports component-wise evaluation and weighted averaging for final scores.
    """
    def __init__(self, 
                 weights: Dict[str, float]=None,
                 has_metadata: bool=False,
                 source_table_weights: Dict[str, float]=None,
                 transformation_weights: Dict[str, float]=None,
                 aggregation_weights: Dict[str, float]=None,
                 metadata_weights: Dict[str, float]=None,
                 keywords=KEYWORDS,
                 sql_syntax=SQL_SYNTAX,
                 python_syntax=PYTHON_SYNTAX,
                 sql_operators=SQL_OPERATORS,
                 python_operators=PYTHON_OPERATORS,
                 csharp_syntax=CSHARP_SYNTAX,
                 csharp_operators=CSHARP_OPERATORS):

        """
        weights : dict, optional
        Dictionary containing weights for each component:
        {
            LineageComponents.SOURCE_TABLE: weight for source tables,
            LineageComponents.TRANSFORMATION: weight for transformation,
            LineageComponents.AGGREGATION: weight for aggregation,
            LineageComponents.METADATA: weight for metadata (if applicable)
        }
        If not provided, equal weights will be assigned.
    
        has_metadata : bool, default=False
            Whether the data pipeline includes metadata.
            If True, 'e' component will be included in the evaluation.
            
        source_tables_weights : dict, default={'f1': 0.7, 'fuzzy': 0.3}
            Weights for F1 score and fuzzy matching in source tables evaluation.
            Must sum to 1.0.
            
        transformation_weights : dict, default={'bleu': 0.5, 'weighted_bleu': 0.3, 'ast': 0.2}
            Weights for BLEU, weighted BLEU, and AST similarity in transformation evaluation.
            Must sum to 1.0.
            
        aggregation_weights : dict, default={'bleu': 0.5, 'weighted_bleu': 0.3, 'ast': 0.2}
            Weights for BLEU, weighted BLEU, and AST similarity in aggregation evaluation.
            Must sum to 1.0.
            
        metadata_weights : dict, default={'bleu': 0.5, 'weighted_bleu': 0.3, 'ast': 0.2}
            Weights for BLEU, weighted BLEU, and AST similarity in metadata evaluation.
            Must sum to 1.0.

        keywords : set, optional
            Set of keywords to use for evaluation.
            If not provided, default keywords will be used.

        operators : set, optional
            Set of operators to use for evaluation.
            If not provided, default operators will be used.

        python_operators : set, optional
            Set of Python operators to use for evaluation.
            If not provided, default operators will be used.

        sql_syntax : set, optional
            Set of SQL syntax to use for evaluation in AST calculation.
            If not provided, default syntax will be used.

        python_syntax : set, optional
            Set of Python syntax to use for evaluation in AST calculation.
            If not provided, default syntax will be used.

        csharp_syntax : set, optional
            Set of C# syntax to use for evaluation in AST calculation.
            If not provided, default syntax will be used.

        csharp_operators : set, optional
            Set of C# operators to use for evaluation.
            If not provided, default operators will be used.
        """
        self.has_metadata = has_metadata
        self.weights = weights or self._get_default_weights()

        # assign hyperparameters with defaults from constants
        self.source_table_weights = source_table_weights or EvaluationDefaults.DEFAULT_SOURCE_TABLE_WEIGHTS
        self.transformation_weights = transformation_weights or EvaluationDefaults.DEFAULT_TEXT_SIMILARITY_WEIGHTS
        self.aggregation_weights = aggregation_weights or EvaluationDefaults.DEFAULT_TEXT_SIMILARITY_WEIGHTS
        self.metadata_weights = metadata_weights or EvaluationDefaults.DEFAULT_TEXT_SIMILARITY_WEIGHTS

        # validate parameters
        self._validate_parameters()

        self.syntax = {
            'sql': sql_syntax,
            'python': python_syntax,
            'csharp': csharp_syntax,
        }
        self.operators = {
            'sql': sql_operators,
            'python': python_operators,
            'csharp': csharp_operators,
        }
        self.keywords = keywords

        self.weighted_bleu_calculator = WeightedBleuCalculator(
            weights=(0.4, 0.3, 0.2, 0.1),
            default_weight=0.2,
            keywords=self.keywords,
            special_operators=self.operators['sql'] | self.operators['python'] | self.operators['csharp']
        )

        self.ast_calculator = ASTCalculator(
            python_syntax=self.syntax['python'] | self.operators['python'],
            sql_syntax=self.syntax['sql'] | self.operators['sql'],
            csharp_syntax=self.syntax['csharp'] | self.operators['csharp'],
        )

    def _validate_lineage(self, lineage: Dict[str, str]) -> bool:
        """
        Validate the lineage.
        """
        if not isinstance(lineage, dict):
            logging.warning(f"Lineage is not a dictionary: {lineage}")
            return False
        
        if self.has_metadata:
            required_attrs = [LineageComponents.SOURCE_SCHEMA, LineageComponents.SOURCE_TABLE, 
                             LineageComponents.TRANSFORMATION, LineageComponents.AGGREGATION, 
                             LineageComponents.METADATA]
        else: 
            required_attrs = [LineageComponents.SOURCE_SCHEMA, LineageComponents.SOURCE_TABLE, 
                             LineageComponents.TRANSFORMATION, LineageComponents.AGGREGATION]
        
        for attr in required_attrs:
            if attr not in lineage:
                logging.warning(f"Missing attribute: {attr}")
                return False
            if not isinstance(lineage[attr], str):
                logging.warning(f"Attribute {attr} is not a string or is empty: {lineage[attr]}")
                return False
        return True
    
    
    def _turn_dict_to_lineage(self, lineage: Dict[str, str]) -> SchemaLineage:
        """
        Validate the lineage.
        """
        if self._validate_lineage(lineage):
            return SchemaLineage(**lineage)
        else:
            raise ValueError(f"Invalid lineage: {lineage}")
        

    def _validate_parameters(self):
        """
        Validate the parameters.
        """
        if self.has_metadata:
            if len(self.weights) != 4:
                raise ValueError("Weights must have 4 components: source_tables, transformation, aggregation, metadata")
        else:
            if len(self.weights) != 3:
                raise ValueError("Weights must have 3 components: source_tables, transformation, aggregation")
        
        # Use tolerance for floating-point comparison
        tolerance = 1e-9
        
        sum_weights = sum(self.weights.values())
        if abs(sum_weights - 1.0) > tolerance:
            raise ValueError(f"Weights must sum to 1.0, but got {sum_weights}")
        
        sum_source_table_weights = sum(self.source_table_weights.values())
        if abs(sum_source_table_weights - 1.0) > tolerance:
            raise ValueError(f"Source table weights must sum to 1.0, but got {sum_source_table_weights}")
        
        sum_transformation_weights = sum(self.transformation_weights.values())
        if abs(sum_transformation_weights - 1.0) > tolerance:
            raise ValueError(f"Transformation weights must sum to 1.0, but got {sum_transformation_weights}")

        sum_aggregation_weights = sum(self.aggregation_weights.values())
        if abs(sum_aggregation_weights - 1.0) > tolerance:
            raise ValueError(f"Aggregation weights must sum to 1.0, but got {sum_aggregation_weights}")

        if self.has_metadata:
            sum_metadata_weights = sum(self.metadata_weights.values())  
            if abs(sum_metadata_weights - 1.0) > tolerance:
                raise ValueError(f"Metadata weights must sum to 1.0, but got {sum_metadata_weights}")   
            

    def _get_default_weights(self) -> Dict[str, float]:
        if self.has_metadata:
            return EvaluationDefaults.DEFAULT_WEIGHTS_WITH_METADATA.copy()
        else:
            return EvaluationDefaults.DEFAULT_WEIGHTS_WITHOUT_METADATA.copy()

    def evaluate(self, predicted_lineage: Dict[str, str],
                  gold_lineage: Dict[str, str]) -> Dict[str, float]:
        """
        Evaluate the predicted schema lineage against the gold standard.
        Returns
        -------
        dict
            Detailed evaluation scores with overall score and component scores.
            {
                'overall': float,  # Overall weighted score
                'format': float,   # Format correctness (0 or 1)
                LineageComponents.SOURCE_TABLE: float,  # Source tables score (0 to 1)
                LineageComponents.TRANSFORMATION: float,  # Transformation score (0 to 1)
                LineageComponents.AGGREGATION: float,  # Aggregation score (0 to 1)
                LineageComponents.METADATA: float  # Row transformation score (0 to 1, if applicable)
            }
        """
        # validate the ground truth format
        if not self._validate_lineage(gold_lineage):
            raise ValueError(f"Invalid gold lineage: {gold_lineage}")
        
        results = {}

        # Check if the format is correct
        is_format_correct = int(self._validate_lineage(predicted_lineage))
        results['format'] = is_format_correct

        # Turn the predicted and gold lineages into SchemaLineage objects
        try:
            predicted_lineage_obj = self._turn_dict_to_lineage(predicted_lineage)
            gold_lineage_obj = self._turn_dict_to_lineage(gold_lineage)
        except ValueError as e:
            # If conversion fails, return with format=0 and overall=0
            logging.error(f"Error converting lineage: {e}")
            results['format'] = 0
            results['overall'] = 0
            return results

        # Check if metadata is required and present in gold standard
        if self.has_metadata and not gold_lineage_obj.metadata:
            raise ValueError("Ground truth must have metadata when has_metadata=True")

        # Evaluate the source schema
        source_schema_score = self.evaluate_source_schema(predicted_lineage_obj, gold_lineage_obj)
        results[ScoreComponents.SOURCE_SCHEMA] = source_schema_score

        # Evaluate the source tables
        source_tables_score = self.evaluate_source_tables(predicted_lineage_obj, gold_lineage_obj)
        results[ScoreComponents.SOURCE_TABLE] = source_tables_score
        
        # Evaluate the transformation
        transformation_score = self.evaluate_transformation(predicted_lineage_obj, gold_lineage_obj)
        results[ScoreComponents.TRANSFORMATION] = transformation_score

        # Evaluate the aggregation
        aggregation_score = self.evaluate_aggregation(predicted_lineage_obj, gold_lineage_obj)
        results[ScoreComponents.AGGREGATION] = aggregation_score

        # Evaluate the metadata if applicable
        if self.has_metadata:
            metadata_score = self.evaluate_metadata(predicted_lineage_obj, gold_lineage_obj)
            results[ScoreComponents.METADATA] = metadata_score
            # Calculate the overall score
            results[ScoreComponents.ALL] = is_format_correct * source_schema_score * (
                self.weights[LineageComponents.SOURCE_TABLE] * source_tables_score +
                self.weights[LineageComponents.TRANSFORMATION] * transformation_score +
                self.weights[LineageComponents.AGGREGATION] * aggregation_score +
                self.weights[LineageComponents.METADATA] * metadata_score
            )
        else:
            # Calculate the overall score
            results[ScoreComponents.ALL] = is_format_correct * source_schema_score * (
                self.weights[LineageComponents.SOURCE_TABLE] * source_tables_score +
                self.weights[LineageComponents.TRANSFORMATION] * transformation_score +
                self.weights[LineageComponents.AGGREGATION] * aggregation_score
            )

        return results

    def evaluate_batch(self, predicted_lineage: List[Dict[str, str]], gold_lineage: List[Dict[str, str]]) -> List[Dict[str, float]]:
        """
        Evaluate a batch of predicted schema lineages against gold standards in parallel.
        
        Args:
            predicted_lineage: List of predicted schema lineages as dictionaries
            gold_lineage: List of gold standard schema lineages as dictionaries
            
        Returns:
            List of evaluation results dictionaries
            
        Raises:
            ValueError: If the lengths of predicted and gold lineages don't match
        """
        import multiprocessing as mp
        from functools import partial
        
        if len(predicted_lineage) != len(gold_lineage):
            raise ValueError(f"Number of predicted lineages ({len(predicted_lineage)}) " 
                             f"does not match number of gold lineages ({len(gold_lineage)})")
        
        # Use a reasonable number of processes (CPU count or batch size, whichever is smaller)
        num_processes = min(mp.cpu_count(), len(predicted_lineage))
        
        if num_processes <= 1 or len(predicted_lineage) <= 1:
            # For small batches, don't use multiprocessing
            return [self.evaluate(pred, gold) for pred, gold in zip(predicted_lineage, gold_lineage)]
        
        try:
            # Create a partial function with self as the first argument
            evaluate_func = partial(self._parallel_evaluate, self)
            
            # Create a pool of processes
            with mp.Pool(processes=num_processes) as pool:
                # Map the evaluation function to the data pairs
                results = pool.starmap(
                    evaluate_func,
                    zip(predicted_lineage, gold_lineage)
                )
            
            return results
        
        except Exception as e:
            logging.warning(f"Error in parallel processing: {e}")
            # Fall back to sequential processing if parallel fails
            return [self.evaluate(pred, gold) for pred, gold in zip(predicted_lineage, gold_lineage)]
    
    @staticmethod
    def _parallel_evaluate(self_instance, pred_lineage, gold_lineage):
        """
        Static method to support parallel evaluation.
        
        Args:
            self_instance: Instance of SchemaLineageEvaluator
            pred_lineage: A single predicted schema lineage
            gold_lineage: A single gold schema lineage
            
        Returns:
            Evaluation result dictionary
        """
        return self_instance.evaluate(pred_lineage, gold_lineage)

    def evaluate_source_schema(self, predicted_lineage: SchemaLineage, gold_lineage: SchemaLineage) -> float:
        """
        Evaluate the source schema of the predicted schema lineage.
        """
        predicted_schema = predicted_lineage.source_schema
        gold_schema = gold_lineage.source_schema

        # turn the strings into sets of columns
        predicted_set = parse_entity_names(predicted_schema)
        gold_set = parse_entity_names(gold_schema)

        # compare the sets
        return int(predicted_set == gold_set)
    
    def evaluate_source_tables(self, predicted_lineage: SchemaLineage, gold_lineage: SchemaLineage) -> float:
        """
        Evaluate the source tables of the predicted schema lineage.
        """
        predicted_tables = predicted_lineage.source_table
        gold_tables = gold_lineage.source_table

        # turn the strings into sets of tables. TRICKY 
        predicted_set = parse_entity_names(predicted_tables, delimiters=[';', ','])
        gold_set = parse_entity_names(gold_tables, delimiters=[';', ','])

        # calculate the fuzzy match score
        fuzzy_score = _fuzzy_match(predicted_set, gold_set)

        # calculate the f1 score
        f1_score = calc_f1_score(predicted_set, gold_set)

        # calculate the weighted score
        weighted_score = self.source_table_weights['f1'] * f1_score + self.source_table_weights['fuzzy'] * fuzzy_score

        return weighted_score
    
    def _evaluate_text_similarity(self, predicted_text: str, gold_text: str, 
                                weights: Dict[str, float]={'bleu': 0.5, 'weighted_bleu': 0.3, 'ast': 0.2}) -> float:
        """
        Helper method to evaluate text similarity between predicted and gold text.
        
        Args:
            predicted_text: The predicted text
            gold_text: The gold standard text
            weights: Weights dictionary for the different metrics (bleu, weighted_bleu, ast)
            special_operators: Optional special operators to use for BLEU calculation
            
        Returns:
            float: Weighted similarity score between 0 and 1
        """
        # edge cases, 'NULL' or 'None' (or similar, small letters) or 'nan' or empty string or etc,. those should be considered as the same
        if predicted_text.lower() in ['null', 'none', 'nan', 'nan', '']:
            predicted_text = ''
        if gold_text.lower() in ['null', 'none', 'nan', 'nan', '']:
            gold_text = ''
        
        # clean the code formatting
        predicted_text_cleaned = clean_code_formatting(predicted_text)
        gold_text_cleaned = clean_code_formatting(gold_text)

        # calculate the bleu score
        bleu_score = calc_bleu_score(predicted_text_cleaned, 
                                     gold_text_cleaned,
                                     special_operators=self.operators['sql'] | self.operators['python'])

        # calculate the weighted bleu score
        weighted_bleu_score = self.weighted_bleu_calculator.calc_score(predicted_text_cleaned, gold_text_cleaned)

        # calculate the ast score
        ast_score = self.ast_calculator.calc_ast_score(predicted_text_cleaned, gold_text_cleaned)

        # calculate the weighted score
        weighted_score = (weights['bleu'] * bleu_score + 
                        weights['weighted_bleu'] * weighted_bleu_score + 
                        weights['ast'] * ast_score)

        return weighted_score
    
    def evaluate_transformation(self, predicted_lineage: SchemaLineage, gold_lineage: SchemaLineage) -> float:
        """
        Evaluate the transformation of the predicted schema lineage.
        """
        return self._evaluate_text_similarity(
            predicted_lineage.transformation, 
            gold_lineage.transformation,
            self.transformation_weights
        )
    
    def evaluate_aggregation(self, predicted_lineage: SchemaLineage, gold_lineage: SchemaLineage) -> float:
        """
        Evaluate the aggregation of the predicted schema lineage.
        """
        return self._evaluate_text_similarity(
            predicted_lineage.aggregation, 
            gold_lineage.aggregation,
            self.aggregation_weights
        )
        
    def evaluate_metadata(self, predicted_lineage: SchemaLineage, gold_lineage: SchemaLineage) -> float:
        """
        Evaluate the metadata of the predicted schema lineage.
        """
        return self._evaluate_text_similarity(
            predicted_lineage.metadata, 
            gold_lineage.metadata,
            self.metadata_weights
        )
    

# helper functions
def _fuzzy_match(predicted: Set[str], truth: Set[str]) -> float:
    """
    Calculate fuzzy match score between two sets of strings.
    
    Parameters
    ----------
    predicted : Set[str]
        Set of predicted strings
    truth : Set[str]
        Set of ground truth strings
        
    Returns
    -------
    float
        Fuzzy match score between 0 and 1
    """
    if not predicted and not truth:
        return 1.0
    if not predicted or not truth:
        return 0.0
    
    # Convert sets to lists for indexing
    pred_list = list(predicted)
    truth_list = list(truth)
    
    # Initialize score matrix
    scores = np.zeros((len(pred_list), len(truth_list)))
    
    # Calculate similarity scores using get_fuzzy_match_score
    for i, pred in enumerate(pred_list):
        for j, true in enumerate(truth_list):
            scores[i][j] = calc_fuzzy_match_score(pred, true)
    
    # find the best match row-wise (for each prediction, find the best match in the truth)
    row_max_indices = np.argmax(scores, axis=1)
    mean_row_scores = np.mean(scores[np.arange(len(scores)), row_max_indices])

    # find the best match column-wise (for each truth, find the best match in the predictions)
    col_max_indices = np.argmax(scores, axis=0)
    mean_col_scores = np.mean(scores[col_max_indices, np.arange(len(scores.T))])
    
    return (mean_row_scores + mean_col_scores) / 2.0

    

        
