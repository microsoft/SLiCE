from fuzzywuzzy import fuzz
from typing import Set, Tuple, Dict, List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tree_sitter import Parser
from tree_sitter_languages import get_language
from collections import Counter
import math
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")


def calc_fuzzy_match_score(str1: str, str2: str):
    """
    Compare two strings and return a fuzzy match score between 0 and 1.
    1 means perfect match, higher scores indicate greater similarity.

    Args:
        string1 (str): First string to compare
        string2 (str): Second string to compare
        
    Returns:
        float: Similarity score between 0 and 1
    """
    ratio = fuzz.ratio(str1, str2)
    normalized_ratio = ratio / 100.0
    return normalized_ratio

def calc_precision_score(predicted: Set[str], truth: Set[str]):
    """
    Get the precision score between two sets of strings.
    Precision: all correct predictions / all predictions
    Args:
        predicted (Set[str]): The predicted set of strings
        truth (Set[str]): The truth set of strings
        
    Returns:
        float: The precision score between 0 and 1
    """
    if len(predicted) == 0 and len(truth) == 0:
        return 1
    if len(predicted) == 0 and len(truth) != 0:
        return 0
    tp = len(predicted.intersection(truth))
    precision = tp / len(predicted)

    return precision

def calc_recall_score(predicted: Set[str], truth: Set[str]):
    """
    Get the recall score between two sets of strings.
    Recall: all correct predictions / all truths
    """
    if len(truth) == 0 and len(predicted) == 0:
        return 1
    if len(truth) == 0 and len(predicted) != 0:
        return 0
    tp = len(predicted.intersection(truth))
    recall = tp / len(truth)

    return recall

def calc_f1_score(predicted: Set[str], truth: Set[str]):
    """
    Get the F1 score between two sets of strings.
    F1: 2 * (precision * recall) / (precision + recall)
    """
    precision = calc_precision_score(predicted, truth)
    recall = calc_recall_score(predicted, truth)

    if precision == 0 or recall == 0:
        return 0
    if precision == 0 and recall == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
 
def calc_bleu_score(predicted: str, 
                    truth: str, 
                    weights: Tuple[float, float, float, float]=(0.4, 0.3, 0.2, 0.1),
                    special_operators: Set[str] = None) -> float:
    """
    Calculate the BLEU score between a predicted and truth string.

    BLEU (Bilingual Evaluation Understudy) measures the similarity between
    two text segments by comparing n-gram matches. It returns a score between
    0 and 1, where 1 indicates perfect match.

    Args: 
        predicted (str): The predicted string
        truth (str): The truth string
        weights (Tuple): The weights for the BLEU score. Default is (0.4, 0.3, 0.2, 0.1), which emphasizes unigrams and bigrams.
        special_operators (Set[str]): Optional set of special operators to preserve during tokenization

    Returns:
        float: The BLEU score between 0 and 1
    """
    # handle edge cases
    if not predicted and not truth:
        return 1.0 # both empty strings
    
    if not predicted or not truth:
        return 0.0 # one empty string

    
    # Use default special operators if none provided
    if special_operators is None:
        special_operators = {'+=', '-=', '*=', '/=', '%=', '>=', '<=', '==', '!=', '<>', 
                            '=>', '->', '::', '||', '&&', '...', '**'}
    
    # Tokenize the text using the specialized code tokenizer
    predicted_tokens = _code_tokenize(predicted, special_operators=special_operators)
    truth_tokens = _code_tokenize(truth, special_operators=special_operators)

    max_token_len = max(len(predicted_tokens), len(truth_tokens))
    if max_token_len < 4:
        # calibrate the weights only take the first max_token_len and make the rest 0
        weights = tuple(weights[:max_token_len]) + (0,) * (len(weights) - max_token_len)

    # BLEU requires a list of references, but we only have one reference
    references = [truth_tokens]

    # Use smoothing to avoid zero scores when there's no n-gram overlap
    # Method 1 adds epsilon to precision with 0 counts
    smoothing = SmoothingFunction().method1

    try:
        score = sentence_bleu(references, predicted_tokens, weights=weights, smoothing_function=smoothing)
        return score
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        return 0.0


def _code_tokenize(code: str,
                  special_operators: Set[str] = {'+=', '-=', '*=', '/=', '%=', 
                                                 '>=', '<=', '==', '!=', '<>', 
                                                 '=>', '->', '::', '||', '&&', 
                                                 '...', '**'}) -> List[str]:
    """
    Tokenize code text for BLEU score calculation.
    
    This tokenizer is specialized for code, preserving important syntax elements
    like brackets, operators, etc. It also handles special multi-character operators
    commonly found in Python and SQL.
    
    Args:
        code (str): The code string to tokenize
        special_operators (Set[str]): Set of special multi-character operators to preserve
        
    Returns:
        List[str]: A list of tokens
    """
    # First, normalize whitespace
    text = re.sub(r'\s+', ' ', code.strip())
    placeholders = {}
    # Handle special operators before general tokenization
    # Sort operators by length (descending) to ensure longer operators are replaced first
    for i, op in enumerate(sorted(special_operators, key=len, reverse=True)):
        placeholder = f"SEPCIALOP{i}PLACEHOLDER"
        placeholders[placeholder] = op
        # Escape special characters for regex
        escaped_op = re.escape(op)
        text = re.sub(escaped_op, f" {placeholder} ", text)
    
    # Insert spaces around punctuation and special characters commonly found in code
    # Excluding special operators that we've already handled
    text = re.sub(r'([^\w\s])', r' \1 ', text)
    
    # Clean up any extra spaces that might have been introduced
    text = re.sub(r'\s+', ' ', text.strip())

    tokens = text.split()
    # replace placeholders with original operators
    for i, token in enumerate(tokens):
        if token in placeholders:
            tokens[i] = placeholders[token]

    # Filter out any empty tokens
    tokens = [token for token in tokens if token]
    return tokens

class WeightedBleuCalculator:
    """
    Calculator for weighted BLEU scores in code comparison, specifically
    designed for data transformation code evaluation.
    """
    def __init__(self, 
                 weights: Tuple[float, float, float, float]=(0.4, 0.3, 0.2, 0.1),
                 default_weight: float = 0.2,
                 keywords: Set[str] = None,
                 special_operators: Set[str] = None):
        self.ngram_weights = weights
        self.default_weight = default_weight
        
        # Use default special operators if none provided
        if special_operators is None:
            self.special_operators = {'+=', '-=', '*=', '/=', '%=', '>=', '<=', '==', '!=', '<>', 
                                     '=>', '->', '::', '||', '&&', '...', '**'}
        else:
            self.special_operators = special_operators
        
        if keywords is None:
            self.keywords = {
                # SQL keywords
                'select', 'from', 'where', 'group', 'by', 'having', 'order', 'join',
                'inner', 'outer', 'left', 'right', 'on', 'union', 'all', 'insert',
            'update', 'delete', 'create', 'table', 'view', 'with', 'as', 'case',
            'when', 'then', 'else', 'end', 'distinct', 'between', 'like', 'in',
            'exists', 'not', 'null', 'and', 'or', 'count', 'sum', 'avg', 'min', 'max',
            
            # Python data transformation keywords
            'def', 'return', 'if', 'else', 'elif', 'for', 'in', 'while', 'try',
            'except', 'import', 'from', 'class', 'lambda', 'map', 'filter', 'reduce',
            'zip', 'sorted', 'list', 'dict', 'set', 'tuple', 'append', 'extend',
            'pandas', 'numpy', 'pd', 'np', 'dataframe', 'series', 'array', 'groupby',
            'agg', 'apply', 'transform', 'pivot', 'merge', 'join', 'concat', 'value_counts'
            
            # PySpark keywords
            'spark', 'dataframe', 'rdd', 'select', 'where', 'filter', 'map',
            'flatmap', 'groupby', 'agg', 'withcolumn', 'join', 'unionall'
            }
        else:
            self.keywords = keywords


    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """
        Generate n-grams from a list of tokens.
        Args:
            tokens: List of tokens
            n: The n-gram size
            
        Returns:
            List[Tuple[str, ...]]: List of n-grams as tuples
        """
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def _brevity_penalty(self,
                        truth_len: int,
                        predicted_len: int) -> float:
        """
        Calculate brevity penalty based on truth and predicted lengths. If the predicted code is shorter than the truth, the penalty multiplier is less than 1
        """
        if predicted_len >= truth_len:
            return 1.0
        elif predicted_len == 0:
                return 0.0
        else:
            return math.exp(1 - (truth_len / predicted_len))

    def _modified_precision(self, truth_tokens: List[str], 
                          predicted_tokens: List[str],
                          token_weights: Dict[str, float],
                          n: int) -> Tuple[float, float]:
        """
        Calculate the modified weighted precision for a given n-gram order.
        
        Args:
            truth_tokens: Tokens from the truth code
            predicted_tokens: Tokens from the predicted code
            token_weights: Dictionary mapping tokens to weights
            n: The n-gram order
            default_weight: The default weight for tokens that are not in the token_weights dictionary
        Returns:
            Tuple[float, float]: Numerator and denominator of the precision
        """
        # Handle edge cases
        if len(predicted_tokens) < n or len(truth_tokens) < n:
            return 0.0, 1.0

        # Get n-grams from predicted and truth tokens
        predicted_ngrams = Counter(self._get_ngrams(predicted_tokens, n))
        truth_ngrams = Counter(self._get_ngrams(truth_tokens, n))
        
        # Calculate clipped counts (min of predicted and truth counts for each n-gram)
        clipped_counts = {
            ngram: min(count, predicted_ngrams.get(ngram, 0)) 
            for ngram, count in truth_ngrams.items()
        }
        
        # For unigrams, apply token weights
        if n == 1 and token_weights:
            numerator = 0.0
            for ngram, count in clipped_counts.items():
                # Get weight for the token or use default weight
                token = ngram[0]
                weight = token_weights.get(token, self.default_weight)
                numerator += count * weight
                
            denominator = 0.0
            for ngram, count in predicted_ngrams.items():
                token = ngram[0]
                weight = token_weights.get(token, self.default_weight)
                denominator += count * weight
            
            # Ensure denominator is at least 1 to avoid division by zero
            denominator = max(1.0, denominator)
        else:
            # For higher-order n-grams, use regular counts
            numerator = sum(clipped_counts.values())
            denominator = max(1.0, sum(predicted_ngrams.values()))
            
        return numerator, denominator

    def calc_score(self, predicted: str, truth: str, weights: Tuple = None) -> float:
        """
        Calculate the weighted BLEU score between predicted and truth code strings.
        
        Args: 
            predicted (str): The predicted code string
            truth (str): The ground truth code string
            weights (Tuple): Optional custom weights for n-gram orders
            
        Returns:
            float: The weighted BLEU score between 0 and 1
        """
        # Use provided weights or default weights
        weights = weights or self.ngram_weights
        
        # Tokenize the text using the specialized code tokenizer
        predicted_tokens = _code_tokenize(predicted, special_operators=self.special_operators)
        truth_tokens = _code_tokenize(truth, special_operators=self.special_operators)
        
        # Handle edge cases
        if not predicted_tokens and not truth_tokens:
            return 1.0  # Both empty strings
        if not predicted_tokens or not truth_tokens:
            return 0.0  # One empty string
            
        # Create token weights dictionary for code-specific weighting
        # Keywords get higher weights
        token_weights = {}
        for token in set(truth_tokens):
            if token.lower() in self.keywords:
                token_weights[token] = 1.0
        if len(token_weights) == 0:
            return calc_bleu_score(predicted, truth, weights, special_operators=self.special_operators)
                
        # Calculate precision for each n-gram order
        precisions = []
        for i, weight in enumerate(weights, start=1):
            numerator, denominator = self._modified_precision(
                truth_tokens, 
                predicted_tokens, 
                token_weights, 
                i
            )
            precisions.append((numerator, denominator))
            
        # If there's no unigram matches, return 0
        if precisions[0][0] == 0:
            return 0.0
            
        # Calculate brevity penalty
        bp = self._brevity_penalty(len(truth_tokens), len(predicted_tokens))
        
        # Calculate final score using the geometric mean of modified precisions
        score = 0.0
        def _smoothing(numerator, denominator, epsilon=0.1):
            if numerator == 0:
                return (numerator + epsilon) / denominator
            else:
                return numerator / denominator
        for i, ((numerator, denominator), weight) in enumerate(zip(precisions, weights)):
            if numerator > 0:
                # We take log of the precision to use in the geometric mean
                precision = _smoothing(numerator, denominator)
                score += weight * math.log(precision)
                
        # Apply brevity penalty and convert from log space
        score = bp * math.exp(score)
        
        return score
    
def normalize_quotes(text: str) -> str:
    """
    Normalize quotes by converting all double quotes to single quotes
    
    Args:
        text: Text with mixed quote styles
        
    Returns:
        Text with normalized quotes
    """
    if not text:
        return ""
    
    import re
    
    # Replace double-quoted content with single-quoted content
    text = re.sub(r'"([^"]*)"', r"'\1'", text)
    
    return text

def clean_code_formatting(code: str) -> str:
    """
    Clean code formatting, remove excessive whitespace while preserving basic structure
    
    Args:
        code: Original code string
        
    Returns:
        Cleaned code
    """
    if not code:
        return ""
    
    # Normalize quotes first
    code = normalize_quotes(code)
    
    # Split into lines and remove leading/trailing whitespace from each line
    lines = [line.strip() for line in code.split('\n')]
    
    # Remove empty lines
    lines = [line for line in lines if line]
    
    # For single-line code, return directly
    if len(lines) == 1:
        return lines[0]
    
    # For multi-line code, maintain basic structure but normalize indentation
    cleaned_code = []
    for line in lines:
        # Remove line-end comments
        if '#' in line:
            line = line[:line.find('#')].rstrip()
        if '--' in line:
            line = line[:line.find('--')].rstrip()
        
        if line:  # Skip empty lines
            cleaned_code.append(line)
        
    return ' '.join(cleaned_code)


class ASTCalculator:
    """
    Calculate the AST similarity between two code strings.
    """
    def __init__(self,
                 python_syntax: Set[str] = None,
                 sql_syntax: Set[str] = None,
                 csharp_syntax: Set[str] = None):
        """
        Initialize AST calculator with optional syntax patterns for language detection.
        
        Args:
            python_syntax: Set of Python syntax patterns
            sql_syntax: Set of SQL syntax patterns
            csharp_syntax: Set of C# syntax patterns
        """
        self.python_syntax = python_syntax or {
            # basic python syntax
            'def ', 'import ', 'class ', 'if ', 'for ', 'while ', 'print(',
            'return ', 'with ', 'try:', 'except:', 'lambda ', '==',
            'yield ', 'async ', 'await ', '+=', '-=', '>=', '<=', '!=',
            '=>', '**', '//', '__init__', '.py', 'self.', 'lower(', 'upper(',
            
            # data analysis related
            'pandas', 'pd.', 'np.', 'numpy', 'matplotlib', 'plt.',
            'df.', '.iloc', '.loc', '.groupby', '.agg', '.apply',
            '.drop', '.fillna', '.value_counts', '.describe', '.head',
            '.merge', '.join', '.concat', '.read_csv', '.to_csv',
            
            # PySpark related
            'spark.', 'sparkContext', 'SparkSession', 'SparkConf',
            'pyspark.', 'rdd.', '.rdd', '.collect()', '.count()',
            '.createOrReplaceTempView', '.registerTempTable',
            '.udf', '.withColumn', '.select', '.filter', '.where',
            '.withColumnRenamed', '.join', '.groupBy', '.agg',
            '.sql', '.createDataFrame', '.read.parquet', '.write.parquet',
            '.read.csv', '.write.csv', '.saveAsTextFile',
        }

        # Default SQL syntax features if not provided
        self.sql_syntax = sql_syntax or {
            # basic SQL syntax
            'select ', 'from ', 'where ', 'group by ', 'order by ', 'having ',
            'join ', 'inner join', 'left join', 'right join', 'full join',
            'insert into', 'update ', 'delete from', 'create table', 'alter table',
            'drop table', 'truncate', 'union ', 'distinct', 'count(*)', 'as ',
            'between', 'like ', 'in (', 'exists', 'desc', 'asc', 'order by', 'group by',
            
            # data warehouse/ETL related SQL
            'partition by', 'over (', 'window', 'rank()', 'dense_rank()',
            'row_number()', 'lead(', 'lag(', 'first_value(', 'last_value(',
            'merge into', 'when matched', 'when not matched',
            'with ', 'cte', 'recursive', 'pivot', 'unpivot',
            
            # advanced functions
            'case when', 'coalesce(', 'nullif(', 'cast(', 'extract(',
            'date_trunc', 'to_char(', 'to_date(', 'regexp_replace(',
            'regexp_extract(', 'substring(', 'concat(', 'lower(', 'upper(',
        }
        
        self.csharp_syntax = csharp_syntax or {
            # basic C# syntax
            'using ', 'namespace ', 'class ', 'interface ', 'struct ', 'enum ',
            'public ', 'private ', 'protected ', 'internal ', 'static ', 'void ',
            'return ', 'if ', 'else ', 'for ', 'foreach ', 'while ', 'do ',
            'switch ', 'case ', 'break ', 'continue ', 'try ', 'catch ', 'finally ',
            'throw ', 'new ', 'this ', 'base ', 'async ', 'await ', 'var ', 'const ',
        }

        self.language_map = {
            'python': 'python',
            'sql': 'sql',
            'csharp': 'c_sharp',
        }

        self.sql_weight = None
        self.python_weight = None
        self.csharp_weight = None
        self.python_ast_score = None
        self.sql_ast_score = None
        self.csharp_ast_score = None
        # Cache the parsers to avoid creating them multiple times
        self._parsers = {}
    
    def _get_parser(self, lang: str):
        """Get a parser for the given language with caching"""
        if lang not in self._parsers:
            try: 
                tree_sitter_parser = self.language_map.get(lang, lang)
                language = get_language(tree_sitter_parser)
                parser = Parser()
                parser.set_language(language)
                self._parsers[lang] = parser
            except Exception as e:
                raise ValueError(f"Failed to get parser for language '{lang}': {str(e)}")
        return self._parsers[lang]
    
    def calc_language_confident(self, code: str) -> Tuple[float, float, float]:
        """
        Calculate the language confident score for a given code snippet.
        Args:
            code: Code string
        Returns:
            Tuple of (python_confident, sql_confident, csharp_confident), values between 0 and 1
        """
        code_lower = code.lower()
        python_matches = sum(code_lower.count(feature.lower()) for feature in self.python_syntax)
        sql_matches = sum(code_lower.count(feature.lower()) for feature in self.sql_syntax)
        csharp_matches = sum(code_lower.count(feature.lower()) for feature in self.csharp_syntax)
        total_matches = python_matches + sql_matches + csharp_matches
        if total_matches == 0:
            return 0.0, 0.0, 0.0
        return (min(1, python_matches / total_matches), 
                min(1, sql_matches / total_matches),
                min(1, csharp_matches / total_matches))

    def _get_all_sub_trees(self, root_node):
        """
        Extract all sub-trees from an AST.
        
        Args:
            root_node: The root node of the AST
            
        Returns:
            List of tuples (sub_tree_sexp, depth)
        """
        node_stack = []
        sub_tree_sexp_list = []
        
        node_stack.append([root_node, 1])  # Start with depth 1
        
        while node_stack:
            cur_node, cur_depth = node_stack.pop()
                
            sub_tree_sexp_list.append((cur_node.sexp(), cur_depth))
            
            for child_node in cur_node.children:
                if len(child_node.children) != 0:
                    node_stack.append([child_node, cur_depth + 1])
                    
        return sub_tree_sexp_list
        
    def calc_ast_similarity(self, 
                            predicted: str, 
                            truth: str,
                            lang: str) -> float:
        """
        Calculate AST similarity score for a specific language.
        
        Args:
            predicted: Predicted code string
            truth: Truth code string
            lang: Language ('python' or 'sql', 'csharp')
            
        Returns:
            Similarity score between 0 and 1
        """
        if lang not in self.language_map:
            raise ValueError(f"Unsupported language: {lang}")
        try: 
            parser = self._get_parser(lang)
        except Exception as e:
            raise ValueError(f"Failed to get parser for language '{lang}': {str(e)}")
            return 0.0
        
        # clean code formatting
        predicted = clean_code_formatting(predicted)
        truth = clean_code_formatting(truth)

        # parse the code
        try:
            predicted_tree = parser.parse(bytes(predicted, 'utf8')).root_node
            truth_tree = parser.parse(bytes(truth, 'utf8')).root_node
        except Exception as e:
            raise ValueError(f"Failed to parse code for language '{lang}': {str(e)}")
            return 0.0

        # extract sub trees
        predicted_sub_tree = self._get_all_sub_trees(predicted_tree)
        truth_sub_tree = self._get_all_sub_trees(truth_tree)

        # Extract just the S-expressions from the truth sub-trees for comparison
        truth_sexps = [sexp for sexp, _ in truth_sub_tree]
        
        # count matching sub-trees
        match_count = sum(1 for sub_tree, _ in predicted_sub_tree if sub_tree in truth_sexps)
        total_count = len(predicted_sub_tree)

        # avoid division by zero
        if total_count == 0:
            return 0.0

        # calculate similarity score
        similarity_score = match_count / total_count
        return similarity_score

    def calc_ast_score(self, 
                       predicted: str, 
                       truth: str) -> float:
        """
        Calculate weighted AST similarity score considering both Python and SQL.
        
        Args:
            reference: Reference code string
            candidate: Candidate code string
            
        Returns:
            Weighted similarity score between 0 and 1
        """
        python_confident, sql_confident, csharp_confident = self.calc_language_confident(predicted)

        # normalize weights
        if python_confident + sql_confident == 0:
            sql_confident = 1.0
            python_confident = 0.0
            csharp_confident = 0.0

        self.python_weight = python_confident
        self.sql_weight = sql_confident
        self.csharp_weight = csharp_confident
        # calculate similarity scores
        python_score = self.calc_ast_similarity(predicted, truth, 'python')
        sql_score = self.calc_ast_similarity(predicted, truth, 'sql')
        csharp_score = self.calc_ast_similarity(predicted, truth, 'csharp')

        self.python_ast_score = python_score
        self.sql_ast_score = sql_score
        self.csharp_ast_score = csharp_score
        # calculate weighted score
        return python_confident * python_score + sql_confident * sql_score + csharp_confident * csharp_score
        

                
                