import pytest

from slice.eval import (
    calc_f1_score,
    calc_fuzzy_match_score,
    calc_precision_score,
    calc_recall_score,
)


def test_get_fuzzy_match_score():
    # Test exact matches
    assert calc_fuzzy_match_score("hello", "hello") == 1.0
    assert calc_fuzzy_match_score("", "") == 1.0

    # Test similar strings
    assert calc_fuzzy_match_score("hello", "hallo") == 0.8
    assert calc_fuzzy_match_score("hello", "helloo") > 0.8

    # Test different strings
    assert calc_fuzzy_match_score("hello", "world") < 0.5
    assert calc_fuzzy_match_score("hello", "") == 0.0

    # Test case sensitivity
    assert calc_fuzzy_match_score("Hello", "hello") == 0.8

    # Test with special characters
    assert calc_fuzzy_match_score("hello!", "hello!") == 1.0
    assert calc_fuzzy_match_score("hello!", "hello") > 0.8


def test_calc_precision_score():
    # Test exact matches
    assert calc_precision_score({"a", "b"}, {"a", "b"}) == 1.0
    assert calc_precision_score(set(), set()) == 1.0

    # Test partial matches
    assert calc_precision_score({"a", "b", "c"}, {"a", "b"}) == 2 / 3
    assert calc_precision_score({"a", "b"}, {"a", "b", "c"}) == 1.0

    # Test no matches
    assert calc_precision_score({"a", "b"}, {"c", "d"}) == 0.0

    # Test empty sets
    assert calc_precision_score(set(), {"a", "b"}) == 0.0
    assert calc_precision_score({"a", "b"}, set()) == 0.0


def test_calc_recall_score():
    # Test exact matches
    assert calc_recall_score({"a", "b"}, {"a", "b"}) == 1.0
    assert calc_recall_score(set(), set()) == 1.0

    # Test partial matches
    assert calc_recall_score({"a", "b"}, {"a", "b", "c"}) == 2 / 3
    assert calc_recall_score({"a", "b", "c"}, {"a", "b"}) == 1

    # Test no matches
    assert calc_recall_score({"a", "b"}, {"c", "d"}) == 0.0

    # Test empty sets
    assert calc_recall_score(set(), {"a", "b"}) == 0.0
    assert calc_recall_score({"a", "b"}, set()) == 0.0


def test_calc_f1_score():
    # Test perfect matches
    assert calc_f1_score({"a", "b"}, {"a", "b"}) == 1.0

    # Test partial matches
    assert calc_f1_score({"a", "b", "c"}, {"a", "b"}) == 0.8
    assert calc_f1_score({"a", "b"}, {"a", "b", "c"}) == 0.8

    # Test no matches
    assert calc_f1_score({"a", "b"}, {"c", "d"}) == 0.0

    # Test empty sets
    assert calc_f1_score(set(), {"a", "b"}) == 0.0
    assert calc_f1_score({"a", "b"}, set()) == 0.0

    # Test edge cases
    assert calc_f1_score({"a"}, {"a", "b"}) == 2 / 3
    assert calc_f1_score({"a", "b"}, {"a"}) == 2 / 3


def test_code_tokenize():
    from slice.eval import _code_tokenize

    # Test basic code tokenization
    assert _code_tokenize("x = y + 1") == ["x", "=", "y", "+", "1"]
    assert _code_tokenize("if(x>0){return x;}") == [
        "if",
        "(",
        "x",
        ">",
        "0",
        ")",
        "{",
        "return",
        "x",
        ";",
        "}",
    ]

    # Test with different whitespace patterns
    assert _code_tokenize("x  =  y  +  1") == ["x", "=", "y", "+", "1"]
    assert _code_tokenize("x=\ny+\n1") == ["x", "=", "y", "+", "1"]
    assert _code_tokenize("x\t=\ty\t+\t1") == ["x", "=", "y", "+", "1"]

    # Test with string literals
    assert _code_tokenize('x = "hello world"') == ["x", "=", '"', "hello", "world", '"']
    assert _code_tokenize("x = 'hello world'") == ["x", "=", "'", "hello", "world", "'"]

    # Test empty input
    assert _code_tokenize("") == []
    assert _code_tokenize("   ") == []

    # Test special operators in Python
    assert _code_tokenize("x += 1") == ["x", "+=", "1"]
    assert _code_tokenize("x -= 1") == ["x", "-=", "1"]
    assert _code_tokenize("if x >= 0 and y <= 10:") == [
        "if",
        "x",
        ">=",
        "0",
        "and",
        "y",
        "<=",
        "10",
        ":",
    ]
    assert _code_tokenize("x == y") == ["x", "==", "y"]
    assert _code_tokenize("x != y") == ["x", "!=", "y"]
    assert _code_tokenize("a ** 2 + b ** 2") == ["a", "**", "2", "+", "b", "**", "2"]
    assert _code_tokenize("lambda x: x**2") == ["lambda", "x", ":", "x", "**", "2"]

    # Test special operators in SQL
    assert _code_tokenize("WHERE age <> 30") == ["WHERE", "age", "<>", "30"]
    assert _code_tokenize("WHERE col1 => col2") == ["WHERE", "col1", "=>", "col2"]
    assert _code_tokenize("customer->>'name'") == [
        "customer",
        "->",
        ">",
        "'",
        "name",
        "'",
    ]
    assert _code_tokenize("col1 || col2") == ["col1", "||", "col2"]

    # Test combined Python/SQL syntax
    assert _code_tokenize("df.query('age >= 18 and age <= 65')") == [
        "df",
        ".",
        "query",
        "(",
        "'",
        "age",
        ">=",
        "18",
        "and",
        "age",
        "<=",
        "65",
        "'",
        ")",
    ]

    # Test with custom special operators
    custom_ops = ["===", "!==", "?:", "??"]
    assert _code_tokenize("x === y", special_operators=custom_ops) == ["x", "===", "y"]
    assert _code_tokenize("x !== y", special_operators=custom_ops) == ["x", "!==", "y"]
    assert _code_tokenize(
        "result = condition ?: true_val", special_operators=custom_ops
    ) == ["result", "=", "condition", "?:", "true_val"]


def test_calc_bleu_score():
    from slice.eval import calc_bleu_score

    # Test exact matches
    assert calc_bleu_score("x = y + 1", "x = y + 1") == 1.0
    assert calc_bleu_score("", "") == 1.0

    # Test similar code with different variable names
    score1 = calc_bleu_score("x = y + 1", "a = b + 1")
    assert 0.1 < score1 < 0.3

    # Test similar code with different operators
    score2 = calc_bleu_score("x = y + 1", "x = y - 1")
    assert 0.4 < score2 < 0.6

    # Test completely different code
    score3 = calc_bleu_score("x = y + 1", "if (x > 0) return x;")
    assert score3 < 0.3  # Should be low due to different structure

    # Test with different whitespace
    assert calc_bleu_score("x=y+1", "x = y + 1") == 1.0

    # # Test with comments
    # score4 = calc_bleu_score("x = y + 1 // comment", "x = y + 1")
    # assert score4 > 0.5  # Should be high despite comment difference

    # Test edge cases
    assert calc_bleu_score("", "x = y + 1") == 0.0
    assert calc_bleu_score("x = y + 1", "") == 0.0

    # Test with custom weights
    custom_weights = (1, 0, 0, 0)  # Only using 1 grams
    score5 = calc_bleu_score("x = y + 1", "x = y - 1", weights=custom_weights)
    assert score5 == 0.8  # Should still be high with custom weights

    # Test with longer code snippets
    code1 = """
    def add(x, y):
        return x + y
    """
    code2 = """
    def subtract(x, y):
        return x - y
    """
    score6 = calc_bleu_score(code1, code2)
    assert (
        0.3 < score6 < 0.7
    )  # Should be moderate due to similar structure but different operations


def test_weighted_bleu_calculator():
    from slice.eval import WeightedBleuCalculator, calc_bleu_score

    # Initialize calculator with default weights
    calculator = WeightedBleuCalculator()

    # Test exact matches
    assert calculator.calc_score("IDEAS_Teams as Teams", "IDEAS_Teams as Teams") == 1.0
    assert calculator.calc_score("", "") == 1.0

    # Test similar code with different variable names
    score1 = calculator.calc_score("x = y + 1", "a = b + 1")
    assert 0.1 < score1 < 0.3

    # Test similar code with different operators
    score2 = calculator.calc_score("x = y + 1", "x = y - 1")
    assert 0.4 < score2 < 0.6

    # Test with SQL keywords
    sql1 = "SELECT * FROM users WHERE age > 18"
    sql2 = "SELECT * FROM users WHERE age < 18"
    score3 = calculator.calc_score(sql1, sql2)
    score4 = calc_bleu_score(sql1, sql2)
    assert score3 > 0.7  # Should be high due to matching keywords
    assert score3 > score4  # weighted bleu has less weight on non-keywords

    # Test with Python keywords
    py1 = "def add(x, y): return x + y"
    py2 = "def sub(x, y): return x - y"
    score4 = calculator.calc_score(py1, py2)
    score5 = calc_bleu_score(py1, py2)
    assert score4 > 0.6  # Should be high due to matching keywords
    assert score4 > score5

    # Test with different default weights
    calculator_low_weight = WeightedBleuCalculator(default_weight=0.1)
    calculator_high_weight = WeightedBleuCalculator(default_weight=0.5)
    sql1 = "IDEAS_Teams as Teams from table"
    sql2 = "IDEAS_Teams as Teams"
    score5 = calculator_low_weight.calc_score(sql1, sql2)
    score6 = calculator_high_weight.calc_score(sql1, sql2)
    assert (
        score5 > score6
    )  # add lower weight to non-keywords, more penalty for not matching keywords

    # Test with custom ngram weights
    custom_weights = (0.6, 0.3, 0.1, 0.0)  # Emphasize unigrams
    calculator_custom = WeightedBleuCalculator(weights=custom_weights)
    score6 = calculator_custom.calc_score("x = y + 1", "x = y - 1")
    assert score6 > 0.5  # Should be high with custom weights

    # Test with longer code snippets
    code1 = """
    def process_data(df):
        df = df.filter(df.age > 18)
        df = df.groupBy("gender").agg({"salary": "avg"})
        return df
    """
    code2 = """
    def process_data(df):
        df = df.filter(df.age < 18)
        df = df.groupBy("gender").agg({"salary": "max"})
        return df
    """
    score7 = calculator.calc_score(code1, code2)
    assert score7 > 0.6  # Should be high due to matching structure and keywords

    # Test edge cases
    assert calculator.calc_score("", "x = y + 1") == 0.0
    assert calculator.calc_score("x = y + 1", "") == 0.0

    # Test with mixed keywords and non-keywords
    mixed1 = "df = pd.DataFrame(data) if data else None"
    mixed2 = "df = pd.DataFrame(data) if not data else None"
    score8 = calculator.calc_score(mixed1, mixed2)
    assert 0.7 < score8 < 0.9  # Should be high due to matching structure and keywords

    # test the codes comnbining both python and sql keywords
    str1 = "Select IDEAs_outook as Outlook; df['outlook'].max()"
    str2 = "Select IDEAs_outook as Outlook; df.groupby(datetime.now())['outlook'].value_counts() as counts"
    score9 = calculator.calc_score(str1, str2)
    score10 = calc_bleu_score(str1, str2)
    assert score9 > score10  # weighted bleu has less weight on non-keywords


def test_weighted_bleu_calculator_with_special_operators():
    from slice.eval import WeightedBleuCalculator

    # Initialize calculator with default settings
    calculator = WeightedBleuCalculator()

    # Test Python special operators
    assert calculator.calc_score("x += 1", "x += 1") == 1.0

    # Compare different operators but same structure
    score1 = calculator.calc_score("x += 1", "x -= 1")
    assert score1 < 0.3  # should not be high because operators are different

    # Test SQL special operators
    sql1 = "SELECT * FROM users WHERE age >= 18 AND income <= 50000"
    sql2 = "SELECT * FROM users WHERE age >= 18 AND income >= 50000"
    score2 = calculator.calc_score(sql1, sql2)
    assert 0.7 < score2 < 1.0  # Should be high due to mostly matching

    # Test arrow operators in SQL/JSON
    json1 = "data->>'name'"
    json2 = "data->>'name'"
    assert calculator.calc_score(json1, json2) == 1.0

    # Test more complex expressions
    expr1 = "x ** 2 + y ** 2 <= r ** 2"
    expr2 = "x ** 2 + y ** 2 >= r ** 2"
    score3 = calculator.calc_score(expr1, expr2)
    assert (
        0.7 < score3 < 1.0
    )  # Should be high despite the different comparison operator

    # Test with custom special operators
    custom_ops = {"===", "!==", "?:", "??"}
    custom_calculator = WeightedBleuCalculator(special_operators=custom_ops)

    score4 = custom_calculator.calc_score("x === y", "x !== y")
    assert score4 < 0.3  # Should recognize the similarity in structure

    # Test special operators in a larger context
    code1 = """
    if (status === 'active' && count > 0) {
        result = value ?? defaultValue;
    }
    """
    code2 = """
    if (status !== 'active' && count > 0) {
        result = value ?? fallbackValue;
    }
    """
    score5 = custom_calculator.calc_score(code1, code2)
    assert 0.7 < score5 < 1.0  # Should recognize the structural similarity


def test_ast_calculator_init():
    from slice.eval import ASTCalculator

    # Test default initialization
    calculator = ASTCalculator()
    assert len(calculator.python_syntax) > 0
    assert len(calculator.sql_syntax) > 0
    assert calculator.language_map == {
        "python": "python",
        "sql": "sql",
        "csharp": "c_sharp",
    }
    assert calculator._parsers == {}

    # Test custom initialization
    custom_python = {"def", "import", "class"}
    custom_sql = {"select", "from", "where"}
    calculator = ASTCalculator(python_syntax=custom_python, sql_syntax=custom_sql)
    assert calculator.python_syntax == custom_python
    assert calculator.sql_syntax == custom_sql


def test_ast_calculator_language_confidence():
    from slice.eval import ASTCalculator

    calculator = ASTCalculator()

    # Test pure Python code
    py_code = """
    def calculate(x, y):
        result = x + y
        return result
    """
    py_conf, sql_conf, cs_conf = calculator.calc_language_confident(py_code)
    assert py_conf > 0.6  # Should be high Python confidence
    assert sql_conf < 0.2  # Should be low SQL confidence

    # Test pure SQL code
    sql_code = """
    SELECT id, name, age 
    FROM users
    WHERE age > 18
    ORDER BY name
    """
    py_conf, sql_conf, cs_conf = calculator.calc_language_confident(sql_code)
    assert py_conf < 0.2  # Should be low Python confidence
    assert sql_conf > 0.8  # Should be high SQL confidence

    # Test mixed code
    mixed_code = """
    def process_data():
        query = "SELECT * FROM users WHERE age > 18"
        df = pd.read_sql(query, conn)
        return df
    """
    py_conf, sql_conf, cs_conf = calculator.calc_language_confident(mixed_code)
    # Both should have reasonable confidence
    assert py_conf > 0.4
    assert sql_conf > 0.2


def test_ast_calculator_get_parser():
    from slice.eval import ASTCalculator

    calculator = ASTCalculator()

    # Test getting Python parser
    python_parser = calculator._get_parser("python")
    assert python_parser is not None

    # Test getting SQL parser
    sql_parser = calculator._get_parser("sql")
    assert sql_parser is not None

    # Test getting C# parser
    csharp_parser = calculator._get_parser("csharp")
    assert csharp_parser is not None

    # Test parser caching
    assert calculator._parsers["python"] is python_parser
    assert calculator._parsers["sql"] is sql_parser
    assert calculator._parsers["csharp"] is csharp_parser

    # Test getting the same parser again (should use cache)
    python_parser2 = calculator._get_parser("python")
    assert python_parser2 is python_parser


def test_ast_calculator_get_all_sub_trees():
    from slice.eval import ASTCalculator

    calculator = ASTCalculator()

    # Simple Python code
    code = "x = 1 + 2"
    parser = calculator._get_parser("python")
    root_node = parser.parse(bytes(code, "utf8")).root_node

    # Without leaf nodes
    sub_trees = calculator._get_all_sub_trees(root_node)
    assert len(sub_trees) > 0
    assert all(isinstance(item, tuple) for item in sub_trees)
    assert all(isinstance(item[0], str) for item in sub_trees)
    assert all(isinstance(item[1], int) for item in sub_trees)

    # With leaf nodes
    sub_trees_with_leaves = calculator._get_all_sub_trees(root_node)
    assert len(sub_trees_with_leaves) >= len(sub_trees)  # Should include more nodes


def test_ast_calculator_ast_similarity():
    from slice.eval import ASTCalculator

    calculator = ASTCalculator()

    # Test identical Python code
    code1 = "x = 1 + 2"
    code2 = "x = 1 + 2"
    similarity = calculator.calc_ast_similarity(code1, code2, "python")
    assert similarity == 1.0

    # Test different Python code structure
    code4 = "if x > 0: print(x)"
    similarity = calculator.calc_ast_similarity(code1, code4, "python")
    assert similarity < 0.5  # Should be low due to different structure

    # Test identical SQL code
    sql1 = "SELECT id FROM users"
    sql2 = "SELECT id FROM users"
    similarity = calculator.calc_ast_similarity(sql1, sql2, "sql")
    assert similarity == 1.0

    # Test similar SQL code with column changes
    sql3 = "SELECT name FROM users"
    similarity = calculator.calc_ast_similarity(sql1, sql3, "sql")
    assert similarity == 1.0  # should be 1.0 because the structure is the same


def test_ast_calculator_ast_score():
    from slice.eval import ASTCalculator

    calculator = ASTCalculator()

    # Test Python code
    py_code1 = "def add(x, y): return x + y"
    py_code2 = "def add(a, b): return a + b"
    score_py = calculator.calc_ast_score(py_code1, py_code2)
    assert score_py == 1.0  # should be 1.0 because the structure is the same

    # Test SQL code
    sql_code1 = "SELECT * FROM users WHERE age > 18"
    sql_code2 = "SELECT * FROM users WHERE age < 18"
    score_sql = calculator.calc_ast_score(sql_code1, sql_code2)
    assert score_sql == 1.0  # should be 1.0 because the structure is the same

    # Test mixed code
    mixed_code1 = "def query(): return 'SELECT * FROM users'"
    mixed_code2 = "def query(): return 'SELECT name FROM users'"
    score_mixed = calculator.calc_ast_score(mixed_code1, mixed_code2)
    assert 0.6 < score_mixed < 1.0  # Should be high for similar structure

    # Test vastly different code
    diff_code1 = "x = 1 + 2"
    diff_code2 = "for i in range(10): print(i)"
    score_diff = calculator.calc_ast_score(diff_code1, diff_code2)
    assert score_diff < 0.5  # Should be low for different structure


def test_ast_calculator_edge_cases():
    from slice.eval import ASTCalculator

    calculator = ASTCalculator()

    # Test empty strings
    assert calculator.calc_language_confident("") == (0.0, 0.0, 0.0)

    # Calculate AST similarity with empty strings
    empty_similarity = calculator.calc_ast_similarity("", "", "python")
    assert empty_similarity == 1.0

    # Test with different whitespace and formatting
    formatted_code1 = """
    def add(x, y):
        return x + y
    """
    formatted_code2 = "def add(x, y): return x + y"

    # AST should be the same regardless of formatting
    assert (
        calculator.calc_ast_similarity(formatted_code1, formatted_code2, "python")
        == 1.0
    )


def test_ast_calculator_error_handling():
    from slice.eval import ASTCalculator

    calculator = ASTCalculator()

    # Test with unsupported language
    with pytest.raises(ValueError):
        calculator.calc_ast_similarity("x = 1", "x = 2", "javascript")

    # Test with syntax errors in both code snippets
    bad_code1 = "def missing_parenthesis: return x"
    bad_code2 = "if x > 0 print(x)"

    # Should handle gracefully without raising exceptions
    result = calculator.calc_ast_score(bad_code1, bad_code2)
    assert result == 0.0 or isinstance(result, float)


def test_ast_calculator_csharp():
    from slice.eval import ASTCalculator

    calculator = ASTCalculator()

    # Test identical C# code
    code1 = "public class User { public string Name { get; set; } }"
    code2 = "public class User { public string Name { get; set; } }"
    similarity = calculator.calc_ast_similarity(code1, code2, "csharp")
    assert similarity == 1.0

    # Test similar C# code with different property names
    code3 = "public class User { public string FirstName { get; set; } }"
    similarity = calculator.calc_ast_similarity(code1, code3, "csharp")
    assert similarity == 1.0  # Structure is identical

    # Test different C# code structure
    code4 = "public void ProcessData() { Console.WriteLine('Hello'); }"
    similarity = calculator.calc_ast_similarity(code1, code4, "csharp")
    assert similarity < 0.5  # Different structure should give lower score

    # Test mixed language confidence
    mixed_code = """
    public class DataProcessor {
        public void Process() {
            var query = "SELECT * FROM users";
            var result = db.Execute(query);
        }
    }
    """
    py_conf, sql_conf, cs_conf = calculator.calc_language_confident(mixed_code)
    assert cs_conf > 0.4  # Should detect C# structure
    assert sql_conf > 0.2  # Should detect SQL
    assert py_conf < 0.2  # Should have low Python confidence


def test_ast_calculator_csharp_edge_cases():
    from slice.eval import ASTCalculator

    calculator = ASTCalculator()

    # Test empty C# code
    assert calculator.calc_language_confident("") == (0.0, 0.0, 0.0)

    # Test with syntax errors
    bad_code = (
        "public class User { public string Name { get; set; }"  # Missing closing brace
    )
    result = calculator.calc_ast_score(bad_code, bad_code)
    assert result == 0.0 or isinstance(result, float)

    # Test with different formatting
    formatted_code1 = """
    public class User {
        public string Name { get; set; }
    }
    """
    formatted_code2 = "public class User { public string Name { get; set; } }"
    assert (
        calculator.calc_ast_similarity(formatted_code1, formatted_code2, "csharp")
        == 1.0
    )
