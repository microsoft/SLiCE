import pytest

from slice import SchemaLineage, SchemaLineageEvaluator
from slice.constants import LineageComponents


@pytest.fixture
def evaluator():
    return SchemaLineageEvaluator()


@pytest.fixture
def evaluator_with_metadata():
    return SchemaLineageEvaluator(has_metadata=True)


@pytest.fixture
def sample_lineage():
    return SchemaLineage(
        source_schema="public",
        source_table="users",
        transformation="SELECT * FROM users",
        aggregation="COUNT(*)",
        metadata="user_count",
    )


def test_validate_parameters():
    # Test valid weights
    valid_weights = {
        LineageComponents.SOURCE_TABLE: 0.4,
        LineageComponents.TRANSFORMATION: 0.4,
        LineageComponents.AGGREGATION: 0.2,
    }
    evaluator = SchemaLineageEvaluator(weights=valid_weights)
    assert evaluator.weights == valid_weights

    # Test invalid weights that don't sum to 1.0
    with pytest.raises(ValueError):
        SchemaLineageEvaluator(
            weights={
                LineageComponents.SOURCE_TABLE: 0.5,
                LineageComponents.TRANSFORMATION: 0.5,
                LineageComponents.AGGREGATION: 0.5,
            }
        )

    # Test valid weights with metadata
    valid_weights_with_metadata = {
        LineageComponents.SOURCE_TABLE: 0.4,
        LineageComponents.TRANSFORMATION: 0.4,
        LineageComponents.AGGREGATION: 0.1,
        LineageComponents.METADATA: 0.1,
    }
    evaluator = SchemaLineageEvaluator(
        weights=valid_weights_with_metadata, has_metadata=True
    )
    assert evaluator.weights == valid_weights_with_metadata


def test_get_default_weights():
    # Test without metadata
    evaluator = SchemaLineageEvaluator()
    assert evaluator.weights == {
        LineageComponents.SOURCE_TABLE: 0.4,
        LineageComponents.TRANSFORMATION: 0.4,
        LineageComponents.AGGREGATION: 0.2,
    }

    # Test with metadata
    evaluator = SchemaLineageEvaluator(has_metadata=True)
    assert evaluator.weights == {
        LineageComponents.SOURCE_TABLE: 0.3,
        LineageComponents.TRANSFORMATION: 0.3,
        LineageComponents.AGGREGATION: 0.2,
        LineageComponents.METADATA: 0.2,
    }


def test_validate_lineage():
    evaluator = SchemaLineageEvaluator()
    evaluator_with_metadata = SchemaLineageEvaluator(has_metadata=True)

    # Test valid lineage
    valid_lineage = {
        LineageComponents.SOURCE_SCHEMA: "public",
        LineageComponents.SOURCE_TABLE: "users",
        LineageComponents.TRANSFORMATION: "SELECT * FROM users",
        LineageComponents.AGGREGATION: "COUNT(*)",
    }
    assert evaluator._validate_lineage(valid_lineage) is True

    # Test valid lineage with metadata
    valid_lineage_with_metadata = {
        LineageComponents.SOURCE_SCHEMA: "public",
        LineageComponents.SOURCE_TABLE: "users",
        LineageComponents.TRANSFORMATION: "SELECT * FROM users",
        LineageComponents.AGGREGATION: "COUNT(*)",
        LineageComponents.METADATA: "user_count",
    }
    assert (
        evaluator_with_metadata._validate_lineage(valid_lineage_with_metadata) is True
    )

    # Test invalid lineage (non-string value)
    invalid_lineage = {
        LineageComponents.SOURCE_SCHEMA: "public",
        LineageComponents.SOURCE_TABLE: "users",
        LineageComponents.TRANSFORMATION: "SELECT * FROM users",
        LineageComponents.AGGREGATION: 123,  # Not a string
    }
    assert evaluator._validate_lineage(invalid_lineage) is False

    # Test invalid lineage (missing required field)
    invalid_lineage = {
        LineageComponents.SOURCE_SCHEMA: "public",
        LineageComponents.SOURCE_TABLE: "",  # Empty string
        LineageComponents.TRANSFORMATION: "SELECT * FROM users",
        LineageComponents.AGGREGATION: "COUNT(*)",
    }
    assert evaluator._validate_lineage(invalid_lineage) is True

    # Test missing metadata for evaluator that requires it
    invalid_lineage_with_metadata = {
        LineageComponents.SOURCE_SCHEMA: "public",
        LineageComponents.SOURCE_TABLE: "users",
        LineageComponents.TRANSFORMATION: "SELECT * FROM users",
        LineageComponents.AGGREGATION: "COUNT(*)",
    }
    assert (
        evaluator_with_metadata._validate_lineage(invalid_lineage_with_metadata)
        is False
    )


def test_turn_dict_to_lineage():
    evaluator = SchemaLineageEvaluator()

    # Test valid conversion
    valid_dict = {
        LineageComponents.SOURCE_SCHEMA: "public",
        LineageComponents.SOURCE_TABLE: "users",
        LineageComponents.TRANSFORMATION: "SELECT * FROM users",
        LineageComponents.AGGREGATION: "COUNT(*)",
    }

    lineage = evaluator._turn_dict_to_lineage(valid_dict)
    assert isinstance(lineage, SchemaLineage)
    assert lineage.source_schema == "public"
    assert lineage.source_table == "users"
    assert lineage.transformation == "SELECT * FROM users"
    assert lineage.aggregation == "COUNT(*)"
    assert lineage.metadata is None

    # Test with metadata
    valid_dict_with_metadata = {
        LineageComponents.SOURCE_SCHEMA: "public",
        LineageComponents.SOURCE_TABLE: "users",
        LineageComponents.TRANSFORMATION: "SELECT * FROM users",
        LineageComponents.AGGREGATION: "COUNT(*)",
        LineageComponents.METADATA: "user_count",
    }

    lineage = evaluator._turn_dict_to_lineage(valid_dict_with_metadata)
    assert isinstance(lineage, SchemaLineage)
    assert lineage.metadata == "user_count"


def test_fuzzy_match():
    from slice.SchemaLineageEvaluator import _fuzzy_match

    # Test exact matches
    assert _fuzzy_match({"user_id", "name"}, {"user_id", "name"}) == 1.0
    assert _fuzzy_match(set(), set()) == 1.0

    # Test similar strings
    assert _fuzzy_match({"user_id"}, {"userid"}) > 0.8
    assert _fuzzy_match({"first_name"}, {"firstname"}) > 0.8
    assert _fuzzy_match({"email_address"}, {"email"}) > 0.5

    # Test case sensitivity
    assert _fuzzy_match({"UserID"}, {"userid"}) == 0.5

    # Test with different lengths
    assert _fuzzy_match({"user"}, {"user_id"}) > 0.5
    assert _fuzzy_match({"user_id"}, {"user"}) > 0.5

    # Test with completely different strings
    assert _fuzzy_match({"age"}, {"old"}) < 0.3

    # Test with empty sets
    assert _fuzzy_match(set(), {"user_id"}) == 0.0
    assert _fuzzy_match({"user_id"}, set()) == 0.0

    # Test with multiple items
    pred = {"user_id", "first_name", "last_name"}
    truth = {"userid", "firstname", "lastname"}
    assert _fuzzy_match(pred, truth) > 0.8

    # Test with partial matches
    pred = {"user_id", "first_name", "email"}
    truth = {"userid", "firstname", "phone"}
    score = _fuzzy_match(pred, truth)
    assert 0.5 < score < 0.9  # Should be between perfect and poor match


def test_evaluate_source_schema():
    evaluator = SchemaLineageEvaluator()

    # Create test cases
    test_cases = [
        # Test case 1: Exact matches
        {
            "predicted": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT user_id, name FROM users",
                aggregation="COUNT(*)",
                metadata="user_count",
            ),
            "gold": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT user_id, name FROM users",
                aggregation="COUNT(*)",
                metadata="user_count",
            ),
            "expected_score": 1.0,  # Perfect match should give 1.0
        },
        # Test case 2: Similar but not exact matches
        {
            "predicted": SchemaLineage(
                source_schema="user_id, name",
                source_table="users",
                transformation="SELECT userid, firstname FROM users",
                aggregation="COUNT(*)",
                metadata="user_count",
            ),
            "gold": SchemaLineage(
                source_schema="userid, name",
                source_table="users",
                transformation="SELECT user_id, first_name FROM users",
                aggregation="COUNT(*)",
                metadata="user_count",
            ),
            "expected_score": 0.0,
        },
        # Test case 4: Empty columns
        {
            "predicted": SchemaLineage(
                source_schema="",
                source_table="users",
                transformation="SELECT email, phone FROM users",
                aggregation="COUNT(*)",
                metadata="user_count",
            ),
            "gold": SchemaLineage(
                source_schema="",
                source_table="users",
                transformation="SELECT email, phone FROM users",
                aggregation="COUNT(*)",
                metadata="user_count",
            ),
            "expected_score": 1.0,
        },
    ]

    for test_case in test_cases:
        score = evaluator.evaluate_source_schema(
            test_case["predicted"], test_case["gold"]
        )

        if "expected_score" in test_case:
            assert (
                score == test_case["expected_score"]
            ), f"Expected score {test_case['expected_score']} but got {score}"
        else:
            assert (
                test_case["expected_min"] <= score <= test_case["expected_max"]
            ), f"Expected score between {test_case['expected_min']} and {test_case['expected_max']} but got {score}"


def test_evaluate_source_tables():
    evaluator = SchemaLineageEvaluator()

    # Test cases with source tables
    test_cases = [
        # Exact match
        {
            "predicted": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT * FROM users",
                aggregation="COUNT(*)",
            ),
            "gold": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT * FROM users",
                aggregation="COUNT(*)",
            ),
            "expected_score": 1.0,
        },
        # Similar but not exact
        {
            "predicted": SchemaLineage(
                source_schema="public",
                source_table="user_table",
                transformation="SELECT * FROM users",
                aggregation="COUNT(*)",
            ),
            "gold": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT * FROM users",
                aggregation="COUNT(*)",
            ),
            "expected_min": 0.1,
            "expected_max": 0.5,  # fuzzy match
        },
        {
            "predicted": SchemaLineage(
                source_schema="public",
                source_table="""MODULE @"/shares/IDEAs.Prod.Data/Publish.Profiles.Tenant.Commercial.IDEAsTenantProfile/Resources/v4/IDEAsTenantProfileExtension_v2.module" AS IDEAsTenantProfileExtension ; IDEAsTenantProfileExtension.IDEAsTenantProfileExtensionView( Extensions = new ARRAY<string>{
"IDEAsAvailableUnits","IDEAsCloudAscentData","IDEAsDomains","IDEAsExternalEnableUsers","IDEAsExternalUsers","IDEAsFirstPaidDates","IDEAsInternal","IDEAsMSSales","IDEAsPublicSector","IDEAsSKU","IDEAsSubscription","IDEAsTenantTags","IDEAsViral","IDEAsFastTrackTenants","IDEAsCALC","IDEAsHasWorkloads"})""",
                transformation="SELECT * FROM users",
                aggregation="COUNT(*)",
            ),
            "gold": SchemaLineage(
                source_schema="public",
                source_table=""""IDEAsTenantProfileExtension_v2.module" ; IDEAsTenantProfileExtension.IDEAsTenantProfileExtensionView( Extensions = new ARRAY<string>{
"IDEAsAvailableUnits","IDEAsCloudAscentData","IDEAsDomains","IDEAsFirstPaidDates","IDEAsInternal","IDEAsMSSales","IDEAsPublicSector","IDEAsSKU","IDEAsSubscription","IDEAsTenantTags","IDEAsViral","IDEAsFastTrackTenants","IDEAsCALC","IDEAsHasWorkloads"})""",
                transformation="SELECT * FROM users",
                aggregation="COUNT(*)",
            ),
            "expected_min": 0.7,
            "expected_max": 0.9,
        },
        # Completely different
        {
            "predicted": SchemaLineage(
                source_schema="public",
                source_table="customers",
                transformation="SELECT * FROM users",
                aggregation="COUNT(*)",
            ),
            "gold": SchemaLineage(
                source_schema="public",
                source_table="apples",
                transformation="SELECT * FROM users",
                aggregation="COUNT(*)",
            ),
            "expected_min": 0.0,
            "expected_max": 0.1,  # Should be low
        },
    ]

    for test_case in test_cases:
        score = evaluator.evaluate_source_tables(
            test_case["predicted"], test_case["gold"]
        )

        if "expected_score" in test_case:
            assert (
                score == test_case["expected_score"]
            ), f"Expected score {test_case['expected_score']} but got {score}"
        else:
            assert (
                test_case["expected_min"] <= score <= test_case["expected_max"]
            ), f"Expected score between {test_case['expected_min']} and {test_case['expected_max']} but got {score}"


def test_evaluate_transformation():
    evaluator = SchemaLineageEvaluator()

    # Test cases for transformation evaluation
    test_cases = [
        # Exact match
        {
            "id": 1,
            "predicted": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT user_id, name FROM users WHERE status = 'active'",
                aggregation="COUNT(*)",
            ),
            "gold": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT user_id, name FROM users WHERE status = 'active'",
                aggregation="COUNT(*)",
            ),
            "expected_score": 1.0,
        },
        # Similar but with different formatting
        {
            "id": 2,
            "predicted": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT user_id,name FROM users WHERE status='active'",
                aggregation="COUNT(*)",
            ),
            "gold": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT user_id, name FROM users WHERE status = 'active'",
                aggregation="COUNT(*)",
            ),
            "expected_score": 1.0,
        },
        # Different but with same structure
        {
            "id": 3,
            "predicted": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT email, phone FROM users WHERE status = 'inactive'",
                aggregation="COUNT(*)",
            ),
            "gold": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT user_id, name FROM users WHERE status = 'active'",
                aggregation="COUNT(*)",
            ),
            "expected_min": 0.6,
            "expected_max": 0.8,
        },
        # Completely different
        {
            "id": 4,
            "predicted": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="INSERT INTO logs VALUES ('user_checked')",
                aggregation="COUNT(*)",
            ),
            "gold": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT user_id, name FROM users WHERE status = 'active'",
                aggregation="COUNT(*)",
            ),
            "expected_min": 0.0,
            "expected_max": 0.4,  # Should be low
        },
    ]

    for test_case in test_cases:
        score = evaluator.evaluate_transformation(
            test_case["predicted"], test_case["gold"]
        )

        if "expected_score" in test_case:
            assert (
                score == test_case["expected_score"]
            ), f"ID: {test_case['id']} - Expected score {test_case['expected_score']} but got {score}"
        else:
            assert (
                test_case["expected_min"] <= score <= test_case["expected_max"]
            ), f"ID: {test_case['id']} - Expected score between {test_case['expected_min']} and {test_case['expected_max']} but got {score}"


def test_evaluate_aggregation():
    evaluator = SchemaLineageEvaluator()

    # Test cases for aggregation evaluation
    test_cases = [
        # Exact match
        {
            "id": 1,
            "predicted": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT * FROM users",
                aggregation="COUNT(*) AS total_users",
            ),
            "gold": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT * FROM users",
                aggregation="COUNT(*) AS total_users",
            ),
            "expected_score": 1.0,
        },
        # Similar aggregations
        {
            "id": 2,
            "predicted": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT * FROM users",
                aggregation="COUNT(user_id) AS total_users",
            ),
            "gold": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT * FROM users",
                aggregation="COUNT(*) AS total_users",
            ),
            "expected_min": 0.4,
            "expected_max": 0.6,
        },
        # Different aggregation functions
        {
            "id": 3,
            "predicted": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT * FROM users",
                aggregation="SUM(age) AS total_age",
            ),
            "gold": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT * FROM users",
                aggregation="COUNT(*) AS total_users",
            ),
            "expected_min": 0.2,
            "expected_max": 0.6,  # Moderate difference
        },
    ]

    for test_case in test_cases:
        score = evaluator.evaluate_aggregation(
            test_case["predicted"], test_case["gold"]
        )

        if "expected_score" in test_case:
            assert (
                score == test_case["expected_score"]
            ), f"ID: {test_case['id']} - Expected score {test_case['expected_score']} but got {score}"
        else:
            assert (
                test_case["expected_min"] <= score <= test_case["expected_max"]
            ), f"ID: {test_case['id']} - Expected score between {test_case['expected_min']} and {test_case['expected_max']} but got {score}"


def test_evaluate_metadata(evaluator_with_metadata):
    # Test cases for metadata evaluation
    test_cases = [
        # Exact match
        {
            "predicted": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT * FROM users",
                aggregation="COUNT(*)",
                metadata="where status = 'active'",
            ),
            "gold": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT * FROM users",
                aggregation="COUNT(*)",
                metadata="where status = 'active'",
            ),
            "expected_min": 0.95,
            "expected_max": 1.0,  # Perfect match should give close to 1.0
        },
        # Similar metadata
        {
            "predicted": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT * FROM users",
                aggregation="COUNT(*)",
                metadata="where status = 'active' and age > 18",
            ),
            "gold": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT * FROM users",
                aggregation="COUNT(*)",
                metadata="where status = 'active' and age > 28",
            ),
            "expected_min": 0.7,
            "expected_max": 0.95,  # Should be high but not perfect
        },
        # Different metadata
        {
            "predicted": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT * FROM users",
                aggregation="COUNT(*)",
                metadata="order by age",
            ),
            "gold": SchemaLineage(
                source_schema="public",
                source_table="users",
                transformation="SELECT * FROM users",
                aggregation="COUNT(*)",
                metadata="where status = 'active'",
            ),
            "expected_min": 0.0,
            "expected_max": 0.2,  # Should be low
        },
    ]

    for test_case in test_cases:
        score = evaluator_with_metadata.evaluate_metadata(
            test_case["predicted"], test_case["gold"]
        )

        assert (
            test_case["expected_min"] <= score <= test_case["expected_max"]
        ), f"Expected score between {test_case['expected_min']} and {test_case['expected_max']} but got {score}"


def test_evaluate_wrong_schema_name():
    evaluator = SchemaLineageEvaluator()
    evaluator_with_metadata = SchemaLineageEvaluator(has_metadata=True)

    predicted = {
        LineageComponents.SOURCE_SCHEMA: "user_id, name, email",
        LineageComponents.SOURCE_TABLE: "users",
        LineageComponents.TRANSFORMATION: "SELECT * FROM users",
        LineageComponents.AGGREGATION: "COUNT(*)",
    }

    gold = {
        LineageComponents.SOURCE_SCHEMA: "userid, name, email",
        LineageComponents.SOURCE_TABLE: "users",
        LineageComponents.TRANSFORMATION: "SELECT * FROM users",
        LineageComponents.AGGREGATION: "COUNT(*)",
    }

    result = evaluator.evaluate(predicted, gold)
    assert result[LineageComponents.SOURCE_SCHEMA] == 0
    assert result[LineageComponents.SOURCE_TABLE] == 1
    assert result[LineageComponents.TRANSFORMATION] == 1
    assert result[LineageComponents.AGGREGATION] == 1
    assert result["format"] == 1
    assert result["overall"] == 0

    # Test with metadata but missing metadata field
    predicted_without_metadata = {
        LineageComponents.SOURCE_SCHEMA: "user_id, name, email",
        LineageComponents.SOURCE_TABLE: "users",
        LineageComponents.TRANSFORMATION: "SELECT * FROM users",
        LineageComponents.AGGREGATION: "COUNT(*)",
    }

    gold_without_metadata = {
        LineageComponents.SOURCE_SCHEMA: "userid, name, email",
        LineageComponents.SOURCE_TABLE: "users",
        LineageComponents.TRANSFORMATION: "SELECT * FROM users",
        LineageComponents.AGGREGATION: "COUNT(*)",
    }

    # Error because metadata is required but not present
    with pytest.raises(ValueError):
        evaluator_with_metadata.evaluate(
            predicted_without_metadata, gold_without_metadata
        )


def test_evaluate():
    evaluator = SchemaLineageEvaluator()
    evaluator_with_metadata = SchemaLineageEvaluator(has_metadata=True)

    # Test case for evaluate without metadata
    predicted = {
        LineageComponents.SOURCE_SCHEMA: "user_id, name, email",
        LineageComponents.SOURCE_TABLE: "users",
        LineageComponents.TRANSFORMATION: "SELECT user_id, name, email FROM users WHERE status = 'active'",
        LineageComponents.AGGREGATION: "COUNT(*) AS active_users",
    }

    gold = {
        LineageComponents.SOURCE_SCHEMA: "user_id, name, email",
        LineageComponents.SOURCE_TABLE: "users",
        LineageComponents.TRANSFORMATION: "SELECT user_id, name, email FROM users WHERE status = 'active'",
        LineageComponents.AGGREGATION: "COUNT(*) AS active_users",
    }

    result = evaluator.evaluate(predicted, gold)

    # Check that all components are present
    assert "overall" in result
    assert LineageComponents.SOURCE_SCHEMA in result
    assert LineageComponents.SOURCE_TABLE in result
    assert LineageComponents.TRANSFORMATION in result
    assert LineageComponents.AGGREGATION in result
    assert "format" in result

    # Check that overall score is a weighted average of component scores
    expected_overall = (
        evaluator.weights[LineageComponents.SOURCE_TABLE]
        * result[LineageComponents.SOURCE_TABLE]
        + evaluator.weights[LineageComponents.TRANSFORMATION]
        * result[LineageComponents.TRANSFORMATION]
        + evaluator.weights[LineageComponents.AGGREGATION]
        * result[LineageComponents.AGGREGATION]
    )
    assert abs(result["overall"] - expected_overall) < 0.01

    # Test case for evaluate with metadata
    predicted_with_metadata = {
        LineageComponents.SOURCE_SCHEMA: "user_id, name, email",
        LineageComponents.SOURCE_TABLE: "users",
        LineageComponents.TRANSFORMATION: "SELECT user_id, name, email FROM users WHERE status = 'active'",
        LineageComponents.AGGREGATION: "COUNT(*) AS active_users",
        LineageComponents.METADATA: "where status = 'active'",
    }

    gold_with_metadata = {
        LineageComponents.SOURCE_SCHEMA: "user_id, name, email",
        LineageComponents.SOURCE_TABLE: "users",
        LineageComponents.TRANSFORMATION: "SELECT user_id, name, email FROM users WHERE status = 'active'",
        LineageComponents.AGGREGATION: "COUNT(*) AS active_users",
        LineageComponents.METADATA: "where status = 'active'",
    }

    result_with_metadata = evaluator_with_metadata.evaluate(
        predicted_with_metadata, gold_with_metadata
    )

    # Check that all components including metadata are present
    assert LineageComponents.METADATA in result_with_metadata

    # Check that overall score includes metadata component
    expected_overall_with_metadata = (
        evaluator_with_metadata.weights[LineageComponents.SOURCE_TABLE]
        * result_with_metadata[LineageComponents.SOURCE_TABLE]
        + evaluator_with_metadata.weights[LineageComponents.TRANSFORMATION]
        * result_with_metadata[LineageComponents.TRANSFORMATION]
        + evaluator_with_metadata.weights[LineageComponents.AGGREGATION]
        * result_with_metadata[LineageComponents.AGGREGATION]
        + evaluator_with_metadata.weights[LineageComponents.METADATA]
        * result_with_metadata[LineageComponents.METADATA]
    )
    assert abs(result_with_metadata["overall"] - expected_overall_with_metadata) < 0.01


def test_evaluate_batch():
    evaluator = SchemaLineageEvaluator()

    # Create a batch of test cases
    predicted_batch = [
        {
            LineageComponents.SOURCE_SCHEMA: "user_id, name",
            LineageComponents.SOURCE_TABLE: "users",
            LineageComponents.TRANSFORMATION: "SELECT user_id, name FROM users",
            LineageComponents.AGGREGATION: "COUNT(*)",
        },
        {
            LineageComponents.SOURCE_SCHEMA: "product_id, price",
            LineageComponents.SOURCE_TABLE: "products",
            LineageComponents.TRANSFORMATION: "SELECT product_id, price FROM products",
            LineageComponents.AGGREGATION: "AVG(price)",
        },
    ]

    gold_batch = [
        {
            LineageComponents.SOURCE_SCHEMA: "user_id, name",
            LineageComponents.SOURCE_TABLE: "users",
            LineageComponents.TRANSFORMATION: "SELECT user_id, name FROM users",
            LineageComponents.AGGREGATION: "COUNT(*)",
        },
        {
            LineageComponents.SOURCE_SCHEMA: "product_id, price",
            LineageComponents.SOURCE_TABLE: "products",
            LineageComponents.TRANSFORMATION: "SELECT product_id, price FROM products",
            LineageComponents.AGGREGATION: "AVG(price)",
        },
    ]

    # Test batch evaluation
    results = evaluator.evaluate_batch(predicted_batch, gold_batch)

    # Check that we get the correct number of results
    assert len(results) == len(predicted_batch)

    # Check that each result has the expected structure
    for result in results:
        assert "overall" in result
        assert LineageComponents.SOURCE_SCHEMA in result
        assert LineageComponents.SOURCE_TABLE in result
        assert LineageComponents.TRANSFORMATION in result
        assert LineageComponents.AGGREGATION in result
        assert "format" in result

    # Test with mismatched batch sizes
    with pytest.raises(ValueError):
        evaluator.evaluate_batch(predicted_batch, gold_batch[:1])


def test_metadata_check_before_conversion():
    """Test that the metadata check happens after converting to SchemaLineage object."""
    evaluator_with_metadata = SchemaLineageEvaluator(has_metadata=True)

    # Create a valid prediction and gold with metadata
    predicted = {
        LineageComponents.SOURCE_SCHEMA: "user_id, name",
        LineageComponents.SOURCE_TABLE: "users",
        LineageComponents.TRANSFORMATION: "SELECT user_id, name FROM users",
        LineageComponents.AGGREGATION: "COUNT(*)",
        LineageComponents.METADATA: "where status = 'active'",
    }

    gold = {
        LineageComponents.SOURCE_SCHEMA: "user_id, name",
        LineageComponents.SOURCE_TABLE: "users",
        LineageComponents.TRANSFORMATION: "SELECT user_id, name FROM users",
        LineageComponents.AGGREGATION: "COUNT(*)",
        LineageComponents.METADATA: "where status = 'active'",
    }

    # This should work without errors
    result = evaluator_with_metadata.evaluate(predicted, gold)
    assert LineageComponents.METADATA in result

    # Test with missing metadata field
    predicted_without_metadata = {
        LineageComponents.SOURCE_SCHEMA: "user_id, name",
        LineageComponents.SOURCE_TABLE: "users",
        LineageComponents.TRANSFORMATION: "SELECT user_id, name FROM users",
        LineageComponents.AGGREGATION: "COUNT(*)",
    }

    gold_without_metadata = {
        LineageComponents.SOURCE_SCHEMA: "user_id, name",
        LineageComponents.SOURCE_TABLE: "users",
        LineageComponents.TRANSFORMATION: "SELECT user_id, name FROM users",
        LineageComponents.AGGREGATION: "COUNT(*)",
    }

    # This should raise a ValueError
    with pytest.raises(ValueError):
        evaluator_with_metadata.evaluate(
            predicted_without_metadata, gold_without_metadata
        )


def test_normalize_quotes():
    """Test the normalize_quotes function directly."""
    from slice.eval import normalize_quotes

    # Test basic quote normalization
    assert (
        normalize_quotes('SELECT "column" FROM table') == "SELECT 'column' FROM table"
    )
    assert (
        normalize_quotes("SELECT 'column' FROM table") == "SELECT 'column' FROM table"
    )

    # Test multiple quotes
    assert (
        normalize_quotes('SELECT "col1", "col2" FROM "table"')
        == "SELECT 'col1', 'col2' FROM 'table'"
    )

    # Test mixed quotes
    assert (
        normalize_quotes("SELECT \"col1\", 'col2' FROM table")
        == "SELECT 'col1', 'col2' FROM table"
    )

    # Test empty and None cases
    assert normalize_quotes("") == ""
    assert normalize_quotes(None) == ""

    # Test SQL PIVOT example from your case
    sql1 = 'PIVOT(MAX(AllUp) FOR Application IN ("PowerPoint" AS PowerPointWAU))'
    sql2 = "PIVOT(MAX(AllUp) FOR Application IN ('PowerPoint' AS PowerPointWAU))"
    expected = "PIVOT(MAX(AllUp) FOR Application IN ('PowerPoint' AS PowerPointWAU))"

    assert normalize_quotes(sql1) == expected
    assert normalize_quotes(sql2) == expected


def test_evaluate_transformation_with_quote_normalization():
    """Test that transformation evaluation works correctly with quote normalization."""
    evaluator = SchemaLineageEvaluator()

    # Test case where predicted and gold have same SQL but different quote styles
    predicted = SchemaLineage(
        source_schema="public",
        source_table="users",
        transformation='PIVOT(MAX(AllUp) FOR Application IN ("PowerPoint" AS PowerPointWAU))',
        aggregation="COUNT(*)",
    )

    gold = SchemaLineage(
        source_schema="public",
        source_table="users",
        transformation="PIVOT(MAX(AllUp) FOR Application IN ('PowerPoint' AS PowerPointWAU))",
        aggregation="COUNT(*)",
    )

    # With quote normalization, these should have a very high score (close to 1.0)
    score = evaluator.evaluate_transformation(predicted, gold)
    assert score == 1.0, f"Expected score == 1.0 but got {score}"

    # Test another case with multiple quoted elements
    predicted2 = SchemaLineage(
        source_schema="public",
        source_table="sales",
        transformation='SELECT "product_name", "category" FROM "products" WHERE "status" = "active"',
        aggregation="COUNT(*)",
    )

    gold2 = SchemaLineage(
        source_schema="public",
        source_table="sales",
        transformation="SELECT 'product_name', 'category' FROM 'products' WHERE 'status' = 'active'",
        aggregation="COUNT(*)",
    )

    score2 = evaluator.evaluate_transformation(predicted2, gold2)
    assert score2 == 1.0, f"Expected score == 1.0 but got {score2}"
