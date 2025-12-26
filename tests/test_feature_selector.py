"""Tests for active_matcher.feature_selector module.

This module defines the FeatureSelector orchestration logic.
"""
import pandas as pd
import pytest

from active_matcher.feature_selector import FeatureSelector
from active_matcher.feature import (
    ExactMatchFeature,
    RelDiffFeature,
    TFIDFFeature,
    JaccardFeature,
    SIFFeature,
    OverlapCoeffFeature,
    CosineFeature,
    MongeElkanFeature,
    EditDistanceFeature,
    SmithWatermanFeature,
)
from active_matcher.tokenizer import AlphaNumericTokenizer


class TestFeatureSelector:
    """Tests for FeatureSelector selection routines."""

    def test_init_default(self):
        """Test FeatureSelector initialization with default parameters."""
        selector = FeatureSelector()
        assert selector._extra_features is False
        assert len(selector._tokenizers) == len(FeatureSelector.TOKENIZERS)
        assert len(selector._token_features) == len(
            FeatureSelector.TOKEN_FEATURES
        )
        assert selector.projected_columns_ is None

    def test_init_extra_features_false(self):
        """Test FeatureSelector init with extra_features=False."""
        selector = FeatureSelector(extra_features=False)
        assert selector._extra_features is False
        assert len(selector._tokenizers) == len(FeatureSelector.TOKENIZERS)
        assert len(selector._token_features) == len(
            FeatureSelector.TOKEN_FEATURES
        )

    def test_init_extra_features_true(self):
        """Test FeatureSelector init with extra_features=True."""
        selector = FeatureSelector(extra_features=True)
        assert selector._extra_features is True
        assert len(selector._tokenizers) == (
            len(FeatureSelector.TOKENIZERS) +
            len(FeatureSelector.EXTRA_TOKENIZERS)
        )
        assert len(selector._token_features) == (
            len(FeatureSelector.TOKEN_FEATURES) +
            len(FeatureSelector.EXTRA_TOKEN_FEATURES)
        )

    def test_init_type_error(self):
        """Test FeatureSelector raises TypeError for invalid extra_features."""
        with pytest.raises(TypeError):
            FeatureSelector(extra_features="not a bool")
        with pytest.raises(TypeError):
            FeatureSelector(extra_features=1)

    def test_drop_nulls(self, spark_session):
        """Test _drop_nulls method with different thresholds."""
        from pyspark.sql.types import StructType, StructField, StringType
        schema = StructType([
            StructField("col1", StringType(), True),
            StructField("col2", StringType(), True),
            StructField("col3", StringType(), True),
        ])
        df = spark_session.createDataFrame([
            {"col1": "a", "col2": "b", "col3": None},
            {"col1": "b", "col2": None, "col3": None},
            {"col1": "c", "col2": "d", "col3": None},
        ], schema=schema)

        selector = FeatureSelector()
        result = selector._drop_nulls(df, threshold=0.5)
        assert "col1" in result.columns
        assert "col2" in result.columns
        assert "col3" not in result.columns

    def test_drop_nulls_strict_threshold(self, spark_session):
        """Test _drop_nulls with strict threshold."""
        from pyspark.sql.types import StructType, StructField, StringType
        schema = StructType([
            StructField("col1", StringType(), True),
            StructField("col2", StringType(), True),
        ])
        df = spark_session.createDataFrame([
            {"col1": "a", "col2": None},
            {"col1": "b", "col2": None},
            {"col1": "c", "col2": "d"},
        ], schema=schema)

        selector = FeatureSelector()
        result = selector._drop_nulls(df, threshold=0.5)
        assert "col1" in result.columns
        assert "col2" not in result.columns

    def test_drop_nulls_with_nan(self, spark_session):
        """Test _drop_nulls handles NaN values."""
        from pyspark.sql.types import StructType, StructField, DoubleType
        schema = StructType([
            StructField("col1", DoubleType(), True),
            StructField("col2", DoubleType(), True),
        ])
        df = spark_session.createDataFrame([
            {"col1": 1.0, "col2": float('nan')},
            {"col1": 2.0, "col2": float('nan')},
            {"col1": 3.0, "col2": 4.0},
        ], schema=schema)

        selector = FeatureSelector()
        result = selector._drop_nulls(df, threshold=0.5)
        assert "col1" in result.columns
        assert "col2" not in result.columns

    def test_select_features_a_only(self, spark_session):
        """Test select_features with table A only."""
        a_df = spark_session.createDataFrame([
            {"name": "hello world", "age": 25},
            {"name": "test code", "age": 30},
        ])

        selector = FeatureSelector(extra_features=False)
        features = selector.select_features(a_df, None)

        assert len(features) > 0
        assert selector.projected_columns_ is not None
        assert "name" in selector.projected_columns_
        assert "age" in selector.projected_columns_

        exact_match_features = [
            f for f in features if isinstance(f, ExactMatchFeature)
        ]
        assert len(exact_match_features) == len(
            selector.projected_columns_
        )

    def test_select_features_a_and_b(self, spark_session):
        """Test select_features with both tables A and B."""
        a_df = spark_session.createDataFrame([
            {"name": "hello world", "age": 25},
        ])
        b_df = spark_session.createDataFrame([
            {"name": "test code", "age": 30},
        ])

        selector = FeatureSelector(extra_features=False)
        features = selector.select_features(a_df, b_df)

        assert len(features) > 0
        assert selector.projected_columns_ is not None

    def test_select_features_numeric_features(self, spark_session):
        """Test select_features generates RelDiffFeature for numeric."""
        a_df = spark_session.createDataFrame([
            {"age": 25, "score": 10.5},
            {"age": 30, "score": 20.5},
        ])

        selector = FeatureSelector(extra_features=False)
        features = selector.select_features(a_df, None)

        rel_diff_features = [
            f for f in features if isinstance(f, RelDiffFeature)
        ]
        assert len(rel_diff_features) >= 2

    def test_select_features_null_threshold_default(self, spark_session):
        """Test select_features with default null_threshold."""
        from pyspark.sql.types import StructType, StructField, StringType
        schema = StructType([
            StructField("col1", StringType(), True),
            StructField("col2", StringType(), True),
        ])
        a_df = spark_session.createDataFrame([
            {"col1": "a", "col2": None},
            {"col1": "b", "col2": None},
            {"col1": "c", "col2": None},
        ], schema=schema)

        selector = FeatureSelector()
        selector.select_features(a_df, None, null_threshold=0.5)

        assert "col1" in selector.projected_columns_
        assert "col2" not in selector.projected_columns_

    def test_select_features_null_threshold_strict(self, spark_session):
        """Test select_features with strict null_threshold."""
        a_df = spark_session.createDataFrame([
            {"col1": "a", "col2": None},
            {"col1": "b", "col2": "x"},
            {"col1": "c", "col2": "y"},
        ])

        selector = FeatureSelector()
        selector.select_features(a_df, None, null_threshold=0.3)

        assert "col1" in selector.projected_columns_
        assert "col2" not in selector.projected_columns_

    def test_select_features_null_threshold_lenient(self, spark_session):
        """Test select_features with lenient null_threshold."""
        a_df = spark_session.createDataFrame([
            {"col1": "a", "col2": None},
            {"col1": "b", "col2": None},
            {"col1": "c", "col2": "x"},
        ])

        selector = FeatureSelector()
        selector.select_features(a_df, None, null_threshold=0.8)

        assert "col1" in selector.projected_columns_
        assert "col2" in selector.projected_columns_

    def test_select_features_extra_features_false(self, spark_session):
        """Test select_features without extra features."""
        a_df = spark_session.createDataFrame([
            {"name": "hello world python code test"},
            {"name": "test code python"},
        ])

        selector = FeatureSelector(extra_features=False)
        features = selector.select_features(a_df, None)

        assert len(features) > 0
        token_features = [
            f for f in features
            if isinstance(f, (TFIDFFeature, JaccardFeature, SIFFeature,
                              OverlapCoeffFeature, CosineFeature))
        ]
        assert len(token_features) > 0

    def test_select_features_extra_features_true(self, spark_session):
        """Test select_features with extra features enabled."""
        a_df = spark_session.createDataFrame([
            {"name": "hello world python code test"},
            {"name": "test code python"},
        ])

        selector = FeatureSelector(extra_features=True)
        features = selector.select_features(a_df, None)

        assert len(features) > 0
        extra_tokenizer_features = [
            f for f in features
            if any(
                isinstance(f, (TFIDFFeature, JaccardFeature, SIFFeature,
                               OverlapCoeffFeature, CosineFeature)) and
                str(AlphaNumericTokenizer()) in str(f._tokenizer)
                for _ in [None]
            )
        ]
        assert len(extra_tokenizer_features) > 0

    def test_select_features_token_features_avg_count(self, spark_session):
        """Test select_features generates token features based on avg_count."""
        a_df = spark_session.createDataFrame([
            {"name": "hello world python code test"},
            {"name": "test code python"},
            {"name": "python code"},
        ])

        selector = FeatureSelector(extra_features=False)
        features = selector.select_features(a_df, None)

        token_features = [
            f for f in features
            if isinstance(f, (TFIDFFeature, JaccardFeature, SIFFeature,
                              OverlapCoeffFeature, CosineFeature))
        ]
        assert len(token_features) > 0

    def test_select_features_sequence_features(self, spark_session):
        """Test select_features generates sequence features for alnum."""
        a_df = spark_session.createDataFrame([
            {"name": "hello123"},
            {"name": "world456"},
            {"name": "test789"},
        ])

        selector = FeatureSelector(extra_features=True)
        features = selector.select_features(a_df, None)

        sequence_features = [
            f for f in features
            if isinstance(f, (MongeElkanFeature, EditDistanceFeature,
                              SmithWatermanFeature))
        ]
        assert len(sequence_features) > 0

    def test_select_features_no_sequence_features_low_avg(
        self, spark_session
    ):
        """Test sequence features not generated when avg_count > 10."""
        a_df = spark_session.createDataFrame([
            {"name": " ".join(["word"] * 20)},
            {"name": " ".join(["word"] * 20)},
        ])

        selector = FeatureSelector(extra_features=True)
        features = selector.select_features(a_df, None)

        sequence_features = [
            f for f in features
            if isinstance(f, (MongeElkanFeature, EditDistanceFeature,
                              SmithWatermanFeature))
        ]
        assert len(sequence_features) == 0

    def test_tokenize_and_count(self, spark_session):
        """Test _tokenize_and_count static method."""
        from active_matcher.tokenizer import StrippedWhiteSpaceTokenizer

        tokenizer = StrippedWhiteSpaceTokenizer()
        df = pd.DataFrame({
            'col1': ['hello world', 'test code'],
        })

        token_col_map = {
            'tokens': (tokenizer, 'col1')
        }

        result = list(FeatureSelector._tokenize_and_count([df], token_col_map))
        assert len(result) == 1
        result_df = result[0]
        assert 'tokens' in result_df.columns
        assert len(result_df) == 2
        assert all(
            isinstance(v, (int, type(None))) for v in result_df['tokens']
        )

    def test_select_features_all_string_columns(self, spark_session):
        """Test select_features with all string columns."""
        a_df = spark_session.createDataFrame([
            {"name": "hello", "address": "world"},
            {"name": "test", "address": "code"},
        ])

        selector = FeatureSelector(extra_features=False)
        features = selector.select_features(a_df, None)

        assert len(features) > 0
        exact_match_features = [
            f for f in features if isinstance(f, ExactMatchFeature)
        ]
        assert len(exact_match_features) == 2

    def test_select_features_mixed_types(self, spark_session):
        """Test select_features with mixed column types."""
        a_df = spark_session.createDataFrame([
            {"name": "hello", "age": 25, "score": 10.5},
            {"name": "world", "age": 30, "score": 20.5},
        ])

        selector = FeatureSelector(extra_features=False)
        features = selector.select_features(a_df, None)

        exact_match = [f for f in features if isinstance(f, ExactMatchFeature)]
        rel_diff = [f for f in features if isinstance(f, RelDiffFeature)]

        assert len(exact_match) > 0
        assert len(rel_diff) >= 2

    def test_select_features_minimal_dataframe(self, spark_session):
        """Test select_features with minimal DataFrame (single row)."""
        from pyspark.sql.types import (
            StructType, StructField, StringType, IntegerType
        )
        schema = StructType([
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
        ])
        a_df = spark_session.createDataFrame(
            [{"name": "test", "age": 25}], schema=schema
        )

        selector = FeatureSelector()
        features = selector.select_features(a_df, None)

        assert isinstance(features, list)
        assert len(features) >= 0
