"""Tests for active_matcher.feature.vector_feature module.

This module provides document frequency builders and TFIDF/SIF features.
"""
import pandas as pd
import numpy as np

from active_matcher.feature.vector_feature import (
    DocFreqBuilder,
    TFIDFFeature,
    SIFFeature,
)
from active_matcher.tokenizer.vectorizer import SIFVectorizer
from active_matcher.fv_generator import BuildCache


class TestDocFreqBuilder:
    """Tests for DocFreqBuilder."""

    def test_init(self, tokenizer):
        """Test DocFreqBuilder initialization."""
        builder = DocFreqBuilder('a_attr', 'b_attr', tokenizer)
        assert builder.a_attr == 'a_attr'
        assert builder.b_attr == 'b_attr'
        assert builder.tokenizer == tokenizer
        assert builder.doc_freqs_ is None
        assert builder.hashes_ is None
        assert builder.hash_func_ is None
        assert builder.corpus_size_ is None
        assert builder._built is False

    def test_eq(self, tokenizer):
        """Test DocFreqBuilder equality."""
        builder1 = DocFreqBuilder('a_attr', 'b_attr', tokenizer)
        builder2 = DocFreqBuilder('a_attr', 'b_attr', tokenizer)
        builder3 = DocFreqBuilder('a_attr', 'c_attr', tokenizer)

        assert builder1 == builder2
        assert builder1 != builder3

    def test_build_a_only(self, spark_session, tokenizer):
        """Test DocFreqBuilder with table A only."""
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
            {"_id": 11, "a_attr": "world python"},
            {"_id": 12, "a_attr": "python code"},
        ])

        builder = DocFreqBuilder('a_attr', 'b_attr', tokenizer)
        builder.build(a_df, None)

        assert builder._built is True
        assert builder.corpus_size_ == 3
        assert builder.doc_freqs_ is not None
        assert builder.hashes_ is not None
        assert builder.hash_func_ is not None
        assert len(builder.doc_freqs_) > 0
        assert len(builder.hashes_) == len(builder.doc_freqs_)

    def test_build_a_and_b(self, spark_session, tokenizer):
        """Test DocFreqBuilder with both tables A and B."""
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
            {"_id": 11, "a_attr": "world python"},
        ])
        b_df = spark_session.createDataFrame([
            {"_id": 20, "b_attr": "python code"},
            {"_id": 21, "b_attr": "code test"},
        ])

        builder = DocFreqBuilder('a_attr', 'b_attr', tokenizer)
        builder.build(a_df, b_df)

        assert builder._built is True
        assert builder.corpus_size_ == 4
        assert builder.doc_freqs_ is not None
        assert len(builder.doc_freqs_) > 0

    def test_build_idempotent(self, spark_session, tokenizer):
        """Test that build is idempotent (can be called twice)."""
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
        ])

        builder = DocFreqBuilder('a_attr', 'b_attr', tokenizer)
        builder.build(a_df, None)

        initial_doc_freqs = builder.doc_freqs_.copy()
        initial_hashes = builder.hashes_.copy()

        builder.build(a_df, None)

        assert builder._built is True
        assert np.array_equal(builder.doc_freqs_, initial_doc_freqs)
        assert np.array_equal(builder.hashes_, initial_hashes)


class TestTFIDFFeature:
    """Tests for TFIDFFeature."""

    def test_init(self, tokenizer):
        """Test TFIDFFeature initialization."""
        feature = TFIDFFeature('a_attr', 'b_attr', tokenizer)
        assert feature.a_attr == 'a_attr'
        assert feature.b_attr == 'b_attr'
        assert feature._tokenizer == tokenizer
        assert feature.vectorizer is not None
        assert feature._a_vec_col is not None
        assert feature._b_vec_col is not None

    def test_str(self, tokenizer):
        """Test TFIDFFeature string representation."""
        feature = TFIDFFeature('a_attr', 'b_attr', tokenizer)
        expected = f'tf_idf_{str(tokenizer)}(a_attr, b_attr)'
        assert str(feature) == expected

    def test_get_vector_column(self, tokenizer):
        """Test _get_vector_column method."""
        feature = TFIDFFeature('a_attr', 'b_attr', tokenizer)
        token_col = feature._get_token_column('a_attr')
        vec_col = feature._get_vector_column('a_attr')
        assert vec_col == feature.vectorizer.out_col_name(token_col)

    def test_preprocess_output_column(self, tokenizer):
        """Test _preprocess_output_column method."""
        feature = TFIDFFeature('a_attr', 'b_attr', tokenizer)
        out_col = feature._preprocess_output_column('a_attr')
        assert out_col == feature._get_vector_column('a_attr')

    def test_build(self, spark_session, tokenizer):
        """Test TFIDFFeature build method."""
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
            {"_id": 11, "a_attr": "world python"},
        ])
        b_df = spark_session.createDataFrame([
            {"_id": 20, "b_attr": "python code"},
        ])

        cache = BuildCache()
        feature = TFIDFFeature('a_attr', 'b_attr', tokenizer)
        feature.build(a_df, b_df, cache)

        assert feature.vectorizer._N is not None
        assert feature.vectorizer._hash_func is not None
        assert feature.vectorizer._hashes is not None
        assert feature.vectorizer._idfs is not None

    def test_preprocess(self, spark_session, tokenizer):
        """Test TFIDFFeature preprocessing."""
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
            {"_id": 11, "a_attr": "world python"},
        ])
        b_df = spark_session.createDataFrame([
            {"_id": 20, "b_attr": "python code"},
        ])

        cache = BuildCache()
        feature = TFIDFFeature('a_attr', 'b_attr', tokenizer)
        feature.build(a_df, b_df, cache)

        data = pd.DataFrame({
            'a_attr': ['hello world', 'python code'],
        })
        result = feature._preprocess(data, 'a_attr')

        assert isinstance(result, pd.Series)
        assert result.name == feature._get_vector_column('a_attr')
        assert len(result) == 2
        assert all(
            hasattr(v, 'dot') or v is None
            for v in result.values
        )

    def test_call(self, spark_session, tokenizer):
        """Test TFIDFFeature __call__ method."""
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
            {"_id": 11, "a_attr": "world python"},
        ])
        b_df = spark_session.createDataFrame([
            {"_id": 20, "b_attr": "python code"},
        ])

        cache = BuildCache()
        feature = TFIDFFeature('a_attr', 'b_attr', tokenizer)
        feature.build(a_df, b_df, cache)

        a_pdf = pd.DataFrame({
            'a_attr': ['hello world', 'world python'],
        })
        b_pdf = pd.DataFrame({
            'b_attr': ['python code'],
        })

        a_pdf = feature.preprocess(a_pdf, True)
        b_pdf = feature.preprocess(b_pdf, False)

        result = feature(b_pdf.iloc[0], a_pdf)

        assert isinstance(result, pd.Series)
        assert len(result) == 2
        assert all(
            isinstance(v, (float, np.floating)) or np.isnan(v)
            for v in result.values
        )

    def test_call_with_none_vector(self, spark_session, tokenizer):
        """Test __call__ when b vector is None."""
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
        ])
        b_df = spark_session.createDataFrame([
            {"_id": 20, "b_attr": "python code"},
        ])

        cache = BuildCache()
        feature = TFIDFFeature('a_attr', 'b_attr', tokenizer)
        feature.build(a_df, b_df, cache)

        a_pdf = pd.DataFrame({
            'a_attr': ['hello world'],
        })
        a_pdf = feature.preprocess(a_pdf, True)

        rec = pd.Series({feature._b_vec_col: None})

        result = feature(rec, a_pdf)
        assert isinstance(result, pd.Series)
        assert len(result) == 1
        assert np.isnan(result.iloc[0])


class TestSIFFeature:
    """Tests for SIFFeature."""

    def test_init(self, tokenizer):
        """Test SIFFeature initialization."""
        feature = SIFFeature('a_attr', 'b_attr', tokenizer)
        assert feature.a_attr == 'a_attr'
        assert feature.b_attr == 'b_attr'
        assert feature._tokenizer == tokenizer
        assert isinstance(feature.vectorizer, type(feature.vectorizer))
        assert isinstance(feature.vectorizer, SIFVectorizer)

    def test_str(self, tokenizer):
        """Test SIFFeature string representation."""
        feature = SIFFeature('a_attr', 'b_attr', tokenizer)
        expected = f'sif_{str(tokenizer)}(a_attr, b_attr)'
        assert str(feature) == expected

    def test_build(self, spark_session, tokenizer):
        """Test SIFFeature build method."""
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
            {"_id": 11, "a_attr": "world python"},
        ])
        b_df = spark_session.createDataFrame([
            {"_id": 20, "b_attr": "python code"},
        ])

        cache = BuildCache()
        feature = SIFFeature('a_attr', 'b_attr', tokenizer)
        feature.build(a_df, b_df, cache)

        assert feature.vectorizer._N is not None
        assert feature.vectorizer._hash_func is not None
        assert feature.vectorizer._hashes is not None
        assert feature.vectorizer._sifs is not None

    def test_preprocess(self, spark_session, tokenizer):
        """Test SIFFeature preprocessing."""
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
            {"_id": 11, "a_attr": "world python"},
        ])
        b_df = spark_session.createDataFrame([
            {"_id": 20, "b_attr": "python code"},
        ])

        cache = BuildCache()
        feature = SIFFeature('a_attr', 'b_attr', tokenizer)
        feature.build(a_df, b_df, cache)

        data = pd.DataFrame({
            'a_attr': ['hello world', 'python code'],
        })
        result = feature._preprocess(data, 'a_attr')

        assert isinstance(result, pd.Series)
        assert result.name == feature._get_vector_column('a_attr')
        assert len(result) == 2

    def test_call(self, spark_session, tokenizer):
        """Test SIFFeature __call__ method."""
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
            {"_id": 11, "a_attr": "world python"},
        ])
        b_df = spark_session.createDataFrame([
            {"_id": 20, "b_attr": "python code"},
        ])

        cache = BuildCache()
        feature = SIFFeature('a_attr', 'b_attr', tokenizer)
        feature.build(a_df, b_df, cache)

        a_pdf = pd.DataFrame({
            'a_attr': ['hello world', 'world python'],
        })
        b_pdf = pd.DataFrame({
            'b_attr': ['python code'],
        })

        a_pdf = feature.preprocess(a_pdf, True)
        b_pdf = feature.preprocess(b_pdf, False)

        result = feature(b_pdf.iloc[0], a_pdf)

        assert isinstance(result, pd.Series)
        assert len(result) == 2
        assert all(
            isinstance(v, (float, np.floating)) or np.isnan(v)
            for v in result.values
        )
