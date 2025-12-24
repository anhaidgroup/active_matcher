"""Tests for active_matcher.tokenizer.vectorizer module.

This module converts token hashes to dense vector representations.
"""
import numpy as np
import pytest

from active_matcher.tokenizer.vectorizer import (
    TFIDFVectorizer,
    SIFVectorizer,
    _doc_freq_to_idf,
    _doc_freq_to_sif,
    _vectorize_tfidf,
    _vectorize_sif,
)
from active_matcher.feature.vector_feature import DocFreqBuilder
from active_matcher.tokenizer import StrippedWhiteSpaceTokenizer


class TestTFIDFVectorizer:
    """Tests for TFIDFVectorizer and helpers."""

    def test_init(self):
        """Test TFIDFVectorizer initialization."""
        vectorizer = TFIDFVectorizer()
        assert vectorizer._N is None
        assert vectorizer._hash_func is None
        assert vectorizer._hashes is None
        assert vectorizer._idfs is None

    def test_out_col_name(self):
        """Test out_col_name method."""
        vectorizer = TFIDFVectorizer()
        result = vectorizer.out_col_name('tokens')
        assert result == 'term_vec(tokens)'

    def test_build_from_doc_freqs(self, spark_session):
        """Test build_from_doc_freqs method."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
            {"_id": 11, "a_attr": "world python"},
        ])
        b_df = spark_session.createDataFrame([
            {"_id": 20, "b_attr": "python code"},
        ])

        doc_freqs = DocFreqBuilder('a_attr', 'b_attr', tokenizer)
        doc_freqs.build(a_df, b_df)

        vectorizer = TFIDFVectorizer()
        vectorizer.build_from_doc_freqs(doc_freqs)

        assert vectorizer._N is not None
        assert vectorizer._N == len(doc_freqs.doc_freqs_)
        assert vectorizer._hash_func is not None
        assert vectorizer._hashes is not None
        assert vectorizer._idfs is not None
        assert len(vectorizer._idfs) == vectorizer._N
        assert len(vectorizer._hashes) == vectorizer._N

    def test_hash(self, spark_session):
        """Test _hash method."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
        ])

        doc_freqs = DocFreqBuilder('a_attr', 'b_attr', tokenizer)
        doc_freqs.build(a_df, None)

        vectorizer = TFIDFVectorizer()
        vectorizer.build_from_doc_freqs(doc_freqs)

        hash1 = vectorizer._hash('hello')
        hash2 = vectorizer._hash('world')
        assert isinstance(hash1, (int, np.integer))
        assert isinstance(hash2, (int, np.integer))

    def test_init_method(self, spark_session):
        """Test init method."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
        ])

        doc_freqs = DocFreqBuilder('a_attr', 'b_attr', tokenizer)
        doc_freqs.build(a_df, None)

        vectorizer = TFIDFVectorizer()
        vectorizer.build_from_doc_freqs(doc_freqs)
        vectorizer.init()

        assert hasattr(vectorizer._idfs, 'values')
        assert hasattr(vectorizer._hashes, 'values')

    def test_vectorize(self, spark_session):
        """Test vectorize method with valid tokens."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
            {"_id": 11, "a_attr": "world python"},
        ])

        doc_freqs = DocFreqBuilder('a_attr', 'b_attr', tokenizer)
        doc_freqs.build(a_df, None)

        vectorizer = TFIDFVectorizer()
        vectorizer.build_from_doc_freqs(doc_freqs)
        vectorizer.init()

        tokens = ['hello', 'world']
        result = vectorizer.vectorize(tokens)

        assert result is not None
        assert hasattr(result, 'dot')
        assert hasattr(result, 'indexes')
        assert hasattr(result, 'values')
        assert len(result.indexes) == len(result.values)

    def test_vectorize_none(self, spark_session):
        """Test vectorize method with None input."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
        ])

        doc_freqs = DocFreqBuilder('a_attr', 'b_attr', tokenizer)
        doc_freqs.build(a_df, None)

        vectorizer = TFIDFVectorizer()
        vectorizer.build_from_doc_freqs(doc_freqs)
        vectorizer.init()

        assert vectorizer.vectorize(None) is None

    def test_vectorize_empty(self, spark_session):
        """Test vectorize method with empty tokens."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
        ])

        doc_freqs = DocFreqBuilder('a_attr', 'b_attr', tokenizer)
        doc_freqs.build(a_df, None)

        vectorizer = TFIDFVectorizer()
        vectorizer.build_from_doc_freqs(doc_freqs)
        vectorizer.init()

        result = vectorizer.vectorize([])
        assert result is not None
        assert len(result.indexes) == 0
        assert len(result.values) == 0

    def test_doc_freq_to_idf(self):
        """Test _doc_freq_to_idf outputs correct idf values."""
        doc_freq = np.array([1, 2, 5], dtype=np.int64)
        corpus_size = 10

        result = _doc_freq_to_idf(doc_freq, corpus_size)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == len(doc_freq)

        expected = np.log((corpus_size + 1) / (doc_freq + 1)) + 1
        np.testing.assert_array_almost_equal(
            result, expected.astype(np.float32)
        )

    def test_doc_freq_to_idf_edge_cases(self):
        """Test _doc_freq_to_idf with edge cases."""
        doc_freq = np.array([1, 10], dtype=np.int64)
        corpus_size = 10

        result = _doc_freq_to_idf(doc_freq, corpus_size)

        assert len(result) == 2
        assert result[0] > result[1]

    def test_vectorize_tfidf(self):
        """Test _vectorize_tfidf shapes and values."""
        hash_idx = np.array([100, 200, 300, 400], dtype=np.int64)
        idfs = np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float32)
        hashes = np.array([200, 300], dtype=np.int64)
        values = np.array([2, 3], dtype=np.int64)

        idxes, vec_values = _vectorize_tfidf(hash_idx, idfs, hashes, values)

        assert isinstance(idxes, np.ndarray)
        assert isinstance(vec_values, np.ndarray)
        assert idxes.dtype == np.int32
        assert vec_values.dtype == np.float32
        assert len(idxes) == len(vec_values)
        assert len(idxes) == len(hashes)

        assert np.all(idxes >= 0)
        assert np.all(idxes < len(hash_idx))

    def test_vectorize_tfidf_normalized(self):
        """Test _vectorize_tfidf normalizes vectors."""
        hash_idx = np.array([100, 200], dtype=np.int64)
        idfs = np.array([1.0, 1.0], dtype=np.float32)
        hashes = np.array([100, 200], dtype=np.int64)
        values = np.array([1, 1], dtype=np.int64)

        idxes, vec_values = _vectorize_tfidf(hash_idx, idfs, hashes, values)

        norm = np.linalg.norm(vec_values, 2)
        np.testing.assert_almost_equal(norm, 1.0, decimal=5)

    def test_vectorize_tfidf_unknown_hash(self):
        """Test _vectorize_tfidf raises error for unknown hash."""
        hash_idx = np.array([100, 200], dtype=np.int64)
        idfs = np.array([1.0, 1.0], dtype=np.float32)
        hashes = np.array([999], dtype=np.int64)
        values = np.array([1], dtype=np.int64)

        with pytest.raises(ValueError, match='unknown hash'):
            _vectorize_tfidf(hash_idx, idfs, hashes, values)


class TestSIFVectorizer:
    """Tests for SIFVectorizer and helpers."""

    def test_init(self):
        """Test SIFVectorizer initialization."""
        vectorizer = SIFVectorizer()
        assert vectorizer._a_param == 0.001
        assert vectorizer._N is None
        assert vectorizer._sifs is None
        assert vectorizer._hash_func is None
        assert vectorizer._hashes is None

    def test_out_col_name(self):
        """Test out_col_name method."""
        vectorizer = SIFVectorizer()
        result = vectorizer.out_col_name('tokens')
        assert result == 'sif_vec(tokens)'

    def test_build_from_doc_freqs(self, spark_session):
        """Test build_from_doc_freqs method."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
            {"_id": 11, "a_attr": "world python"},
        ])
        b_df = spark_session.createDataFrame([
            {"_id": 20, "b_attr": "python code"},
        ])

        doc_freqs = DocFreqBuilder('a_attr', 'b_attr', tokenizer)
        doc_freqs.build(a_df, b_df)

        vectorizer = SIFVectorizer()
        vectorizer.build_from_doc_freqs(doc_freqs)

        assert vectorizer._N is not None
        assert vectorizer._N == len(doc_freqs.doc_freqs_)
        assert vectorizer._hash_func is not None
        assert vectorizer._hashes is not None
        assert vectorizer._sifs is not None
        assert len(vectorizer._sifs) == vectorizer._N
        assert len(vectorizer._hashes) == vectorizer._N

    def test_hash(self, spark_session):
        """Test _hash method."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
        ])

        doc_freqs = DocFreqBuilder('a_attr', 'b_attr', tokenizer)
        doc_freqs.build(a_df, None)

        vectorizer = SIFVectorizer()
        vectorizer.build_from_doc_freqs(doc_freqs)

        hash1 = vectorizer._hash('hello')
        hash2 = vectorizer._hash('world')
        assert isinstance(hash1, (int, np.integer))
        assert isinstance(hash2, (int, np.integer))
        assert hash1 != hash2

    def test_init_method(self, spark_session):
        """Test init method."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
        ])

        doc_freqs = DocFreqBuilder('a_attr', 'b_attr', tokenizer)
        doc_freqs.build(a_df, None)

        vectorizer = SIFVectorizer()
        vectorizer.build_from_doc_freqs(doc_freqs)
        vectorizer.init()

        assert hasattr(vectorizer._sifs, 'values')
        assert hasattr(vectorizer._hashes, 'values')

    def test_vectorize(self, spark_session):
        """Test vectorize method with valid tokens."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
            {"_id": 11, "a_attr": "world python"},
        ])

        doc_freqs = DocFreqBuilder('a_attr', 'b_attr', tokenizer)
        doc_freqs.build(a_df, None)

        vectorizer = SIFVectorizer()
        vectorizer.build_from_doc_freqs(doc_freqs)
        vectorizer.init()

        tokens = ['hello', 'world']
        result = vectorizer.vectorize(tokens)

        assert result is not None
        assert hasattr(result, 'dot')
        assert hasattr(result, 'indexes')
        assert hasattr(result, 'values')
        assert len(result.indexes) == len(result.values)

    def test_vectorize_none(self, spark_session):
        """Test vectorize method with None input."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
        ])

        doc_freqs = DocFreqBuilder('a_attr', 'b_attr', tokenizer)
        doc_freqs.build(a_df, None)

        vectorizer = SIFVectorizer()
        vectorizer.build_from_doc_freqs(doc_freqs)
        vectorizer.init()

        assert vectorizer.vectorize(None) is None

    def test_vectorize_empty(self, spark_session):
        """Test vectorize method with empty tokens."""
        tokenizer = StrippedWhiteSpaceTokenizer()
        a_df = spark_session.createDataFrame([
            {"_id": 10, "a_attr": "hello world"},
        ])

        doc_freqs = DocFreqBuilder('a_attr', 'b_attr', tokenizer)
        doc_freqs.build(a_df, None)

        vectorizer = SIFVectorizer()
        vectorizer.build_from_doc_freqs(doc_freqs)
        vectorizer.init()

        result = vectorizer.vectorize([])
        assert result is not None
        assert len(result.indexes) == 0
        assert len(result.values) == 0

    def test_doc_freq_to_sif(self):
        """Test _doc_freq_to_sif computes smoothed frequencies."""
        doc_freq = np.array([1, 2, 5], dtype=np.int64)
        corpus_size = 10
        a_param = 0.001

        result = _doc_freq_to_sif(doc_freq, corpus_size, a_param)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == len(doc_freq)

        expected = a_param / ((doc_freq / corpus_size) + a_param)
        np.testing.assert_array_almost_equal(
            result, expected.astype(np.float32)
        )

    def test_doc_freq_to_sif_edge_cases(self):
        """Test _doc_freq_to_sif with edge cases."""
        doc_freq = np.array([1, 10], dtype=np.int64)
        corpus_size = 10
        a_param = 0.001

        result = _doc_freq_to_sif(doc_freq, corpus_size, a_param)

        assert len(result) == 2
        assert result[0] > result[1]

    def test_vectorize_sif(self):
        """Test _vectorize_sif returns expected embeddings."""
        hash_idx = np.array([100, 200, 300, 400], dtype=np.int64)
        sifs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        hashes = np.array([200, 300], dtype=np.int64)
        values = np.array([2, 3], dtype=np.int64)

        idxes, vec_values = _vectorize_sif(hash_idx, sifs, hashes, values)

        assert isinstance(idxes, np.ndarray)
        assert isinstance(vec_values, np.ndarray)
        assert idxes.dtype == np.int32
        assert vec_values.dtype == np.float32
        assert len(idxes) == len(vec_values)
        assert len(idxes) == len(hashes)

        assert np.all(idxes >= 0)
        assert np.all(idxes < len(hash_idx))

    def test_vectorize_sif_normalized(self):
        """Test _vectorize_sif normalizes vectors."""
        hash_idx = np.array([100, 200], dtype=np.int64)
        sifs = np.array([0.1, 0.1], dtype=np.float32)
        hashes = np.array([100, 200], dtype=np.int64)
        values = np.array([1, 1], dtype=np.int64)

        idxes, vec_values = _vectorize_sif(hash_idx, sifs, hashes, values)

        norm = np.linalg.norm(vec_values, 2)
        np.testing.assert_almost_equal(norm, 1.0, decimal=5)

    def test_vectorize_sif_unknown_hash(self):
        """Test _vectorize_sif raises error for unknown hash."""
        hash_idx = np.array([100, 200], dtype=np.int64)
        sifs = np.array([0.1, 0.1], dtype=np.float32)
        hashes = np.array([999], dtype=np.int64)
        values = np.array([1], dtype=np.int64)

        with pytest.raises(ValueError, match='unknown hash'):
            _vectorize_sif(hash_idx, sifs, hashes, values)
