"""Tests for active_matcher.utils module.

This module provides utility functions, compression, hashing, and math helpers.
"""
import pytest
import numpy as np
import pandas as pd

import active_matcher.utils as utils


class TestTypeChecks:
    """Tests for type checking helpers."""

    def test_type_check(self):
        """Verify type_check enforces expected types."""
        utils.type_check(5, 'var', int)
        utils.type_check('hello', 'var', str)
        utils.type_check([1, 2], 'var', list)

        with pytest.raises(TypeError, match='must be type'):
            utils.type_check(5, 'var', str)

        with pytest.raises(TypeError, match='must be type'):
            utils.type_check('hello', 'var', int)

    def test_type_check_iterable(self):
        """Validate type_check_iterable checks element types."""
        utils.type_check_iterable([1, 2, 3], 'var', list, int)
        utils.type_check_iterable((1, 2, 3), 'var', tuple, int)

        with pytest.raises(TypeError, match='must be type'):
            utils.type_check_iterable([1, 2, '3'], 'var', list, int)

        with pytest.raises(TypeError, match='must be type'):
            utils.type_check_iterable('not a list', 'var', list, int)


class TestNullHelpers:
    """Tests for null checking utilities."""

    def test_is_null(self):
        """Ensure is_null returns booleans for scalars."""
        assert utils.is_null(None) is True
        assert utils.is_null(np.nan) is True
        assert utils.is_null(pd.NA) is True
        assert utils.is_null(5) is False
        assert utils.is_null('hello') is False
        assert utils.is_null(0) is False
        assert utils.is_null('') is False


class TestPersistenceHelpers:
    """Tests for persistence utilities."""

    def test_persisted_context(self, spark_session):
        """Check persisted context manager persists and unpersists."""
        df = spark_session.createDataFrame([
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
        ])

        assert not utils.is_persisted(df)

        with utils.persisted(df) as persisted_df:
            assert utils.is_persisted(persisted_df)

        assert not utils.is_persisted(df)

    def test_persisted_context_none(self):
        """Test persisted context manager with None."""
        with utils.persisted(None) as df:
            assert df is None

    def test_is_persisted(self, spark_session):
        """Assert is_persisted reflects StorageLevel flags."""
        from pyspark import StorageLevel

        df = spark_session.createDataFrame([
            {"id": 1, "value": "a"},
        ])

        assert not utils.is_persisted(df)

        df = df.persist(StorageLevel.MEMORY_ONLY)
        assert utils.is_persisted(df)

        df.unpersist()
        assert not utils.is_persisted(df)


class TestLoggingAndRepartition:
    """Tests for logging and repartition helpers."""

    def test_get_logger(self):
        """Ensure get_logger returns configured logger."""
        logger = utils.get_logger('test_module')

        assert logger is not None
        assert logger.name == 'test_module'
        assert logger.level <= utils.logging.DEBUG

    def test_repartition_df(self, spark_session):
        """Confirm repartition_df returns expected partition count."""
        df = spark_session.createDataFrame([
            {"id": i, "value": f"val_{i}"} for i in range(100)
        ])

        repartitioned = utils.repartition_df(df, part_size=10)

        assert repartitioned is not None
        assert repartitioned.count() == 100

        repartitioned_by = utils.repartition_df(df, part_size=10, by='id')

        assert repartitioned_by is not None
        assert repartitioned_by.count() == 100


class TestSparseVec:
    """Tests for SparseVec operations."""

    def test_dot(self):
        """Validate SparseVec.dot uses underlying _sparse_dot correctly."""
        size = 10
        indexes1 = np.array([0, 2, 5], dtype=np.int32)
        values1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        indexes2 = np.array([0, 2, 7], dtype=np.int32)
        values2 = np.array([2.0, 3.0, 4.0], dtype=np.float32)

        vec1 = utils.SparseVec(size, indexes1, values1)
        vec2 = utils.SparseVec(size, indexes2, values2)

        result = vec1.dot(vec2)

        assert abs(result - 8.0) < 1e-6

    def test_sparse_vec_properties(self):
        """Test SparseVec properties."""
        size = 10
        indexes = np.array([0, 2, 5], dtype=np.int32)
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        vec = utils.SparseVec(size, indexes, values)

        assert np.array_equal(vec.indexes, indexes)
        assert np.array_equal(vec.values, values)
        assert vec.indexes.dtype == np.int32
        assert vec.values.dtype == np.float32


class TestPerfectHashFunction:
    """Tests for PerfectHashFunction behavior."""

    def test_create_for_keys(self):
        """Ensure create_for_keys produces unique hashes."""
        keys = ['key1', 'key2', 'key3', 'key4', 'key5']

        hash_func, hash_vals = utils.PerfectHashFunction.create_for_keys(keys)

        assert hash_func is not None
        assert len(hash_vals) == len(keys)

        hashes = [hash_func.hash(k) for k in keys]
        assert len(set(hashes)) == len(keys)

    def test_create_for_keys_duplicates(self):
        """Test create_for_keys raises error for duplicate keys."""
        keys = ['key1', 'key2', 'key1']

        with pytest.raises(ValueError, match='keys must be unique'):
            utils.PerfectHashFunction.create_for_keys(keys)

    def test_hash(self):
        """Test PerfectHashFunction hash method."""
        hash_func = utils.PerfectHashFunction(seed=42)

        hash1 = hash_func.hash('test')
        hash2 = hash_func.hash('test')

        assert hash1 == hash2

        hash3 = hash_func.hash('different')
        assert hash3 != hash1

    def test_init_with_seed(self):
        """Test PerfectHashFunction initialization with seed."""
        hash_func1 = utils.PerfectHashFunction(seed=42)
        hash_func2 = utils.PerfectHashFunction(seed=42)

        hash1 = hash_func1.hash('test')
        hash2 = hash_func2.hash('test')

        assert hash1 == hash2


class TestTrainingDataStreaming:
    """Tests for training data streaming utilities."""

    def test_save_training_data_streaming(self, temp_dir):
        """Write batch to parquet and confirm file contents."""
        parquet_file = temp_dir / 'training_data.parquet'

        new_batch = pd.DataFrame({
            '_id': [1, 2],
            'id1': [10, 11],
            'id2': [20, 21],
            'features': [[0.1, 0.2], [0.3, 0.4]],
            'label': [1.0, 0.0],
        })

        utils.save_training_data_streaming(new_batch, str(parquet_file))

        assert parquet_file.exists()

        loaded = pd.read_parquet(parquet_file)
        assert len(loaded) == 2
        assert list(loaded['_id']) == [1, 2]

    def test_save_training_data_streaming_append(self, temp_dir):
        """Test appending to existing file."""
        parquet_file = temp_dir / 'training_data.parquet'

        batch1 = pd.DataFrame({
            '_id': [1, 2],
            'id1': [10, 11],
            'id2': [20, 21],
            'features': [[0.1, 0.2], [0.3, 0.4]],
            'label': [1.0, 0.0],
        })

        utils.save_training_data_streaming(batch1, str(parquet_file))

        batch2 = pd.DataFrame({
            '_id': [3, 4],
            'id1': [12, 13],
            'id2': [22, 23],
            'features': [[0.5, 0.6], [0.7, 0.8]],
            'label': [1.0, 0.0],
        })

        utils.save_training_data_streaming(batch2, str(parquet_file))

        loaded = pd.read_parquet(parquet_file)
        assert len(loaded) == 4
        assert list(loaded['_id']) == [1, 2, 3, 4]

    def test_load_training_data_streaming(self, temp_dir):
        """Load parquet and validate schema conversion."""
        parquet_file = temp_dir / 'training_data.parquet'

        batch = pd.DataFrame({
            '_id': [1, 2],
            'id1': [10, 11],
            'id2': [20, 21],
            'features': [[0.1, 0.2], [0.3, 0.4]],
            'label': [1.0, 0.0],
        })

        batch.to_parquet(parquet_file, index=False)

        loaded = utils.load_training_data_streaming(str(parquet_file))

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded['_id'].dtype == 'int64'
        assert loaded['id1'].dtype == 'int64'
        assert loaded['id2'].dtype == 'int64'
        assert loaded['label'].dtype == 'float64'
        assert isinstance(loaded['features'].iloc[0], list)

    def test_load_training_data_streaming_nonexistent(self, temp_dir):
        """Test loading non-existent file."""
        parquet_file = temp_dir / 'nonexistent.parquet'

        loaded = utils.load_training_data_streaming(str(parquet_file))

        assert loaded is None


class TestAdjustHelpers:
    """Tests for adjustment helper functions."""

    def test_adjust_iterations_for_existing_data(self):
        """Verify remaining iteration calculation."""
        result = utils.adjust_iterations_for_existing_data(0, 100, 5, 10)
        assert result == 10

        result = utils.adjust_iterations_for_existing_data(5, 100, 5, 10)
        assert result == 9

        result = utils.adjust_iterations_for_existing_data(10, 100, 5, 10)
        assert result == 8

        result = utils.adjust_iterations_for_existing_data(60, 100, 5, 10)
        assert result == 0

    def test_adjust_labeled_examples_for_existing_data(self):
        """Confirm remaining examples calculation respects bounds."""
        result = utils.adjust_labeled_examples_for_existing_data(0, 100)
        assert result == 100

        result = utils.adjust_labeled_examples_for_existing_data(30, 100)
        assert result == 70

        result = utils.adjust_labeled_examples_for_existing_data(100, 100)
        assert result == 0

        result = utils.adjust_labeled_examples_for_existing_data(150, 100)
        assert result == 0
