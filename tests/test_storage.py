"""Tests for active_matcher.storage module.

This module provides memmap and sqlite-backed storage utilities.
"""
import pytest
import numpy as np
import pandas as pd

from active_matcher.storage import (
    MemmapArray,
    MemmapDataFrame,
    SqliteDataFrame,
    SqliteDict,
)


class TestMemmapArray:
    """Tests for MemmapArray behavior."""

    def test_init_and_delete(self):
        """Verify MemmapArray writes to disk and cleans up."""
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        mmap_arr = MemmapArray(arr)

        assert mmap_arr.shape == arr.shape
        assert mmap_arr._dtype == arr.dtype
        assert len(mmap_arr) == len(arr)

        file_path = mmap_arr._local_mmap_file
        assert file_path.exists()

        mmap_arr.delete()
        assert not file_path.exists()

    def test_init_properties(self):
        """Test MemmapArray properties."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        mmap_arr = MemmapArray(arr)

        assert mmap_arr.shape == (2, 2)
        assert mmap_arr._dtype == np.float32
        assert len(mmap_arr) == 2

    def test_init_method(self):
        """Test MemmapArray init method."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        mmap_arr = MemmapArray(arr)
        # _on_spark is not set by default, so init() will use local file
        # We need to set it to False explicitly since it's checked in init()
        if not hasattr(mmap_arr, '_on_spark'):
            mmap_arr._on_spark = False
        mmap_arr.init()
        assert mmap_arr.values is not None
        assert np.array_equal(mmap_arr.values, arr)
        mmap_arr.delete()


class TestMemmapDataFrame:
    """Tests for MemmapDataFrame lifecycle."""

    def test_compress_decompress(self):
        """Test compress and decompress static methods."""
        data = b"test data"
        compressed = MemmapDataFrame.compress(data)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

        decompressed = MemmapDataFrame.decompress(compressed)
        assert decompressed == data

    def test_fetch(self, spark_session):
        """Confirm fetch reconstructs DataFrame from memmap."""
        import pickle
        import zlib
        import pandas as pd
        from pyspark.sql.functions import length

        a_df = spark_session.createDataFrame([
            {"_id": 10, "attr": "a", "num": 1.0},
            {"_id": 11, "attr": "b", "num": 2.0},
        ])

        stored_columns = ['attr', 'num']
        rows_data = []
        for row in a_df.collect():
            row_dict = row.asDict()
            row_data = [row_dict[col] for col in stored_columns]
            pickled = pickle.dumps(row_data)
            compressed = zlib.compress(pickled)
            rows_data.append((row_dict['_id'], compressed))

        pickle_df = spark_session.createDataFrame(
            rows_data,
            schema="_id long, compressed binary"
        )

        pickle_df = pickle_df.withColumn("sz", length("compressed"))

        mmap_df = MemmapDataFrame.from_spark_df(
            pickle_df, 'compressed', stored_columns
        )

        mmap_df._offset_arr._on_spark = False
        mmap_df._id_to_offset_map._memmap_arr._on_spark = False
        mmap_df.init()

        result = mmap_df.fetch([10, 11])

        expected = pd.DataFrame(
            [['a', 1.0], ['b', 2.0]],
            index=[10, 11],
            columns=stored_columns,
            dtype=object,
        )

        pd.testing.assert_frame_equal(result, expected)
        mmap_df.delete()

    def test_init_delete(self):
        """Test MemmapDataFrame init and delete."""
        mmap_df = MemmapDataFrame()

        file_path = mmap_df._local_mmap_file
        assert file_path.exists()

        mmap_df.delete()
        assert not file_path.exists()

    def test_fetch_after_to_spark(self, spark_session):
        """Test MemmapDataFrame to_spark."""
        import pickle
        import zlib
        import pandas as pd
        from pyspark.sql.functions import length

        a_df = spark_session.createDataFrame([
            {"_id": 10, "attr": "a", "num": 1.0},
            {"_id": 11, "attr": "b", "num": 2.0},
        ])

        stored_columns = ['attr', 'num']
        rows_data = []
        for row in a_df.collect():
            row_dict = row.asDict()
            row_data = [row_dict[col] for col in stored_columns]
            pickled = pickle.dumps(row_data)
            compressed = zlib.compress(pickled)
            rows_data.append((row_dict['_id'], compressed))

        pickle_df = spark_session.createDataFrame(
            rows_data,
            schema="_id long, compressed binary"
        )

        pickle_df = pickle_df.withColumn("sz", length("compressed"))

        mmap_df = MemmapDataFrame.from_spark_df(
            pickle_df, 'compressed', stored_columns
        )

        mmap_df.to_spark()
        mmap_df.init()

        result = mmap_df.fetch([10, 11])

        expected = pd.DataFrame(
            [['a', 1.0], ['b', 2.0]],
            index=[10, 11],
            columns=stored_columns,
            dtype=object,
        )

        pd.testing.assert_frame_equal(result, expected)
        mmap_df.delete()


class TestSqliteDataFrame:
    """Tests for SqliteDataFrame operations."""

    def test_fetch(self, spark_session):
        """Validate fetch returns expected rows from sqlite backend."""
        import pickle

        a_df = spark_session.createDataFrame([
            {"_id": 10, "attr": "a", "num": 1.0},
            {"_id": 11, "attr": "b", "num": 2.0},
        ])

        stored_columns = ['attr', 'num']
        rows_data = []
        for row in a_df.collect():
            row_dict = row.asDict()
            row_data = {col: row_dict[col] for col in stored_columns}
            rows_data.append((row_dict['_id'], pickle.dumps(row_data)))

        pickle_df = spark_session.createDataFrame(
            [(id_val, pickle_val) for id_val, pickle_val in rows_data],
            schema="_id long, pickle binary"
        )

        sqlite_df = SqliteDataFrame.from_spark_df(
            pickle_df, 'pickle', stored_columns
        )

        sqlite_df.init()

        result = sqlite_df.fetch([10, 11])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.index) == [10, 11]
        assert 'attr' in result.columns
        assert 'num' in result.columns
        assert result.loc[10, 'attr'] == 'a'
        assert result.loc[10, 'num'] == 1.0
        assert result.loc[11, 'attr'] == 'b'
        assert result.loc[11, 'num'] == 2.0

        result_single = sqlite_df.fetch([10])
        assert len(result_single) == 1
        assert result_single.loc[10, 'attr'] == 'a'

        result_reordered = sqlite_df.fetch([11, 10])
        assert len(result_reordered) == 2
        assert set(result_reordered.index) == {11, 10}
        assert result_reordered.loc[11, 'attr'] == 'b'
        assert result_reordered.loc[10, 'attr'] == 'a'

        sqlite_df._local_tmp_file.unlink()

    def test_init_to_spark(self, spark_session):
        """Test SqliteDataFrame init and to_spark."""
        sqlite_df = SqliteDataFrame()

        sqlite_df.init()
        assert sqlite_df._conn is not None

        sqlite_df.to_spark()
        sqlite_df._local_tmp_file.unlink()


class TestSqliteDict:
    """Tests for SqliteDict operations."""

    def test_getitem(self):
        """Ensure __getitem__ returns values and raises on missing keys."""
        d = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
        sqlite_dict = SqliteDict.from_dict(d)

        sqlite_dict.init()

        result = sqlite_dict[['key1', 'key2']]
        assert result == ['value1', 'value2']

        result = sqlite_dict[['key3']]
        assert result == ['value3']

        with pytest.raises(KeyError):
            sqlite_dict[['missing_key']]

        sqlite_dict._local_tmp_file.unlink()

    def test_getitem_empty(self):
        """Test __getitem__ with empty list."""
        d = {'key1': 'value1'}
        sqlite_dict = SqliteDict.from_dict(d)

        sqlite_dict.init()

        result = sqlite_dict[[]]
        assert result == []

        sqlite_dict._local_tmp_file.unlink()

    def test_init_deinit_to_spark(self, spark_session):
        """Test SqliteDict init, deinit, and to_spark."""
        d = {'key1': 'value1'}
        sqlite_dict = SqliteDict.from_dict(d)

        sqlite_dict.init()
        assert sqlite_dict._conn is not None

        sqlite_dict.deinit()
        assert sqlite_dict._conn is None

        sqlite_dict.to_spark()

        sqlite_dict._local_tmp_file.unlink()
