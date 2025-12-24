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
    hash_map_insert_key,
    hash_map_insert_keys,
    hash_map_get_key,
    hash_map_get_keys,
    LongIntHashMap,
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
        from pyspark.sql.functions import length

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

        pickle_df = pickle_df.withColumn("sz", length("pickle"))

        mmap_df = MemmapDataFrame.from_spark_df(
            pickle_df, 'pickle', stored_columns
        )

        # _on_spark is False by default for MemmapDataFrame
        # But nested MemmapArray objects need _on_spark set
        mmap_df._offset_arr._on_spark = False
        mmap_df._id_to_offset_map._memmap_arr._on_spark = False
        mmap_df.init()

        result = mmap_df.fetch([10, 11])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.index) == [10, 11]
        assert 'attr' in result.columns
        assert 'num' in result.columns
        assert result.loc[10, 'attr'] == 'a'
        assert result.loc[10, 'num'] == 1.0
        assert result.loc[11, 'attr'] == 'b'
        assert result.loc[11, 'num'] == 2.0

        result_single = mmap_df.fetch([10])
        assert len(result_single) == 1
        assert result_single.loc[10, 'attr'] == 'a'

        mmap_df.delete()

    def test_init_delete(self):
        """Test MemmapDataFrame init and delete."""
        mmap_df = MemmapDataFrame()

        file_path = mmap_df._local_mmap_file
        assert file_path.exists()

        mmap_df.delete()
        assert not file_path.exists()

    def test_to_spark(self, spark_session):
        """Test MemmapDataFrame to_spark."""
        import pickle
        from pyspark.sql.functions import length
        a_df = spark_session.createDataFrame([
            {"_id": 10, "attr": "a", "num": 1.0},
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
        pickle_df = pickle_df.withColumn("sz", length("pickle"))

        mmap_df = MemmapDataFrame.from_spark_df(
            pickle_df, 'pickle', stored_columns
        )
        mmap_df.to_spark()
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


class TestHashMapHelpers:
    """Tests for hash map helper functions and LongIntHashMap."""

    def test_hash_map_insert_and_get(self):
        """Confirm hash_map_insert/get round-trip values."""
        from active_matcher.storage import map_entry_t

        arr_size = 100
        arr = np.zeros(arr_size, dtype=map_entry_t)

        hash_map_insert_key(arr, 5, 10)
        result = hash_map_get_key(arr, 5)
        assert result == 10

        keys = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
        vals = np.array([10, 20, 30, 40, 50], dtype=np.int32)

        arr2 = np.zeros(arr_size, dtype=map_entry_t)
        hash_map_insert_keys(arr2, keys, vals)

        results = hash_map_get_keys(arr2, keys)
        assert np.array_equal(results, vals)

        result = hash_map_get_key(arr2, 999)
        assert result == -1

    def test_hash_map_insert_key_zero(self):
        """Test hash_map_insert_key raises error for zero key."""
        from active_matcher.storage import map_entry_t

        arr = np.zeros(10, dtype=map_entry_t)

        with pytest.raises(ValueError, match='keys must be non zero'):
            hash_map_insert_key(arr, 0, 10)

    def test_long_int_hash_map(self):
        """Ensure LongIntHashMap.build creates accessible map."""
        keys = np.array([10, 20, 30], dtype=np.uint64)
        vals = np.array([0, 1, 2], dtype=np.int32)

        hash_map = LongIntHashMap.build(keys, vals)

        # Set _on_spark = False for memmap array (not set in __init__)
        hash_map._memmap_arr._on_spark = False
        hash_map.init()

        assert hash_map[10] == 0
        assert hash_map[20] == 1
        assert hash_map[30] == 2

        result = hash_map[np.array([10, 20], dtype=np.uint64)]
        assert np.array_equal(result, np.array([0, 1], dtype=np.int32))

        assert hash_map[999] == -1

        assert hash_map[int(10)] == 0
        assert hash_map[np.int64(20)] == 1

    def test_long_int_hash_map_type_error(self):
        """Test LongIntHashMap raises TypeError for invalid types."""
        keys = np.array([10, 20], dtype=np.uint64)
        vals = np.array([0, 1], dtype=np.int32)

        hash_map = LongIntHashMap.build(keys, vals)
        hash_map._memmap_arr._on_spark = False
        hash_map.init()

        with pytest.raises(TypeError):
            hash_map['invalid']

        with pytest.raises(TypeError):
            hash_map[['list', 'of', 'strings']]

    def test_long_int_hash_map_load_factor(self):
        """Test LongIntHashMap with different load factors."""
        keys = np.array([10, 20, 30], dtype=np.uint64)
        vals = np.array([0, 1, 2], dtype=np.int32)

        hash_map = LongIntHashMap.build(keys, vals, load_factor=0.5)
        hash_map._memmap_arr._on_spark = False
        hash_map.init()

        assert hash_map[10] == 0
        assert hash_map[20] == 1
        assert hash_map[30] == 2

    def test_memmap_array_values_property(self):
        """Test MemmapArray values property before and after init."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        mmap_arr = MemmapArray(arr)

        assert mmap_arr.values is None

        if not hasattr(mmap_arr, '_on_spark'):
            mmap_arr._on_spark = False
        mmap_arr.init()

        assert mmap_arr.values is not None
        assert np.array_equal(mmap_arr.values, arr)

        mmap_arr.delete()

    def test_sqlite_dataframe_fetch_missing_ids(
        self, spark_session
    ):
        """Test SqliteDataFrame fetch raises error for missing IDs."""
        import pickle

        a_df = spark_session.createDataFrame([
            {"_id": 10, "attr": "a", "num": 1.0},
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

        with pytest.raises(RuntimeError, match='not all ids found'):
            sqlite_df.fetch([10, 999])

        sqlite_df._local_tmp_file.unlink()

    def test_sqlite_dict_large_dataset(self):
        """Test SqliteDict with larger dataset."""
        d = {f'key_{i}': f'value_{i}' for i in range(100)}

        sqlite_dict = SqliteDict.from_dict(d)

        sqlite_dict.init()

        keys_to_fetch = [f'key_{i}' for i in range(0, 100, 10)]
        result = sqlite_dict[keys_to_fetch]

        assert len(result) == 10
        assert result[0] == 'value_0'
        assert result[-1] == 'value_90'

        sqlite_dict._local_tmp_file.unlink()

    def test_memmap_dataframe_fetch_missing_id(self, spark_session):
        """Test MemmapDataFrame fetch raises error for missing ID."""
        import pickle
        from pyspark.sql.functions import length

        a_df = spark_session.createDataFrame([
            {"_id": 10, "attr": "a", "num": 1.0},
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

        pickle_df = pickle_df.withColumn("sz", length("pickle"))

        mmap_df = MemmapDataFrame.from_spark_df(
            pickle_df, 'pickle', stored_columns
        )

        # Set _on_spark for nested MemmapArray objects
        mmap_df._offset_arr._on_spark = False
        mmap_df._id_to_offset_map._memmap_arr._on_spark = False
        mmap_df.init()

        with pytest.raises(ValueError, match='unknown id'):
            mmap_df.fetch([999])

        with pytest.raises(ValueError, match='unknown id'):
            mmap_df.fetch([10, 999])

        mmap_df.delete()

    def test_memmap_dataframe_fetch_large_dataset(self, spark_session):
        """Test MemmapDataFrame fetch with larger dataset."""
        import pickle
        from pyspark.sql.functions import length

        a_df = spark_session.createDataFrame([
            {"_id": i, "attr": f"val_{i}", "num": float(i)}
            for i in range(20, 30)
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

        pickle_df = pickle_df.withColumn("sz", length("pickle"))

        mmap_df = MemmapDataFrame.from_spark_df(
            pickle_df, 'pickle', stored_columns
        )

        # Set _on_spark for nested MemmapArray objects
        mmap_df._offset_arr._on_spark = False
        mmap_df._id_to_offset_map._memmap_arr._on_spark = False
        mmap_df.init()
        result = mmap_df.fetch([20, 25, 29])
        assert len(result) == 3
        assert result.loc[20, 'attr'] == 'val_20'
        assert result.loc[25, 'num'] == 25.0
        assert result.loc[29, 'attr'] == 'val_29'

        all_ids = list(range(20, 30))
        result_all = mmap_df.fetch(all_ids)
        assert len(result_all) == 10
        assert list(result_all.index) == all_ids

        mmap_df.delete()

    def test_sqlite_dataframe_fetch_single_and_multiple(self, spark_session):
        """Test SqliteDataFrame fetch with single and multiple IDs."""
        import pickle

        a_df = spark_session.createDataFrame([
            {"_id": 10, "attr": "a", "num": 1.0},
            {"_id": 11, "attr": "b", "num": 2.0},
            {"_id": 12, "attr": "c", "num": 3.0},
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

        result = sqlite_df.fetch([10])
        assert len(result) == 1
        assert result.loc[10, 'attr'] == 'a'

        result = sqlite_df.fetch([10, 12])
        assert len(result) == 2
        assert result.loc[10, 'attr'] == 'a'
        assert result.loc[12, 'attr'] == 'c'

        result = sqlite_df.fetch([10, 11, 12])
        assert len(result) == 3
        assert result.loc[11, 'num'] == 2.0

        sqlite_df._local_tmp_file.unlink()
