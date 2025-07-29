import zlib
import pyspark
import pandas as pd
from pyspark import SparkContext
import numpy as np
import numba as nb
from contextlib import contextmanager
from pyspark import StorageLevel
from random import randint
import pyarrow as pa
import pyarrow.parquet as pq
import mmh3
import sys
import logging
import os
from math import floor

# compression for storage
compress = zlib.compress
decompress = zlib.decompress

logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format='[%(filename)s:%(lineno)s - %(funcName)s() ] %(asctime)-15s : %(message)s',
)


def type_check(var, var_name, expected):
    """
    type checking utility, throw a type error if the var isn't the expected type
    """
    if not isinstance(var, expected):
        raise TypeError(f'{var_name} must be type {expected} (got {type(var)})')

def type_check_iterable(var, var_name, expected_var_type, expected_element_type):
    """
    type checking utility for iterables, throw a type error if the var isn't the expected type
    or any of the elements are not the expected type
    """
    type_check(var, var_name, expected_var_type)
    for e in var:
        if not isinstance(e, expected_element_type):
            raise TypeError(f'all elements of {var_name} must be type{expected_element_type} (got {type(var)})')

def is_null(o):
    """
    check if the object is null, note that this is here to 
    get rid of the weird behavior of np.isnan and pd.isnull
    """
    r = pd.isnull(o)
    return r if isinstance(r, bool) else False

@contextmanager
def persisted(df, storage_level=StorageLevel.MEMORY_AND_DISK):
    """
    context manager for presisting a dataframe in a with statement.
    This automatically unpersists the dataframe at the end of the context
    """
    if df is not None:
        df = df.persist(storage_level) 
    try:
        yield df
    finally:
        if df is not None:
            df.unpersist()

def is_persisted(df):
    """
    check if the pyspark dataframe is persist
    """
    sl = df.storageLevel
    return sl.useMemory or sl.useDisk

def get_logger(name, level=logging.DEBUG):
    """
    Get the logger for a module

    Returns
    -------
    Logger

    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger

def repartition_df(df, part_size, by=None):
    """
    repartition the dataframe into chunk of size 'part_size'
    by column 'by'
    """
    cnt = df.count()
    n = max(cnt // part_size, SparkContext.getOrCreate().defaultParallelism * 4)
    n = min(n, cnt)
    if by is not None:
        return df.repartition(n, by)
    else:
        return df.repartition(n)


class SparseVec: 
    def __init__(self, size, indexes, values):    
        self._size = size    
        self._indexes = indexes.astype(np.int32)
        self._values = values.astype(np.float32)

    @property
    def indexes(self):
        return self._indexes

    @property
    def values(self):
        return self._values
    
    def dot(self, other):    
        return _sparse_dot(self._indexes, self._values, other._indexes, other._values)

@nb.njit('float32(int32[::1], float32[::1],int32[::1], float32[::1])')
def _sparse_dot(l_ind, l_val, r_ind, r_val):
    l = 0
    r = 0
    s = 0.0

    while l < l_ind.size and r < r_ind.size:
        li = l_ind[l]
        ri = r_ind[r]
        if li == ri:
            s += l_val[l] * r_val[r]
            l += 1
            r += 1
        elif li < ri:
            l += 1
        else:
            r += 1

    return s

class PerfectHashFunction:

    def __init__(self, seed=None):
        self._seed = seed if seed is not None else randint(0, 2**31)
    

    @classmethod
    def create_for_keys(cls, keys):
        if len(set(keys)) != len(keys):
            raise ValueError('keys must be unique')
        # used because it is ordered
        hashes = {}

        MAX_RETRIES = 10
        for i in range(MAX_RETRIES):
            hashes.clear()
            hash_func = cls()
            hash_vals = map(hash_func.hash, keys)
            for h, k in zip(hash_vals, keys):
                if h in hashes:
                    break
                hashes[h] = k
            if len(hashes) == len(keys):
                break
        else:
            raise RuntimeError(f'max retries ({MAX_RETRIES}) exceeded')

        return hash_func, np.fromiter(hashes.keys(), dtype=np.int64, count=len(hashes))
    
    def hash(self, s):
        return mmh3.hash64(s, self._seed)[0]


# Training Data Persistence Utilities
def save_training_data_streaming(new_batch, parquet_file_path, logger=None):
    """Save training data using streaming writes for efficiency
    
    This method appends new labeled pairs to an existing parquet file without
    loading the entire dataset into memory. Each labeled pair is saved immediately.
    
    Parameters
    ----------
    new_batch : pandas.DataFrame
        New batch of labeled data to append (columns: _id, id1, id2, label)
    parquet_file_path : str
        Path to save the parquet file
    logger : logging.Logger, optional
        Logger instance for logging messages
    """
    if logger is None:
        logger = get_logger(__name__)
        
    # Ensure we only save the essential columns in consistent order
    required_columns = ['_id', 'id1', 'id2', 'features', 'label']
    new_batch_clean = new_batch[required_columns].copy()
        
    try:
        table = pa.Table.from_pandas(new_batch_clean)
        
        if os.path.exists(parquet_file_path):
            # Read existing data and append
            existing_table = pq.read_table(parquet_file_path)
            # Ensure existing data has same columns
            existing_df = existing_table.to_pandas()
            existing_df_clean = existing_df[required_columns].copy()
            existing_table_clean = pa.Table.from_pandas(existing_df_clean)
            
            combined_table = pa.concat_tables([existing_table_clean, table])
            pq.write_table(combined_table, parquet_file_path)
            logger.info(f'Appended {len(new_batch)} labeled pairs to '
                       f'{parquet_file_path}')
        else:
            # Create new file
            pq.write_table(table, parquet_file_path)
            logger.info(f'Created new training data file: {parquet_file_path}')
            

    except Exception as e:
        logger.warning(f'Streaming save failed: {e}, falling back to pandas save')
        _save_with_pandas(new_batch_clean, parquet_file_path, logger)


def load_training_data_streaming(parquet_file_path, logger=None):
    """Load training data from a parquet file.
    
    Parameters
    ----------
    parquet_file_path : str
        Path to the parquet file
    logger : logging.Logger, optional
        Logger instance for logging messages
        
    Returns
    -------
    pandas.DataFrame or None
        Training data if file exists, None otherwise
    """
    if logger is None:
        logger = get_logger(__name__)
        
    try:
        import pyarrow.parquet as pq
        
        if os.path.exists(parquet_file_path):
            # Read all data efficiently
            table = pq.read_table(parquet_file_path)
            training_data = table.to_pandas()
            
            logger.info(f'Loaded {len(training_data)} labeled pairs from '
                       f'{parquet_file_path}')
            return training_data
        return None
        
    except Exception as e:
        logger.warning(f'Streaming load failed: {e}, falling back to pandas read')
        return _load_with_pandas(parquet_file_path, logger)


def _save_with_pandas(training_data, parquet_file_path, logger):
    """Fallback save method using pandas"""
    try:
        if os.path.exists(parquet_file_path):
            # Read existing data and append
            existing_data = pd.read_parquet(parquet_file_path)
            combined_data = pd.concat([existing_data, training_data], 
                                    ignore_index=True)
            combined_data.to_parquet(parquet_file_path, index=False)
        else:
            training_data.to_parquet(parquet_file_path, index=False)
        logger.info(f'Saved {len(training_data)} labeled pairs to '
                   f'{parquet_file_path}')
    except Exception as e:
        logger.warning(f'Pandas save failed: {e}')


def _load_with_pandas(parquet_file_path, logger):
    """Fallback load method using pandas"""
    try:
        if os.path.exists(parquet_file_path):
            training_data = pd.read_parquet(parquet_file_path)
            logger.info(f'Loaded {len(training_data)} labeled pairs from '
                       f'{parquet_file_path}')
            return training_data
        return None
    except Exception as e:
        logger.warning(f'Pandas load failed: {e}')
        return None


def adjust_iterations_for_existing_data(existing_data_size, n_fvs, batch_size, max_iter):
    """Calculate remaining iterations based on existing data and constraints
    
    This function is designed for batch active learning where iterations
    correspond to discrete batches of labeled examples.
    
    Parameters
    ----------
    existing_data_size : int
        Number of existing labeled examples
    n_fvs : int
        Total number of feature vectors
    batch_size : int
        Number of examples per batch
    max_iter : int
        Maximum number of iterations
        
    Returns
    -------
    int
        Adjusted number of iterations
    """
    completed_iterations = floor(existing_data_size / batch_size)
    remaining_iterations = max_iter - completed_iterations    
    return max(0, remaining_iterations)


def adjust_labeled_examples_for_existing_data(existing_data_size, max_labeled):
    """Calculate remaining labeled examples for continuous active learning
    
    This function is designed for continuous active learning where we track
    the total number of labeled examples rather than iterations.
    
    Parameters
    ----------
    existing_data_size : int
        Number of existing labeled examples
    max_labeled : int
        Maximum number of examples to label
        
    Returns
    -------
    int
        Remaining number of examples that should be labeled
    """
    remaining_examples = max_labeled - existing_data_size
    return max(0, remaining_examples)
