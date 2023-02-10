import sqlite3
import numpy as np
from pyspark import SparkFiles, SparkContext, StorageLevel
import pyspark.sql.functions as F
import pandas as pd
import pickle
from timeit import timeit
from random import choices
from tempfile import mkstemp
from pathlib import Path
import os
from itertools import islice
from active_matcher.utils import get_logger
from active_matcher.utils import decompress, compress
from joblib import Parallel, delayed

log = get_logger(__name__)



class MemmapArray:

    def __init__(self, arr):
        self._dtype = arr.dtype
        self._shape = arr.shape
        self._local_mmap_file = Path(mkstemp(suffix='.mmap_arr')[1])
        mmap = np.memmap(self._local_mmap_file, shape=self._shape, dtype=self._dtype, mode='w+')
        mmap[:] = arr[:]
        mmap.flush()
        self._mmap_arr = None

    @property
    def values(self):
        return self._mmap_arr

    @property
    def shape(self):
        return self._shape
    
    def __len__(self):
        return self._shape[0]

    def init(self):
        if self._mmap_arr is None:
            f = self._local_mmap_file
            if self._on_spark:
                f = SparkFiles.get(f.name)
                if not os.path.exists(f):
                    raise RuntimeError('cannot find database file at {f}')
            self._mmap_arr = np.memmap(f, mode='r', shape=self._shape, dtype=self._dtype)
    
    def delete(self):
        if self._local_mmap_file.exists():
            self._local_mmap_file.unlink()

    def to_spark(self):
        SparkContext.getOrCreate().addFile(str(self._local_mmap_file))
        self._on_spark = True


class MemmapDataFrame:

    def __init__(self):
        self._local_mmap_file = Path(mkstemp(suffix='.df.db')[1])
        self._index_arr = None
        self._offset_arr = None
        self._mmap_arr = None
        self._map_arr_shape = None
        self._on_spark = False
        self._conn = None
        self._columns = None

    
    @staticmethod
    def compress(o):
        return compress(o)

    @staticmethod
    def decompress(o):
        return decompress(o)
    
    def write_chunk(self, fd, _id, pic):
        idx = np.searchsorted(self._index_arr, _id)
        start = self._offset_arr[idx]
        os.pwrite(fd, memoryview(pic), start)


    @classmethod
    def from_spark_df(cls, spark_df, pickle_column, stored_columns, id_col='_id'):
        obj = cls()
        obj._columns = stored_columns

        spark_df = spark_df.select(
                        F.col(id_col).alias('_id'),
                        F.col(pickle_column).alias('pickle'), 
                        F.length(pickle_column).alias('sz')
                    )\
                    .persist(StorageLevel.DISK_ONLY)
        # this is necessary to prevent memory errors
        pdf = spark_df.select('_id', 'sz')\
                        .toPandas()\
                        .sort_values('_id')

        nrows = len(pdf)

        index_arr = pdf['_id'].values

        offset_arr = np.zeros(nrows+1, dtype=np.uint64)
        offset_arr[1:] = pdf['sz'].cumsum()

        total_bytes = offset_arr[-1]

        obj._index_arr = index_arr
        obj._offset_arr = offset_arr
        log.info('building memmap dataframe')
        itr = spark_df.toLocalIterator(True)
        with open(obj._local_mmap_file, 'wb') as ofs:
            fd = ofs.fileno()
            os.set_inheritable(fd, True)
            Parallel(n_jobs=-1, backend='threading')(delayed(obj.write_chunk)(fd, _id, pic) for _id, pic, sz in itr)

        obj._mmap_arr_shape = total_bytes

        spark_df.unpersist()
        return obj
 
    def init(self):
        if self._mmap_arr is None:
            f = self._local_mmap_file
            if self._on_spark:
                f = SparkFiles.get(f.name)
                if not os.path.exists(f):
                    raise RuntimeError('cannot find database file at {f}')
            self._mmap_arr = np.memmap(f, mode='r', shape=self._mmap_arr_shape)
    
    def delete(self):
        if self._local_mmap_file.exists():
            self._local_mmap_file.unlink()

    def to_spark(self):
        SparkContext.getOrCreate().addFile(str(self._local_mmap_file))
        self._on_spark = True
    

    def fetch(self, ids):
        self.init()
        ids = np.array(ids) 
        # sort for the ids
        srt = ids.argsort()
        # inverse sort to return the results in the order
        # that the ids were provided
        inv_srt = np.empty_like(srt)
        np.put(inv_srt, srt, np.arange(srt.size, dtype=srt.dtype))
        # sort ids to improve binary search perf
        sorted_ids = ids[srt]
        idxes = np.searchsorted(self._index_arr, sorted_ids)

        if np.any(self._index_arr[idxes] != sorted_ids):
            raise ValueError('unknown id')  

        starts = self._offset_arr[idxes]
        ends = self._offset_arr[idxes+1]
        # create array and invert sort to return 
        # in order that ids was passed to function
        rows = np.array([pickle.loads(self.decompress(self._mmap_arr[start:end])) for start, end in zip(starts, ends)], dtype=object)[inv_srt]
        df = pd.DataFrame(rows, index=ids, columns=self._columns, dtype=object)
        return df


class SqliteDataFrame:

    def __init__(self):
        self._local_tmp_file = Path(mkstemp(suffix='.df.db')[1])
        self._on_spark = False
        self._conn = None
        self._columns = None


    @classmethod
    def from_spark_df(cls, spark_df, pickle_column, stored_columns, id_col='_id'):
        chunk_size = 250
        obj = cls()
        obj._columns = stored_columns
        obj._init_db()

        spark_df = spark_df.select(
                        F.col(id_col).alias('_id'),
                        F.col(pickle_column).alias('pickle')
                    ).persist(StorageLevel.DISK_ONLY)
        spark_df.count()

        log.info('building sqlite dataframe')
        conn = obj._get_conn()       
        itr = spark_df.toLocalIterator(True)
        df = pd.DataFrame.from_records(list(islice(itr, chunk_size)), columns=['_id', 'pickle'])
        while len(df) > 0:
            df.to_sql('dataframe', conn, if_exists='append', index=False, method='multi')
            df = pd.DataFrame.from_records(list(islice(itr, chunk_size)), columns=['_id', 'pickle'])
        conn.commit()
        conn.close()

        spark_df.unpersist()
        return obj
 
    def init(self):
        if self._conn is None:
            self._conn = self._get_conn()

    def to_spark(self):
        SparkContext.getOrCreate().addFile(str(self._local_tmp_file))
        self._on_spark = True
    
    def _init_db(self):
        conn = self._get_conn()
        conn.execute('DROP TABLE IF EXISTS dataframe;')
        conn.execute('''CREATE TABLE dataframe (
                            _id,
                            pickle,
                            PRIMARY KEY (_id)
                    ) WITHOUT ROWID;
                '''
                )

    def _get_conn(self):
        f = self._local_tmp_file
        if self._on_spark:
            f = SparkFiles.get(f.name)
            if not os.path.exists(f):
                raise RuntimeError('cannot find database file at {f}')
        f = str(f)
        conn = sqlite3.connect(f)
        conn.execute('PRAGMA synchronous=OFF')
        conn.execute('PRAGMA journal_mode=OFF')
        sz = os.path.getsize(f)
        if sz > 0:
            conn.execute(f'PRAGMA mmap_size = {sz};')
        conn.commit()
        return conn


    def fetch(self, ids):
        self.init()
        # TODO add caching for queries?
        query = f'''
        SELECT _id, pickle
        FROM dataframe
        WHERE _id in (?{",?" * (len(ids)-1)});
        '''

        res = self._conn.execute(query, ids).fetchall()
        index = [t[0] for t in res]
        if len(res) != len(ids):
            missings_ids = set(ids) - set(index) 
            raise RuntimeError(f'not all ids found : {len(res)} of {len(ids)} returned, missing {missings_ids}')

        rows = [pickle.loads(t[1]) for t in res]
        df = pd.DataFrame(rows, index=index, columns=self._columns, dtype=object)
        return df


class SqliteDict:

    def __init__(self):
        self._local_tmp_file = Path(mkstemp(suffix='.df.db')[1])
        self._on_spark = False
        self._conn = None


    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj._init_db()

        log.info('building sqlite dict')
        conn = obj._get_conn()       

        df = pd.DataFrame.from_records(list(d.items()), columns=['key', 'val'])
        chunk_size = 10000
        for i in range(0, len(df), chunk_size):
            df.iloc[i : min(i+chunk_size, len(df))].to_sql('dataframe', conn, if_exists='append', index=False, method='multi')

        if i != len(df):
            df.iloc[i:].to_sql('dataframe', conn, if_exists='append', index=False, method='multi')

        conn.commit()
        conn.close()

        return obj
 
    
    def init(self):
        if self._conn is None:
            self._conn = self._get_conn()

    def deinit(self):
        self._conn = None

    def to_spark(self):
        SparkContext.getOrCreate().addFile(str(self._local_tmp_file))
        self._on_spark = True
    
    def _init_db(self):
        conn = self._get_conn()
        conn.execute('DROP TABLE IF EXISTS dict;')
        conn.execute('''CREATE TABLE dict (
                            key,
                            val,
                            PRIMARY KEY (key, val)
                    ) WITHOUT ROWID;
                '''
                )

    def _get_conn(self):
        f = self._local_tmp_file
        if self._on_spark:
            f = SparkFiles.get(f.name)
            if not os.path.exists(f):
                raise RuntimeError('cannot find database file at {f}')
        f = str(f)
        conn = sqlite3.connect(f)
        conn.execute('PRAGMA synchronous=OFF')
        conn.execute('PRAGMA journal_mode=OFF')
        sz = os.path.getsize(f)
        if sz > 0:
            conn.execute(f'PRAGMA mmap_size = {sz};')
        conn.commit()
        return conn


    def __getitem__(self, keys):
        self.init()
        keys = list(keys)

        if len(keys) == 0:
            return []

        # TODO add caching for queries?
        query = f'''
        SELECT key, val
        FROM dataframe
        WHERE key in (?{",?" * (len(keys)-1)});
        '''

        res = self._conn.execute(query, keys).fetchall()
        res_dict = dict(res)

        missing = {k for k in keys if k not in res_dict}
        if len(missing) != 0:
            raise KeyError(f'not all ids found : {len(res)} of {len(keys)} returned, missing {missing}')

        return [res_dict[k] for k in keys]

