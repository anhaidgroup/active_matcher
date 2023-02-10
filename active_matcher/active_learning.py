import pyspark.sql.functions as F    
from pyspark.sql import SparkSession
import pandas as pd
import pyspark.sql.types as T    
from copy import deepcopy, copy
from active_matcher.utils import persisted, get_logger, repartition_df, type_check
from active_matcher.labeler import Labeler
from active_matcher.ml_model import MLModel, SKLearnModel, convert_to_array, convert_to_vector
from pyspark.ml.functions import vector_to_array, array_to_vector
import pyspark
from math import ceil

log = get_logger(__name__)
    
    
class EntropyActiveLearner:    
    
    def __init__(self, model, labeler, batch_size=10, max_iter=50):    
        """

        Parameters
        ----------
        model : MLModel
            the model to be trained

        labeler : Labeler
            the labeler that will be used to label batches during active learning

        batch_size : int
            the number of examples to select per batch

        max_iter : int 
            the maximum number of iterations of active learning

        """

        self._check_init_args(model, labeler, batch_size, max_iter)

        self._batch_size = batch_size
        self._labeler = labeler
        self._labeled_ids = None
        self._model = copy(model)
        self._max_iter = max_iter
        self.local_training_fvs_ = None
        self._terminate_if_label_everything = False
    
    def _check_init_args(self, model, labeler, batch_size=10, max_iter=50):    

        type_check(model, 'model', MLModel)
        type_check(labeler, 'labeler', Labeler)
        type_check(batch_size, 'batch_size', int)
        type_check(max_iter, 'max_iter', int)
        if batch_size <= 0 :
            raise ValueError('batch_size must be > 0')
        if max_iter <= 0 :
            raise ValueError('max_iter must be > 0')
    
    def _select_training_vectors(self, fvs, ids):
        return fvs.filter(F.col('_id').isin(ids))

    
    def _get_pos_negative(self, batch):
        pos = batch['label'].sum()
        neg = len(batch) - pos

        return pos, neg
    
    def _label_everything(self, fvs):
        spark = SparkSession.builder.getOrCreate()
        batch = fvs.toPandas()
        batch['label'] = batch[['id1', 'id2']].apply(lambda x: float(self._labeler(*x.values)), axis=1)
        batch['labeled_in_iteration'] = -2
        self.local_training_fvs_ = batch
        training_fvs = spark.createDataFrame(self.local_training_fvs_)

        self._model.train(training_fvs, 'features', 'label')
        return copy(self._model)
    
    def _prep_fvs(self, fvs):
        # fvs(_id, id1, id2, features)
        fvs = repartition_df(fvs, 25000, '_id')

        return self._model.prep_fvs(fvs)

    def train(self, fvs, start_ids):
        """
        run active learning

        Parameters
        ----------
        fvs : pyspark.sql.DataFrame
            the feature vectors for training, schema must contain (_id Any, id1 long, id2 long, features array<double or float>)

        start_ids : List of _id
            the ids of the seed feature vectors for starting active learning, these must be present in `fvs`
        """
        type_check(fvs, 'fvs', pyspark.sql.DataFrame)

        spark = SparkSession.builder.getOrCreate()
        self._labeled_ids = start_ids.copy()

        fvs = self._prep_fvs(fvs)

        with persisted(fvs) as fvs:
            n_fvs = fvs.count()
            # just label everything and return 
            if n_fvs <= len(start_ids) + (self._batch_size * self._max_iter):
                if self._terminate_if_label_everything:
                    log.info('running al to completion would label everything, labeling all fvs and returning')
                    return self._label_everything(fvs)
                else:
                    log.info('running al to completion would label everything, but self._terminate_if_label_everything is False so AL will still run')


            self.local_training_fvs_ = self._select_training_vectors(fvs, self._labeled_ids).toPandas()
            self.local_training_fvs_['label'] = self.local_training_fvs_[['id1', 'id2']].apply(lambda x: float(self._labeler(*x.values)), axis=1)
            # seed feature vectors
            self.local_training_fvs_['labeled_in_iteration'] = -1
            schema = spark.createDataFrame(self.local_training_fvs_).schema

            max_itr = min(
                    ceil((fvs.count() - len(self.local_training_fvs_) // self._batch_size)),
                    self._max_iter
            )
            
            total_pos, total_neg = self._get_pos_negative(self.local_training_fvs_)
            if total_pos == 0 or total_neg == 0:
                log.error(f'total positive = {total_pos} negative = {total_neg}')
                raise RuntimeError('both postive and negative vectors are required for training')

            log.info(f'max iter = {max_itr}')
            for i in range(max_itr):
                log.info(f'starting iteration {i}')
                # train model
                log.info('training model')
                training_fvs = spark.createDataFrame(self.local_training_fvs_, schema=schema)\
                                    .repartition(len(self.local_training_fvs_) // 100 + 1, '_id')\
                                    .persist()

                self._model.train(training_fvs, 'features', 'label')

                cand_fvs = fvs.join(training_fvs, on='_id', how='left_anti')
                log.info('selecting and labeling new examples')
                # get next labeled batch
                # sort by ids to make training consistent 
                new_labeled_batch = self._model.entropy(cand_fvs, 'features', 'entropy')\
                                        .sort(['entropy', '_id'], ascending=False)\
                                        .limit(self._batch_size)\
                                        .drop('entropy')\
                                        .toPandas()

                training_fvs.unpersist()

                new_labeled_batch['label'] = new_labeled_batch[['id1', 'id2']].apply(lambda x: float(self._labeler(*x.values)), axis=1)
                new_labeled_batch['labeled_in_iteration'] = i

                pos, neg = self._get_pos_negative(new_labeled_batch)
                total_pos += pos
                total_neg += neg

                self.local_training_fvs_ = pd.concat([self.local_training_fvs_, new_labeled_batch], ignore_index=True)
                log.info(f'new batch positive = {pos} negative = {neg}, total positive = {total_pos} negative = {total_neg}')
                
                if n_fvs <= len(self.local_training_fvs_):
                    log.info('all fvs labeled, terminating active learning')
                    break


            training_fvs = spark.createDataFrame(self.local_training_fvs_)
            # final train model
            self._model.train(training_fvs, 'features', 'label')

        return copy(self._model)
