import sys
sys.path.append('.')

import pyspark.sql.functions as F
from active_matcher.active_learning import EntropyActiveLearner
from active_matcher.fv_generator import FVGenerator
from active_matcher.feature_selector import FeatureSelector
from active_matcher.ml_model import  SKLearnModel, SparkMLModel
from xgboost import XGBClassifier
import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category=FutureWarning)


class Labeler:

    def __init__(self, gold):
        self._gold = gold

    def __call__(self, id1, id2):
        return (id1, id2) in self._gold


def run():
    A = None # TODO spark dataframe of table A  '_id' is id column
    B = None # TODO spark dataframe of table B  '_id' is id column, set to None if dedupe
    columns = None # the columns to generate featues on
    cands_df = None # TODO candidates with (id2 : long, id1_list : array<long>)
    selector = FeatureSelector()
    features = selector.select_features(columns, A, B)

    fv_gen = FVGenerator(features)
    # do preprocessing 
    fv_gen.build(A, B)
    # generate the feature vectors
    fvs = fv_gen.generate_fvs(cands_df)

    # the model that will be trained
    model = SKLearnModel(XGBClassifier, eval_metric='logloss', objective='binary:logistic', max_depth=6, seed=42)

    train_fvs = fvs
    
    gold_df = None # DS.gold.read()
    gold = None # TODO set(zip(gold_df.id1, gold_df.id2))
    labeler = Labeler(gold)
    seed_ids = None # ids from 'fvs'
    #seed_ids = fvs.limit(50).toPandas()['_id'].values.tolist()

    al = EntropyActiveLearner(model, labeler, batch_size=10, max_iter=50)
    # fillna for spark models since it will throw an error otherwise
    if isinstance(model, SparkMLModel):
        feature_col = 'features'
        fvs = fvs.withColumn(feature_col, F.transform(feature_col, lambda x : F.when(x.isNotNull() & ~F.isnan(x), x).otherwise(0.0)))
    
    # perform active learning
    trained = al.train(train_fvs, seed_ids)
    
    # do predictions
    fvs = trained.predict(fvs, 'features', 'prediction')
    fvs = trained.prediction_conf(fvs, 'features', 'confidence')

    return True

def main():
    run()

if __name__ == '__main__':
    main()
