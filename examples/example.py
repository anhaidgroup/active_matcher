import sys
sys.path.append('.')

from sklearn.metrics import f1_score
import pyspark.sql.functions as F
from active_matcher.active_learning import EntropyActiveLearner
from active_matcher.fv_generator import FVGenerator
from active_matcher.feature_selector import FeatureSelector
from active_matcher.ml_model import  SKLearnModel, SparkMLModel
from active_matcher.labeler import  GoldLabeler
from active_matcher.algorithms import select_seeds
from xgboost import XGBClassifier
import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category=FutureWarning)




def run():
    A = None # TODO spark dataframe of table A  '_id' is id column
    B = None # TODO spark dataframe of table B  '_id' is id column, set to None if dedupe
    columns = None # the columns to generate featues on
    cands_df = None # TODO candidates with (id2 : long, id1_list : array<long>)

    # select the features to be used for feature vector generation
    selector = FeatureSelector(extra_features=False)

    features = selector.select_features(A.drop('_id'), B.drop('_id'))

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
    labeler = GoldLabeler(gold)
    
    # add score column for select seeds
    # here we assume that all features are positively correlated with being a match 
    # hence we sum all of the feature scores, we replace  nan values for scoring purposes
    fvs = fvs.withColumn('score', F.aggregate('features', F.lit(0.0), lambda acc, x : acc + F.when(x.isNotNull() & ~F.isnan(x), x).otherwise(0.0) ))
    seed_ids = select_seeds(fvs, 'score', 50, labeler)

    # Each active learning iteration takes the top-10 examples
    # based on the entropy, for at most 50 iterations
    al = EntropyActiveLearner(model, labeler, batch_size=10, max_iter=50)
    
    # perform active learning
    trained_model = al.train(train_fvs, seed_ids)
    
    # do predictions
    # prediction output into 'prediction' column
    fvs = trained_model.predict(fvs, 'features', 'prediction')
    # prediction confidence output into 'confidence' column
    fvs = trained_model.prediction_conf(fvs, 'features', 'confidence')

    res = fvs.toPandas()
    
    predicted_matches = set(res.loc[res['prediction'].eq(1.0)][['id1', 'id2']].itertuples(name=None, index=False))

    true_positives = len(gold & predicted_matches)
    precision = true_positives / len(predicted_matches)
    recall = true_positives / len(gold)
    f1 = (precision * recall * 2) / (precision + recall)

    print(
f'''
{true_positives=}
{precision=}
{recall=}
{f1=}
'''
)
    
    
    


def main():
    run()

if __name__ == '__main__':
    main()
