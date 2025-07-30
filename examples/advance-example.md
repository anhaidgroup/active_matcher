## Running ActiveMatcher in the Advanced Mode

ActiveMatcher has two modes: basic and advanced. The default is the basic mode and it works quite well for small to moderate size datasets. For large datasets (for example, tables of 5M+ tuples), you may want to use the advanced mode. Here we explain the difference between the two, then explain how you can use the advanced mode. We first discuss the single-machine setting, then the cluster-of-machines setting. 

### The Advanced Mode for the Single-Machine Setting.

#### Motivation
We describe using ActiveMatcher in the basic mode in the single-machine setting [here](https://github.com/anhaidgroup/active_matcher/blob/main/examples/Single-Machine-Example.md). You may want to study that document and become familiar with the basic mode before continuing with this document. 

As that document discusses, in the basic mode, ActiveMatcher executes multiple iterations of active learning. In each iteration, it performs the following steps:
1. Trains a matcher using all examples that have been labeled so far. 
2. Applies the matcher to all examples in the candidate set and select a set of most informative examples (say 10). Recall that the candidate set is the output of the blocking step. It consists of tuple pairs (x,y). We have converted each such tuple pair to a feature vector. For simplicity, we will use the terms "example", "tuple pair", and "feature vector" interchangeably. The context should make clear what we refer to.
3. Ask the user to label the selected examples.

For example, ActiveMatcher may executes 50 iterations, in each of which it asks the user to label 10 examples as match/non-match. 

For small to moderate size tables, the above basic mode works well. But for larger tables (such as 5M+ tuples), it may face a problem. If the tables are large, then the candidate set (the output of blocking) is often also quite large, having 50M to 500M tuple pairs, or more. In such cases, Step 2 of the iteration will take a long time to finish, because we have to apply the trained matcher to *all* examples in the candidate set. This time can be anywhere from 3 to 10 minutes, depending on the size of the candidate set and the underlying hardware. 

This means that after the user has labeled say 10 examples in an iteration, he or she would need to wait 3-10 minutes before a set of 10 new examples becomes available for the user to label. This is not a good user experience, and results in a long wait time. If the user has to label for 50 iterations, the wait time alone is already 150 minutes or more. 

ActiveMatcher provides two solutions to the above problem: Sampling and Continuous Labeling, which we describe below. ***How can these be combined? Can we do CL without sampling?***

#### Solution: Sampling
The advanced mode tries to solve this problem. It takes a sample (of a much smaller size) from the candidate set, then performs active learning only on the sample. For example, if the candidate set has 100M examples, then ActiveMatcher takes a sample S of only 5M examples, then performs active learning on S. That is, in Step 2 of each iteration, it applies the trained matcher to just the 5M examples in S, not to all 100M examples in the candidate set. This incurs far less time. 

Of course, ActiveMatcher cannot take a *random* sample S from the candidate set, because this sample is likely to contain very few true matches, and thus is not a good sample to perform active learning on. Instead, ActiveMatcher tries to ensure that the sample S contains a variety of matches. 

#### Using Sampling 
We now walk you through the steps of using sampling. ***The complete Python file for this case can be found here.***
To use sampling, right after the code to compute a score for each feature vector (Step 10 in the [document](https://github.com/anhaidgroup/active_matcher/blob/main/examples/Single-Machine-Example.md) that describes running ActiveMatcher in the basic mode on a single machine), you should add the following code to the Python file: 

```
from active_matcher.algorithms import down_sample

sampled_fvs = down_sample(fvs, percent = .1, score_column= 'score')
```

By setting percent = .1, we are telling the down_sample method to give us a sample of the feature vectors with a size |fvs|*.1, so we will end up with 10% of the original size of fvs (which contains the feature vectors of all tuple pairs in the candidate set). 

Then in Step 11 (Selecting Seeds) of the same document, instead of selecting seeds from the candidate set, you should select from the sample. So instead of this code: 
```
seeds = select_seeds(fvs, 50, labeler, 'score')
```
You should use this code: 
```
seeds = select_seeds(sampled_fvs, 50, labeler, 'score')
```

Finally, in Step 12 (Using Active Learning to train the Matcher) of the same document, instead of performing active learning on the *entire* candidate set, using this code: 
```
active_learner = EntropyActiveLearner(model, labeler, batch_size=10, max_iter=50)
trained_model = active_learner.train(fvs, seeds)
```
You should perform active learning on the sample, using this code: 
```
active_learner = EntropyActiveLearner(model, labeler, batch_size=10, max_iter=50)
trained_model = active_learner.train(sampled_fvs, seeds)
```

That's it. All other steps remain the same. In particular, note that in Step 13 (Applying the Trained Matcher), even though you trained the matcher on examples selected from the sample (in Step 12), you should now apply the trained matcher to *all* examples in the candidate set to predict match/non-match. So the code for Step 13 remains the same, which is as follows: 
```
fvs = trained_model.predict(fvs, 'features', 'prediction')
fvs = trained_model.prediction_conf(fvs, 'features', 'confidence')
```
***The complete Python file in this case can be found here.***

   
