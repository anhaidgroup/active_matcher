## ActiveMatcher: Using Active Learning to Predict Matches for Entity Matching

ActiveMatcher (AM) is an open-source tool that uses active learning to predict matches for entity matching. It works as follows:
* Suppose you have to match two tables A and B. We assume you have performed blocking on A and B (for example, by using Sparkly or Delex, one of the two blocking solutions that we are providing). Let C be the output of the blocking step. C consists of tuple pairs (x,y) where tuple x in Table A and tuple y in Table B are judged (by the blocking step) to be a possible match.
* AM will train a matcher M, which is a learning-based classifier, then apply M to each tuple pair (x,y) in C to predict if it is indeed a match or a non-match.
* To train matcher M, AM performs a process called active learning, in which it will ask you (the user) to label a set of tuple pairs (say 500 pairs) as match/non-match. AM then uses these labeled pairs to train M. 

As described, AM is distinguished in several important aspects: 
* It solves the problem of labeling data for machine learning. You do not have to worry about how to select a data set to label, how to make sure this data set is good for training purposes, etc. AM will take care of this process. You only need to label a few hundred tuple pairs. In practice, this labeling step typically takes 2-3 hours, and can be done by anyone who has been trained on what it means for two tuples x and y to match.
* AM scales to very large datasets. In practice it is not unusual for the set C (the output of blocking) to have 100M to 1B tuple pairs. At this scale, performing well-known machine learning steps (such as featurization) and active learning is difficult. AM built on years of research in our group to solve these problems.
* AM achieves high accuracy using machine learning (ML). Currently AM works with traditional ML models, such as random forest, XGBoost. We are extending AM to work with deep learning and GenAI technologies, but these are not yet made available in the current release. 

### Installation

See instructions to install ActiveMatcher on [a single machine](https://github.com/anhaidgroup/active_matcher/tree/main/doc/installation-guides)  or [a cloud-based cluster](https://github.com/anhaidgroup/active_matcher/blob/main/doc/installation-guides/install-cloud-based-cluster.md). 

### How to Use

See examples on using ActiveMatcher on a [single machine](https://github.com/anhaidgroup/active_matcher/blob/main/examples/Single-Machine-Example.md) and a [cluster of machines](https://github.com/anhaidgroup/active_matcher/blob/main/examples/Cluster-Example.md). 

### Further Pointers

See [API documentation](https://anhaidgroup.github.io/active_matcher). 
For questions / comments, contact [our research group](mailto:entitymatchinginfo@gmail.com).
