## Running ActiveMatcher in the Advanced Mode

ActiveMatcher has two modes: basic and advanced. The default is the basic mode and it works quite well for small to moderate size datasets. For large datasets (for example, tables of 5M+ tuples), you may want to use the advanced mode. Here we explain the difference between the two, then explain how you can use the advanced mode. We first discuss the single-machine setting, then the cluster-of-machines setting. 

### The Advanced Mode in the Single-Machine Setting.

We describe using ActiveMatcher in the basic mode in the single-machine setting [here](https://github.com/anhaidgroup/active_matcher/blob/main/examples/Single-Machine-Example.md). You may want to study that document and become familiar with the basic mode before continuing with this document. 

As that document discusses, in the basic mode, ActiveMatcher executes multiple iterations of active learning. In each iteration, it performs the following steps:
1. Trains a matcher using all examples that have been labeled so far. 
2. Applies the matcher to all examples in the candidate set and select a set of most informative examples (say 10). Recall that the candidate set is the output of the blocking step. It consists of tuple pairs (x,y). We have converted each such tuple pair to a feature vector. For simplicity, we will use the terms "example", "tuple pair", and "feature vector" interchangeably. The context should make clear what we refer to.
3. Ask the user to label the selected examples.

For example, ActiveMatcher may executes 50 iterations, in each of which it asks the user to label 10 examples as match/non-match. 


   
