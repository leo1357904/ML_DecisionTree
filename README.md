# ML_DecisionTree
***Env: Python 3.6.9***


***inpection.py***: output a file with gini impurity and majority vote error rate for depth 0 tree.
<pre><code>$ python inspection.py example_data/small/small_train.tsv example_data/small/small_inspect.txt</code></pre>
***decisionTree.py***: output prediction files for train and test data and a metrics.txt for both error rates.
<pre><code>$ python decisionTree.py example_data/politicians/politicians_train.tsv example_data/politicians/politicians_test.tsv 2 train_output_name.labels test_output_name.labels metrics_output_name.txt</code></pre>
number 2 is the tree depth.
