# Credit Card Fraud Analysis

Analysis of credit card fraud data using neural networks.

## Dataset

The datasets contains credit card transactions over a two day collection period in September 2013 by European cardholders. There are a total of 284,807 transactions, of which 492 (0.172%) are fraudulent.

The dataset contains numerical variables that are the result of a principal components analysis (PCA) transformation. This transformation was applied by the original authors to maintain confidentiality of sensitive information. Additionally the dataset contains `Time` and `Amount`, which were not transformed by PCA. The `Time` variable contains the seconds elapsed between each transaction and the first transaction in the dataset. The `Amount` variable is the transaction amount, this feature can be used for example-dependant cost-senstive learning. The `Class` variable is the response variable and indicates whether the transaction was fraudulant.

The dataset was collected and analysed during a research collaboration of Worldline and the [Machine Learning Group](http://mlg.ulb.ac.be) of Université Libre de Bruxelles (ULB) on big data mining and fraud detection.

## Model

The model is a stratified k-fold multi-layer perceptron (MLP) neural network, implemented in Python using the Keras module and Theano backend (although you can use TensorFlow if you like).

Run the model using the following command:

```bash
KERAS_BACKEND=theano ipython src/model.py
```

## Performance

The model achieves an overall f1 score of 0.99, with 99% sensitivity and 14% precision for the positive class. That is, the model correctly identifies 99% of the fraud cases (true positives) but only 14% of the transactions predicted as fraudulent were actually fraudulent.

The classification report and cross-tabulation are below:

```
             precision    recall  f1-score   support

        0.0       1.00      0.99      0.99    284315
        1.0       0.14      0.99      0.24       492

avg / total       1.00      0.99      0.99    284807

Predictions       0     1
Truth                    
0.0          281250  3065
1.0               6   486
```

# Reference

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015 ([PDF](https://www3.nd.edu/~dial/publications/dalpozzolo2015calibrating.pdf))

# License

The data was released under [Open Database License](http://opendatacommons.org/licenses/odbl/1.0/) and individual contents under [Database Contents License](http://opendatacommons.org/licenses/dbcl/1.0/).

This code repository is released under the [MIT "Expat" License](http://choosealicense.com/licenses/mit/).