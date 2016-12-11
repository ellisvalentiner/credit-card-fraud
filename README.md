# credit-card-fraud

Analysis of credit card fraud data

## Dataset

The datasets contains credit card transactions over a two day collection period in September 2013 by European cardholders. There are a total of 284,807 transactions, of which 492 (0.172%) are fraudulant.

The dataset contains features are numerical variables that are the result of a PCA transformation. This transformation was applied by the original authors to maintain confidentiality of sensitive information. Additionally the dataset contains `Time` and `Amount`, which were not transformed by PCA. The 'Time' variable contains the seconds elapsed between each transaction and the first transaction in the dataset. The 'Amount' variable is the transaction amount, this feature can be used for example-dependant cost-senstive learning. The `Class` variable is the response variable and indicates whether the transaction was fraudulant.

The dataset was collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection. More details on current and past projects on related topics are available on http://mlg.ulb.ac.be/BruFence and http://mlg.ulb.ac.be/ARTML

## Model

The model is a simple multi-layer perceptron (MLP) neural network, implemented in Python using the Keras module and Theano backend.

Run the model using the following command:

```bash
KERAS_BACKEND=theano ipython src/model.py
```

## Performance

```
             precision    recall  f1-score   support

        0.0       1.00      0.98      0.99    284315
        1.0       0.06      0.90      0.11       492

avg / total       1.00      0.98      0.99    284807
```

# Reference

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

# License

The data was released under [Open Database License](http://opendatacommons.org/licenses/odbl/1.0/) and individual contents under [Database Contents License](http://opendatacommons.org/licenses/dbcl/1.0/).

This code repository is released under the [MIT "Expat" License](http://choosealicense.com/licenses/mit/).