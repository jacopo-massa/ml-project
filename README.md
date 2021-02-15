# Machine Learning Project

This project consists in comparing different tools and models to solve a _regression task_, with given train and test sets.

The whole project has been written in **Python 3.7**.

## Setup 💻
Create a virtual environment , and install the dependecies:

```
python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

## Tasks  ✔️

### Classification Task - [MONK's Problem](https://archive.ics.uci.edu/ml/datasets/MONK's+Problems)
We developed a little Neural Network, expoliting **Keras** library, to solve 3 (plus one with regularization) classification tasks. 
Results can be seen in Section 3.4 of the [report](https://github.com/jacopo-massa/ml-project/blob/master/report.pdf).

### Regression Task
We compared two models: _Neural Networks_ (NN) and _Support Vector Machines_ (SVM), exploiting 3 different libraries:
 - **Keras** and **PyTorch** (for NN)
 - **scikit-learn** (for SVM)
 
For both models we used a validation schema consisting into an exhaustive grid search and K-Fold Cross-validation tecnique
for model selection and hyperparameters' tuning.

Please, read the [report](https://github.com/jacopo-massa/ml-project/blob/master/report.pdf) for a deeper description of our work.


## Contributors ✨

 - [Jacopo Massa](https://github.com/jacopo-massa)
 - [Giulio Purgatorio](https://github.com/GPurgatorio)
