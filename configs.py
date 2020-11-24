from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

CLASSIFIERS_1 = {
    "GNB": GaussianNB(),
    "Base-DT": DecisionTreeClassifier(criterion="entropy"),
    "Best-DT": DecisionTreeClassifier(criterion="entropy", min_samples_split=3),
    "PER": Perceptron(),
    "Base-MLP": MLPClassifier(activation="logistic", solver="sgd"),
    "Best-MLP": MLPClassifier(hidden_layer_sizes=(20, 10, 20),
                              activation="identity",
                              solver="adam")
}

CLASSIFIERS_2 = {
    "GNB": GaussianNB(),
    "Base-DT": DecisionTreeClassifier(criterion="entropy"),
    "Best-DT": DecisionTreeClassifier(criterion="gini",
                                      class_weight="balanced",
                                      min_samples_split=2),
    "PER": Perceptron(),
    "Base-MLP": MLPClassifier(activation="logistic", solver="sgd"),
    "Best-MLP": MLPClassifier(hidden_layer_sizes=(20, 10, 20),
                              activation="tanh",
                              solver="sgd"),
}

CLASSIFIERS_DEFAULT = CLASSIFIERS_1
