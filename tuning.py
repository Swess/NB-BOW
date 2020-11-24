from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


def tune_models(training_set, testing_set, model="DT"):
    train_features = training_set[:, 0:-1]
    train_indexes = training_set[:, -1]

    test_features = testing_set[:, 0:-1]
    test_indexes = testing_set[:, -1]

    scores = [
        'accuracy',
        'precision_macro',
        'precision_weighted',
        'recall_macro',
        'recall_weighted',
        'f1_weighted',
        'f1_macro']  # ['precision', 'recall', 'f1']

    # Set the parameters by cross-validation
    if model == "DT":
        tuned_parameters = [{'criterion': ['gini', 'entropy'],
                             'max_depth': [10, None],
                             'min_samples_split': [2, 3, 4, 5],
                             'min_impurity_decrease': [0, 0.1, 0.2],  # np.arange(0, 0.05, 0.008),
                             'class_weight': [None, "balanced"]}]
    else:
        tuned_parameters = [{'activation': ['logistic', 'tanh', 'relu', 'identity'],
                             'solver': ["sgd", "adam"],
                             'hidden_layer_sizes': [(20, 10, 20), (5, 30, 5)]}]

    for score in scores:
        print("=====")
        print("# Tuning hyper-parameters for %s. Please wait..." % score)
        print()

        clf = GridSearchCV(DecisionTreeClassifier() if model == "DT" else MLPClassifier(), tuned_parameters,
                           scoring=score)
        clf.fit(train_features, train_indexes)

        print("Best parameters set found on training set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on training set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full training set.")
        print("The scores are computed on the full validation set.")
        print()
        indexes_true, pred = test_indexes, clf.predict(test_features)
        print(classification_report(indexes_true, pred))
        print()
