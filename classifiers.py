import math
import numpy as np


class MultinomialNB_BOW:
    def __init__(self, name="NB-BOW", features=None, alpha=0.01):
        self.__data = {}
        self.__c_prob = {}
        self.__fc_prob = {}
        self.name = name
        self.features = features
        self.f_length = len(self.features)
        self.alpha = alpha

    def fit(self, X, Y):
        # Classes prob
        a_y = np.array(Y)
        total_c = len(a_y)
        if total_c < 1:
            raise ValueError("Empty class vector")

        c_unique, c_counts = np.unique(a_y, return_counts=True)
        self.__c_prob = dict(zip(c_unique, [c_count / total_c for c_count in c_counts]))

        # Features prob
        for x, y in zip(X, Y):
            for f, count in x.items():
                if f in self.features:
                    y_data = self.__data.get(y, {})
                    y_data[f] = y_data.get(f, self.alpha) + count
                    self.__data[y] = y_data

        for y in self.__data.keys():
            for x, x_count in self.__data[y].items():
                tmp = self.__fc_prob.get(y, {})
                tmp[x] = x_count / (len(self.__data[y]) + self.f_length)
                self.__fc_prob[y] = tmp

    def predict(self, subjects):
        predictions = []
        probs = []
        log_probs = {}
        for features in subjects:
            for c, c_prob in self.__c_prob.items():
                log_probs[c] = math.log10(c_prob)

                for f, f_count in features.items():
                    if f in self.features:
                        log_probs[c] += math.log10(self.__fc_prob[c].get(f, self.alpha))

            res = max(log_probs, key=log_probs.get)
            probs.append(math.pow(10, log_probs[res]))
            predictions.append(res)

        return predictions, probs
