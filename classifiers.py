class MultinomialNB:

    def __init__(self, name="NB-BOW", vocabulary=None, alpha=0.01):
        self.name = name
        self.voc = vocabulary
        self.alpha = alpha
        self.cb = []

    def register_on_(self, f):
        if callable(f):
            self.cb.append(f)

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass
