class Model:
    def __init__(self, model, eval_range):
        self.model = model
        self.eval_range = eval_range
        self.best_result = None
        self.best_model = None

    def __repr__(self):
        model_name = type(self.model()).__name__
        return f'{model_name}()'

    def evaluate(self, df, metric):
        S = []
        arr = df[['xcord', 'ycord']].to_numpy()
        for i in self.eval_range:
            labels = self.model(i).fit_predict(arr)
            score = metric(arr, labels)
            S.append((self.model(i), score))
        S.sort(key=lambda x: x[1], reverse=True)
        self.best_model = S[0][0]
        self.best_result = S[0][1]
        return None


class GridSearch:
    def __init__(self, models, metric):
        self.models = models
        self.metric = metric
        self.results = []
        self.best_model = None
        self.labels = None

    def __repr__(self):
        if self.best_model:
            model_name = type(self.best_model).__name__
            return f'GridSearch(best_model={model_name}, metric={self.metric})'
        return f'GridSearch(models={self.models}, metric={self.metric.__name__})'

    def search(self, df):
        for i in self.models:
            i.evaluate(df, self.metric)
            self.results.append((i.best_model, i.best_result))
        self.results.sort(key=lambda x: x[1], reverse=True)
        self.best_model = self.results[0][0]
        arr = df[['xcord', 'ycord']].to_numpy()
        self.labels = self.best_model.fit_predict(arr)
        return None

