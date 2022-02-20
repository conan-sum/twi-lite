class Model:
    def __init__(self, clustering, eval_range, parameter):
        self.clustering = clustering
        self.eval_range = eval_range
        self.parameter = parameter


class Validation:
    def __init__(self, model, eval_range, metric):
        self.model = model
        self.eval_range = eval_range
        self.metric = metric

    def evaluate(self, df):
        S = []
        arr = df[['xcord', 'ycord']].to_numpy()
        for i in self.eval_range:
            labels = self.model(i).fit_predict(arr)
            score = self.metric(arr, labels)
            S.append((i, score))
        S.sort(key=lambda x: x[1], reverse=True)
        model = Model(clustering=self.model, eval_range=self.eval_range, parameter=S[0][0])
        return model
