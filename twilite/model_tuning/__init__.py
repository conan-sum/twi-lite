class Validation:
    def __init__(self, model, eval_range, metric):
        self.model = model
        self.eval_range = eval_range
        self.metric = metric
        self.report = []

    def eval(self, df):
        S = []
        arr = df[['xcord', 'ycord']].to_numpy()
        for i in self.eval_range:
            labels = self.model(i).fit_predict(arr)
            score = self.metric(arr, labels)
            S.append((i, score))
            self.report.append((i, score))
        S.sort(key=lambda x: x[1], reverse=True)
        return S[0][0]
