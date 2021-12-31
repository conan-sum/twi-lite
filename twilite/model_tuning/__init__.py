from sklearn.metrics import silhouette_score


class Model:
    def __init__(self, model, eval_range):
        self.model = model
        self.eval_range = eval_range
        self.parameter = None


class Validate:
    def __init__(self, algorithm, data):
        self.algorithm = algorithm
        self.data = data

    def evaluate(self):
        df = self.data
        S = []
        arr = df[['xcord', 'ycord']].to_numpy()
        for i in self.algorithm.eval_range:
            labels = self.algorithm.model(i).fit_predict(arr)
            score = silhouette_score(arr, labels, metric='euclidean')
            S.append((i, score))
        S.sort(key=lambda x: x[1], reverse=True)
        self.algorithm.parameter = S[0][0]
        return self.algorithm

