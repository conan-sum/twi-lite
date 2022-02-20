class Cluster:
    def __init__(self, data, model):
        self.data = data
        self.label = None
        self.model = model

    def fit(self):
        df = self.data
        arr = df[['xcord', 'ycord']].to_numpy()
        model = self.model.clustering(self.model.parameter)
        labels = model.fit_predict(arr)
        self.label = df
        self.label['label'] = labels
        return self.label
