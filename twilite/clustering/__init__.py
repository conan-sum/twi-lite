import seaborn as sns


class Cluster:
    def __init__(self, data, model):
        self.data = data
        self.label = data
        self.model = model

    def fit(self):
        df = self.data
        arr = df[['xcord', 'ycord']].to_numpy()
        model = self.model.clustering(self.model.parameter)
        labels = model.fit_predict(arr)
        self.label['label'] = labels
        return None

    def labels(self, concat=True):
        if concat:
            return self.label.head(15)
        return self.label

    def graph(self):
        df = self.label
        return sns.scatterplot(x=df['xcord'], y=df['ycord'], hue=['c '+str(x) for x in df['label']])
