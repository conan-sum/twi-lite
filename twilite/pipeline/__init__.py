import seaborn as sns


class Pipeline:
    def __init__(self, preprocess, transform, evaluate):
        self.preprocess = preprocess
        self.transform = transform
        self.evaluate = evaluate
        self.model = None
        self.results = None

    def fit(self, data):
        self.preprocess.read_df(data)
        self.preprocess.filter()
        mat, _id = self.preprocess.sparse()
        self.transform.X = mat
        self.transform.author_ids = _id
        self.transform.rescale()
        df = self.transform.projection()
        self.model = self.evaluate.evaluate(df=df)
        arr = df[['xcord', 'ycord']].to_numpy()
        model = self.model.clustering(self.model.parameter)
        labels = model.fit_predict(arr)
        df['label'] = labels
        self.results = df
        return None

    def scatter_plot(self):
        df = self.results
        return sns.scatterplot(x=df['xcord'], y=df['ycord'], hue=['c ' + str(x) for x in df['label']])
