from twilite.clustering import Cluster


class Pipeline:
    def __init__(self, preprocess, transform, evaluate):
        self.preprocess = preprocess
        self.transform = transform
        self.evaluate = evaluate
        self.model = None
        self.results = None

    def load_data(self, data):
        for i in data:
            self.preprocess.extract(i)

    def classify(self):
        df = self.transform.fit_transform(self.preprocess)
        self.model = self.evaluate.evaluate(df=df)
        cluster = Cluster(df, self.model)
        cluster.fit()
        self.results = cluster.labels(concat=False)
        return None
