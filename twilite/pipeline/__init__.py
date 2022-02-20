import seaborn as sns
from twilite.clustering import Cluster


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
        mat, id = self.preprocess.sparse()
        self.transform.X = mat
        self.transform.author_ids = id
        self.transform.rescale()
        df = self.transform.projection()
        self.model = self.evaluate.evaluate(df=df)
        cluster = Cluster(df, self.model)
        self.results = cluster.fit()
        return None

    def scatter_plot(self):
        df = self.results
        return sns.scatterplot(x=df['xcord'], y=df['ycord'], hue=['c ' + str(x) for x in df['label']])
