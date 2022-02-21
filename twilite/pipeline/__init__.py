import seaborn as sns
import numpy as np


class Pipeline:
    def __init__(self, preprocess=None, transform=None, evaluate=None, database=None):
        self.preprocess = preprocess
        self.transform = transform
        self.evaluate = evaluate
        self.database = database
        self.best_param_ = None
        self.labels = None
        self.eval_report = None

    def fit(self, data):
        self.preprocess.read_df(data)
        self.preprocess.filter()
        mat, _id = self.preprocess.sparse()
        self.transform.X = mat
        self.transform.author_ids = _id
        self.transform.rescale()
        df = self.transform.projection()
        param = self.evaluate.eval(df=df)
        self.best_param_ = param
        arr = df[['xcord', 'ycord']].to_numpy()
        model = self.evaluate.model(param)
        labels = model.fit_predict(arr)
        df['label'] = labels
        self.labels = df
        self.eval_report = np.array(self.evaluate.report).reshape(-1,2)
        return None

    def scatter_plot(self):
        df = self.labels
        return sns.scatterplot(x=df['xcord'], y=df['ycord'], hue=['c ' + str(x) for x in df['label']])

    def load_config(self, config_file=None, from_db=False, **kwargs):
        pass

    def save_config(self, config_file=None, to_db=False):
        pass

    def save_labels(self):
        pass
