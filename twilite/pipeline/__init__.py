import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class Pipeline:
    def __init__(self, feature, preprocess=None, transform=None, evaluate=None, database=None):
        self.feature = feature
        self.preprocess = preprocess
        self.transform = transform
        self.evaluate = evaluate
        self.database = database
        self.best_param_ = None
        self.labels = None
        self.eval_report = None

    def run(self, data=None):
        if not data:
            data = self.database.fetch(self.feature)
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
        self.eval_report = np.array(self.evaluate.report).reshape(-1, 2)
        self.database.save_to_db(feature=self.feature, df=self.labels)
        return None

    def scatter_plot(self):
        df = self.labels
        return sns.scatterplot(x=df['xcord'], y=df['ycord'],
                               hue=['c ' + str(x) for x in df['label']])

    def load_config(self, config_file=None, from_db=False, **kwargs):
        pass

    def save_config(self, config_file=None, to_db=False):
        config = {
            'preprocess': self.preprocess.export(),
            'transform': {
                'mapper': str(type(self.transform.mapper)).split('.')[-1][:-2],
            },
            'evaluate': {
                'model': str(type(self.evaluate.model())).split('.')[-1][:-2],
            },
            'range_start': self.evaluate.eval_range[0],
            'range_end': self.evaluate.eval_range[-1],
            'best_parameter': self.best_param_
        }
        return config

    def save_labels(self):
        if self.labels:
            self.database.save_to_db(feature=self.feature, df=self.labels)
        return None
