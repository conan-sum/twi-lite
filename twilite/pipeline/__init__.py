import seaborn as sns
import numpy as np
import time
import warnings
from joblib import logger
warnings.filterwarnings('ignore')


class Pipeline:
    def __init__(self, preprocess, transform, evaluate, feature=None, database=None):
        self.feature = feature
        self.preprocess = preprocess
        self.transform = transform
        self.evaluate = evaluate
        self.database = database
        self.best_param_ = None
        self.labels = None
        self.eval_report = None

    def run(self, data=None):
        start = time.time()
        if not data:
            data = self.database.fetch(self.feature)
        self.preprocess.read_df(data)
        mat, _id = self.preprocess.sparse()
        print(f'[ETL 1/4] COMPLETE .......... PREPROCESS, TOTAL TIME={logger.short_format_time(time.time() - start)}')

        split = time.time()
        self.transform.X = mat
        self.transform.author_ids = _id
        df = self.transform.projection()
        print(f'[ETL 2/4] COMPLETE ...... TRANSFORMATION, TOTAL TIME={logger.short_format_time(time.time() - split)}')

        split = time.time()
        param = self.evaluate.eval(df=df)
        self.best_param_ = param
        arr = df[['xcord', 'ycord']].to_numpy()
        model = self.evaluate.model(param)
        labels = model.fit_predict(arr)
        df['label'] = labels
        print(f'[ETL 3/4] COMPLETE .... MODEL EVALUATION, TOTAL TIME={logger.short_format_time(time.time() - split)}')

        split = time.time()
        self.labels = df.sort_values(by='label')
        self.eval_report = np.array(self.evaluate.report).reshape(-1, 2)
        if self.database:
            self.database.save_to_db(feature=self.feature, df=self.labels)
        else:
            self.labels.to_csv(f'{self.feature}_embeddings.csv')
        print(f'[ETL 4/4] COMPLETE ........... LOAD DATA, TOTAL TIME={logger.short_format_time(time.time() - split)}')
        print(f'PROCESS COMPLETE ...................... , TOTAL TIME={logger.short_format_time(time.time() - start)}')
        return None

    def scatter_plot(self, labels=True):
        df = self.labels
        if labels:
            return sns.scatterplot(x=df['xcord'], y=df['ycord'],
                                   hue=['c' + str(x) for x in df['label']])
        return sns.scatterplot(x=df['xcord'], y=df['ycord'])

    def config(self, config_file=None, to_db=False):
        config = {
            'preprocess': self.preprocess.config(),
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

