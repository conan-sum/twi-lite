import seaborn as sns
import numpy as np
import time
import warnings
from joblib import logger
warnings.filterwarnings('ignore')


class Pipeline:
    def __init__(self, steps, evaluate=None, feature=None, database=None):
        self.steps = steps
        self.evaluate = evaluate
        self.feature = feature
        self.database = database
        self.index = None
        self.best_param_ = None
        self.labels = None
        self.eval_report = None

    def run(self, df=None):
        start = time.time()
        if not df:
            df = self.database.fetch(self.feature)
        self.index = df[df.columns[0]].to_numpy()
        print(f'[ETL 1/4] COMPLETE .......... PREPROCESS, TOTAL TIME={logger.short_format_time(time.time() - start)}')

        split = time.time()
        for i in self.steps:
            df = i.transform(df)
        print(f'[ETL 2/4] COMPLETE ...... TRANSFORMATION, TOTAL TIME={logger.short_format_time(time.time() - split)}')

        split = time.time()
        self.evaluate.search(df)
        df['label'] = self.evaluate.labels
        print(f'[ETL 3/4] COMPLETE .... MODEL EVALUATION, TOTAL TIME={logger.short_format_time(time.time() - split)}')

        split = time.time()
        self.labels = df.sort_values(by='label')
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

