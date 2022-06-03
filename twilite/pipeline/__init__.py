import pandas as pd
import seaborn as sns
import time
import warnings
from joblib import logger
warnings.filterwarnings('ignore')


class Pipeline:
    def __init__(self, feature, steps, source, target, database=None):
        self.feature = feature
        self.steps = steps
        self.source = source
        self.target = target
        self.database = database
        self.index = None
        self.df = None

    def __repr__(self):
        newline = '\n'
        output = f''
        output += 'PIPELINE CONFIGURATION\n'
        output += f'{newline}'
        output += 'FEATURE\n'
        output += f'feature={self.feature}\n'
        output += f'{newline}'
        output += 'PREPROCESSING\n'
        for i in self.steps:
            output += f'{i}\n'
        output += f'{newline}'
        if self.evaluate:
            output += 'PREDICTIVE MODELING\n'
            output += f'{self.evaluate}\n'
            output += f'{newline}'
        output += 'DATABASE\n'
        output += f'{self.database}'
        return output

    def run(self, verbose=False, **kwargs):
        start = time.time()
        if self.database:
            kwargs['db'] = self.database
            self.database.create_all()
        kwargs['feature'] = self.feature
        df = self.source(kwargs)
        if verbose:
            print(f'FETCH DATA FROM SOURCE COMPLETE, TIME={logger.short_format_time(time.time() - start)}')

        split = time.time()
        total = len(self.steps)
        current = 1
        for i in self.steps:
            df = i.fit(df)
            if verbose:
                print(f'[TRANSFORMATION {current}/{total}] COMPLETE, TIME={logger.short_format_time(time.time() - split)}')
            current += 1

        split = time.time()
        self.df = df.sort_values(by='label')
        self.target(self.df, kwargs)
        if verbose:
            print(f'STORE DATA TO TARGET COMPLETE, TIME={logger.short_format_time(time.time() - split)}')
            print('-------------------------------------------')
            print(f'PROCESS COMPLETE, TOTAL TIME={logger.short_format_time(time.time() - start)}')
        return None

    def scatter_plot(self, labels=True):
        df = self.df
        if labels:
            return sns.scatterplot(x=df['xcord'], y=df['ycord'],
                                   hue=['c' + str(x) for x in df['label']])
        return sns.scatterplot(x=df['xcord'], y=df['ycord'])


