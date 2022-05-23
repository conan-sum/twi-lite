import seaborn as sns
import time
import warnings
from joblib import logger
warnings.filterwarnings('ignore')


class Pipeline:
    def __init__(self, extract, transform, load, feature=None, storage=None):
        self.steps = extract
        self.transform = transform
        self.load = load
        self.feature = feature
        self.storage = storage
        self.index = None
        self.labels = None

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

    def run(self, df=None):
        start = time.time()
        if not df:
            df = self.database.fetch(self.feature)
        self.index = df[df.columns[0]].to_numpy()
        print(f'[ETL 1/4] COMPLETE ............. EXTRACT, TOTAL TIME={logger.short_format_time(time.time() - start)}')

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


