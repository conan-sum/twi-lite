from twilite.preprocessing.filter import frequency, feature
from scipy import sparse
import numpy as np


class Matrix:
    def __init__(self, filter_by, scaler=None, **kwargs):
        self.filter_by = filter_by
        self.user_num = kwargs.get('user_num') if 'user_num' in kwargs.keys() else 2
        self.ft_freq = kwargs.get('ft_freq') if 'ft_freq' in kwargs.keys() else 2
        self.ft_num = kwargs.get('ft_num') if 'ft_num' in kwargs.keys() else 1
        self.k = kwargs.get('k') if 'k' in kwargs.keys() else 10
        self.df = None
        self.matrix = None
        self.scaler = scaler

    def read_df(self, df):
        self.df = df
        return None

    def sparse(self):
        df = self.df
        if self.filter_by == 'frequency':
            df = frequency(self.df, self.user_num, self.ft_freq, self.ft_num)
        if self.filter_by == 'feature':
            df = feature(self.df, self.k)
        df.columns = ['userid', 'feature', 'ft_count']
        if self.scaler:
            df['ft_count'] = self.scaler.fit_transform(df['ft_count'])
        df = df.assign(uid_matrixid=df.groupby(['userid']).ngroup(), ft_matrixid=df.groupby(['feature']).ngroup())
        output_matrix = sparse.coo_matrix(
            (df.ft_count.values.tolist(), (df.uid_matrixid.values, df.ft_matrixid.values)))
        user_id = np.array(
            df[['userid', 'uid_matrixid']].groupby(['userid', 'uid_matrixid']).head(1)['userid'])
        return output_matrix, user_id

    def export(self):
        args = None
        if self.filter_by == 'frequency':
            args = {
                'user_num': self.user_num,
                'ft_freq': self.ft_freq,
                'ft_num': self.ft_num,
            }
        if self.filter_by == 'feature':
            args = {
                'k': self.k
            }
        return {
            'filter_by': self.filter_by,
            'kwargs': args
        }
