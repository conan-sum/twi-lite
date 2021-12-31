import pandas as pd
from scipy import sparse
import numpy as np
import umap
import warnings
from twilite.feature_representation.filter import feature, frequency
warnings.filterwarnings('ignore')


class HashtagTransformer:
    def __init__(self, filter_by):
        self.filter = filter_by

    def transform(self, data, **kwargs):
        df = pd.DataFrame(columns=['author_id', 'hashtag', 'ft_count'])
        users = data.user_stats.users
        for i in users:
            _id = i.user_id
            hashtags = i.hashtags
            if len(hashtags) > 0:
                for ht in hashtags:
                    row = {'author_id': _id, 'hashtag': ht.hashtag, 'ft_count': ht.count}
                    df = df.append(row, ignore_index=True)
        if self.filter == 'feature':
            try:
                k = kwargs.get('k')
            except KeyError:
                k = 50
            df = feature(df, k)
        if self.filter == 'usage':
            try:
                user_num = kwargs.get('user_num')
                ft_freq = kwargs.get('ft_freq')
                ft_num = kwargs.get('ft_num')
            except KeyError:
                user_num = 2
                ft_freq = 2
                ft_num = 1
            df = frequency(df, user_num=user_num, ft_freq=ft_freq, ft_num=ft_num)
        df.columns = ['author_id', 'hashtag', 'ft_count']
        df = df.assign(uid_matrixid=df.groupby(['author_id']).ngroup(), ft_matrixid=df.groupby(['hashtag']).ngroup())
        output_matrix = sparse.coo_matrix((df.ft_count.values.tolist(), (df.uid_matrixid.values, df.ft_matrixid.values)))
        user_id = np.array(df[['author_id', 'uid_matrixid']].groupby(['author_id', 'uid_matrixid']).head(1)['author_id'])
        return output_matrix, user_id

    def fit_transform(self, data, **kwargs):
        matrix, ids = self.transform(data, **kwargs)
        cord = umap.UMAP(n_components=2, metric='hellinger', random_state=42).fit(matrix)
        data = np.array(cord.embedding_)
        x, y = data.T
        output_df = pd.DataFrame(ids, columns=['author_id'])
        output_df['xcord'] = x
        output_df['ycord'] = y
        output_df.dropna(inplace=True)
        return output_df


class RetweetTransformer:
    def __init__(self, filter_by):
        self.filter = filter_by

    def transform(self, data, **kwargs):
        df = pd.DataFrame(columns=['author_id', 'retweet', 'ft_count'])
        users = data.user_stats.users
        for i in users:
            _id = i.user_id
            retweets = i.retweets
            if len(retweets) > 0:
                for rt in retweets:
                    row = {'author_id': _id, 'retweet': rt.user_id, 'ft_count': rt.count}
                    df = df.append(row, ignore_index=True)
        if self.filter == 'feature':
            try:
                k = kwargs.get('k')
            except KeyError:
                k = 50
            df = feature(df, k)
        if self.filter == 'usage':
            try:
                user_num = kwargs.get('user_num')
                ft_freq = kwargs.get('ft_freq')
                ft_num = kwargs.get('ft_num')
            except KeyError:
                user_num = 2
                ft_freq = 2
                ft_num = 1
            df = frequency(df, user_num=user_num, ft_freq=ft_freq, ft_num=ft_num)
        df.columns = ['author_id', 'retweet', 'ft_count']
        df = df.assign(uid_matrixid=df.groupby(['author_id']).ngroup(), ft_matrixid=df.groupby(['retweet']).ngroup())
        output_matrix = sparse.coo_matrix((df.ft_count.values.tolist(), (df.uid_matrixid.values, df.ft_matrixid.values)))
        user_id = np.array(df[['author_id', 'uid_matrixid']].groupby(['author_id', 'uid_matrixid']).head(1)['author_id'])
        return output_matrix, user_id

    def fit_transform(self, data, **kwargs):
        matrix, ids = self.transform(data, **kwargs)
        cord = umap.UMAP(n_components=2, metric='hellinger', random_state=42).fit(matrix)
        data = np.array(cord.embedding_)
        x, y = data.T
        output_df = pd.DataFrame(ids, columns=['retweet'])
        output_df['xcord'] = x
        output_df['ycord'] = y
        output_df.dropna(inplace=True)
        return output_df
