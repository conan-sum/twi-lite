from scipy import sparse
import numpy as np
import pandas as pd


class FeatureFilter:
    def __init__(self, user_num=2, ft_freq=2, ft_num=1):
        self.user_num = user_num
        self.ft_freq = ft_freq
        self.ft_num = ft_num

    def __repr__(self):
        return f"FeatureFilter(user_num={self.user_num}, ft_freq={self.ft_freq}, ft_num={self.ft_num})"

    def transform(self, df):
        df.columns = ['userid', 'feature', 'ft_count']
        df = df.groupby(['userid', 'feature'], axis=0, as_index=False).sum()
        # filter out features that have appeared in the data less than a certain times
        ft_count = df["feature"].value_counts()
        ft_count = ft_count[ft_count >= self.ft_freq]
        df = df[df["feature"].isin(ft_count.index)].reset_index()
        # ft_count is the number of times each hashtag is used by each user
        df = df[df['ft_count'] >= self.ft_num]
        # filter out users that has used less than a certain number of hashtags
        user_count = df['userid'].value_counts()
        user_count = user_count[user_count >= self.user_num]
        df = df[df['userid'].isin(user_count.index)].reset_index()
        df.drop(["index", "level_0"], axis=1, inplace=True)
        return df


class FrequencyFilter:
    def __init__(self, k=10):
        self.k = k

    def __repr__(self):
        return f"FrequencyFilter(k={self.k})"

    def transform(self, df):
        df.columns = ['userid', 'feature', 'ft_count']
        df = df.groupby(['userid', 'feature'], axis=0, as_index=False).sum()
        # filter out features that have appeared in the data less than a certain times
        ft_count = df["feature"].value_counts()
        ft_count = ft_count.iloc[:self.k]
        df = df[df["feature"].isin(ft_count.index)].reset_index()
        df.drop(["index"], axis=1, inplace=True)
        return df


class Decomposition:
    def __init__(self, scaler=None, mapper=None):
        self.scaler = scaler
        self.mapper = mapper
        if not self.mapper.random_state:
            self.mapper.random_state = 42
        self.embeddings = None

    def __repr__(self):
        mapper_name = self.mapper.__class__.__name__
        if self.scaler:
            scaler_name = self.scaler.__class__.__name__
            return f"Decomposition(scalar={scaler_name}, mapper={mapper_name})"
        return f"Decomposition(mapper={mapper_name})"

    def transform(self, df):
        df.columns = ['uid', 'feature', 'ft_count']
        df = df.assign(uid_matrixid=df.groupby(['uid']).ngroup(), ft_matrixid=df.groupby(['feature']).ngroup())
        output_matrix = sparse.coo_matrix(
            (df.ft_count.values.tolist(), (df.uid_matrixid.values, df.ft_matrixid.values)))
        u_id = np.array(
            df[['uid', 'uid_matrixid']].groupby(['uid', 'uid_matrixid']).head(1)['uid'])
        cord = self.mapper.fit_transform(output_matrix)
        if self.scaler:
            data = self.scaler.fit_transform(np.array(cord))
        else:
            data = np.array(cord)
        x, y = data.T
        output_df = pd.DataFrame(u_id, columns=['u_id'])
        output_df['xcord'] = x
        output_df['ycord'] = y
        output_df.dropna(inplace=True)
        return output_df
