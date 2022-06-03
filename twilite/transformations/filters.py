class FeatureFilter:
    def __init__(self, user_num=2, ft_freq=2, ft_num=1):
        self.user_num = user_num
        self.ft_freq = ft_freq
        self.ft_num = ft_num

    def __repr__(self):
        return f"FeatureFilter(user_num={self.user_num}, ft_freq={self.ft_freq}, ft_num={self.ft_num})"

    def fit(self, df):
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

    def fit(self, df):
        df.columns = ['userid', 'feature', 'ft_count']
        df = df.groupby(['userid', 'feature'], axis=0, as_index=False).sum()
        # filter out features that have appeared in the data less than a certain times
        ft_count = df["feature"].value_counts()
        ft_count = ft_count.iloc[:self.k]
        df = df[df["feature"].isin(ft_count.index)].reset_index()
        df.drop(["index"], axis=1, inplace=True)
        return df