def feature(df, user_num, ft_freq, ft_num):
    df.columns = ['userid', 'feature', 'ft_count']
    df = df.groupby(['userid', 'feature'], axis=0, as_index=False).sum()
    # filter out features that have appeared in the data less than a certain times
    ft_count = df["feature"].value_counts()
    ft_count = ft_count[ft_count >= ft_freq]
    df = df[df["feature"].isin(ft_count.index)].reset_index()
    # ft_count is the number of times each hashtag is used by each user
    df = df[df['ft_count'] >= ft_num]
    # filter out users that has used less than a certain number of hashtags
    user_count = df['userid'].value_counts()
    user_count = user_count[user_count >= user_num]
    df = df[df['userid'].isin(user_count.index)].reset_index()
    df.drop(["index", "level_0"], axis=1, inplace=True)
    return df


def frequency(df, k):
    df.columns = ['userid', 'feature', 'ft_count']
    df = df.groupby(['userid', 'feature'], axis=0, as_index=False).sum()
    # filter out features that have appeared in the data less than a certain times
    ft_count = df["feature"].value_counts()
    ft_count = ft_count.iloc[:k]
    df = df[df["feature"].isin(ft_count.index)].reset_index()
    df.drop(["index"], axis=1, inplace=True)
    return df
