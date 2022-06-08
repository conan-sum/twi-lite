import pandas as pd


def read_query(kwargs):
    db = kwargs['db']
    df = db.fetch(kwargs['feature'])
    return df


def find_by_id(kwargs):
    db = kwargs['db']
    ids = kwargs['ids']
    df = db.fetch(kwargs['feature'])
    df = df[df['author_id'].isin(ids)]
    return df


def read_csv(kwargs):
    path = kwargs['path']
    return pd.DataFrame()


def fetch_users(kwargs):
    user_ids = kwargs['user_ids']
    return pd.DataFrame()
