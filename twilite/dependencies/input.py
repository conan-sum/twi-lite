import pandas as pd


def read_query(kwargs):
    db = kwargs['db']
    df = db.fetch(kwargs['feature'])
    return df


def read_csv(kwargs):
    path = kwargs['path']
    return pd.DataFrame()


def fetch_users(kwargs):
    user_ids = kwargs['user_ids']
    return pd.DataFrame()
