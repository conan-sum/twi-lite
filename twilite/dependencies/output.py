

def to_database(df, kwargs):
    db = kwargs['db']
    feature = kwargs['feature']
    db.save_to_db(feature=feature, df=df)
    return None


def to_csv(df, kwargs):
    path = kwargs['path']
    return None


def to_dict(df, kwargs):
    return None
