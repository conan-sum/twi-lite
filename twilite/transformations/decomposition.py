import pandas as pd
import numpy as np
from scipy import sparse


class Manifold:
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
            return f"Manifold(scalar={scaler_name}, mapper={mapper_name})"
        return f"Manifold(mapper={mapper_name})"

    def fit(self, df):
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
