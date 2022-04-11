from scipy import sparse
import numpy as np


class SparseMatrix:
    def __init__(self, filter_function, **kwargs):
        self.filter_function = filter_function
        self.kwargs = kwargs
        self.matrix = None

    def sparse(self, df):
        df = self.filter_function(df, self.kwargs)
        df.columns = ['uid', 'feature', 'ft_count']
        df = df.assign(uid_matrixid=df.groupby(['uid']).ngroup(), ft_matrixid=df.groupby(['feature']).ngroup())
        output_matrix = sparse.coo_matrix(
            (df.ft_count.values.tolist(), (df.uid_matrixid.values, df.ft_matrixid.values)))
        u_id = np.array(
            df[['uid', 'uid_matrixid']].groupby(['uid', 'uid_matrixid']).head(1)['uid'])
        self.matrix = output_matrix
        return output_matrix, u_id

    def config(self):
        pass
