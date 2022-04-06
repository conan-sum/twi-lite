import numpy as np
import pandas as pd


class Decomposition:
    def __init__(self, scaler=None, mapper=None):
        self.X = None
        self.author_ids = None
        self.scaler = scaler
        self.mapper = mapper
        if not self.mapper.random_state:
            self.mapper.random_state = 42
        self.embeddings = None

    def projection(self):
        matrix, ids = self.X, self.author_ids
        cord = self.mapper.fit_transform(matrix)
        data = self.scaler.fit_transform(np.array(cord))
        x, y = data.T
        output_df = pd.DataFrame(ids, columns=['author_id'])
        output_df['xcord'] = x
        output_df['ycord'] = y
        output_df.dropna(inplace=True)
        return output_df

