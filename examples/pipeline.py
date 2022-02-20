import pandas as pd
import umap
from twilite.preprocessing import Matrix
from twilite.transformation import Decomposition
from twilite.model_tuning import Validation
from twilite.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('data/data.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

hashtag = Pipeline(
    preprocess=Matrix(filter_by='frequency', user_num=5, ft_freq=50, ft_num=3),
    transform=Decomposition(mapper=umap.UMAP(n_components=2)),
    evaluate=Validation(model=KMeans, eval_range=range(2,10), metric=silhouette_score)
)

hashtag.fit(df)

print(hashtag.results)
print(hashtag.model.parameter)

if __name__ == "__main__":
    pass