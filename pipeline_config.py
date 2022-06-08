from twilite.pipeline import Pipeline
from twilite.transformations import FeatureFilter, Manifold, ParameterEstimation, Model
from twilite.dependencies import find_by_feature, find_by_id, to_database
from twilite.database import Storage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from umap import UMAP
import json


creds = json.load(open("creds.json"))
models = [Model(model=KMeans, eval_range=range(2, 10))]
ht_user = Pipeline(
    feature='ht_user',
    steps=[
        FeatureFilter(user_num=50),
        Manifold(scaler=StandardScaler(), mapper=UMAP(n_components=2)),
        ParameterEstimation(models=models, metric=silhouette_score)
    ],
    source=find_by_feature,
    target=to_database,
    database=Storage(db='blm2', creds=creds)
)


ht_user_subcluster = Pipeline(
    feature='ht_user',
    steps=[
        FeatureFilter(user_num=20),
        Manifold(scaler=StandardScaler(), mapper=UMAP(n_components=2)),
        ParameterEstimation(models=models, metric=silhouette_score)
    ],
    source=find_by_id,
    target=to_database,
    database=Storage(db='blm2', creds=creds)
)


if __name__ == '__main__':
    #ht_user.run()
    print(ht_user)


