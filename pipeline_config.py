"""
from twilite.pipeline import Pipeline
from twilite.transformations import FeatureFilter, Manifold, ParameterEstimation
from twilite.dependencies import read_query, to_database
from twilite.database import Storage

Pipeline(
    feature = 'ht_user'
    steps = [
        #twilite.transformations
        FeatureFilter(),
        Manifold(scaler=StandardScaler(), mapper=UMAP(n_components=2)),
        ParameterEstimation()
    ],
    source = read_query(sql), #twilite.dependencies
    destination = to_database("BLM2") #twilite.dependencies
    database (optional) = Storage() #twilite.database
)
"""

from twilite.pipeline import Pipeline
from twilite.transformations import FeatureFilter, Manifold, ParameterEstimation, Model
from twilite.dependencies import read_query, to_database
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
    source=read_query,
    destination=to_database,
    database=Storage(db='blm2', creds=creds)
)


if __name__ == '__main__':
    ht_user.run()


