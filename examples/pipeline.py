from twilite.preprocessing import Feature
from twilite.feature_representation import HashtagTransformer
from twilite.model_tuning import Validate
from twilite.pipeline import Pipeline
from sklearn.cluster import KMeans
import json

hashtag = Pipeline(
    preprocess=Feature(),
    transform=HashtagTransformer('feature', k=15),
    evaluate=Validate(model=KMeans, eval_range=range(2, 10)),
)

with open("data/tweets.json", encoding='utf-8') as f1:
    data = json.loads(f1.read())['tweets']

hashtag.load_data(data)
hashtag.classify()

print(hashtag.results)

if __name__ == "__main__":
    pass
