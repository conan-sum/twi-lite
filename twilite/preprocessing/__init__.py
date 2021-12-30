from twilite.preprocessing.hashtag import Hashtag, Hashtags
from twilite.preprocessing.retweet import Retweet, Retweets
from twilite.preprocessing.user import User, Users


class Feature:
    def __init__(self, hashtags=True, retweets=True):
        self.hashtags = hashtags
        self.retweets = retweets
        self.user_stats = Users(hashtags=hashtags, retweets=retweets)
        if hashtags:
            self.hashtags_list = Hashtags()
        if retweets:
            self.retweets_list = Retweets()

    def __repr__(self):
        return f"{self.user_stats}"

    def extract(self, data):
        user_id = data['author_id']
        user = self.user_stats.find_user(user_id=user_id)
        if self.hashtags:
            try:
                hts = data['entities']['hashtags']
                for ht in hts:
                    user.add_hashtag(ht['tag'].lower())
                    self.hashtags_list.add_hashtag(ht['tag'].lower())
            except KeyError:
                pass
        if self.retweets:
            try:
                ref_type = data['referenced_tweets'][0]['type']
                if ref_type == 'retweeted':
                    ref_tweet_id = data['referenced_tweets'][0]['id']
                    user.add_retweet(ref_tweet_id)
                    self.retweets_list.add_retweet(ref_tweet_id)
            except KeyError:
                pass
        return None
