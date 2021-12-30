from twilite.preprocessing import Hashtag, Retweet


class User:
    def __init__(self, ht, rt, user_id=""):
        self.user_id = user_id
        self.ht = ht
        self.rt = rt
        if ht:
            self.hashtags = []
        if rt:
            self.retweets = []

    def __repr__(self):
        if self.ht and not self.rt:
            return f"User({self.user_id}, {self.hashtags})"
        if self.rt and not self.ht:
            return f"User({self.user_id}, {self.retweets})"
        return f"User({self.user_id}, {self.hashtags}, {self.retweets})"

    def add_hashtag(self, ht):
        if self.ht:
            for i in self.hashtags:
                if i.hashtag == ht:
                    i.count += 1
                    return None
            self.hashtags.append(Hashtag(hashtag=ht, count=1))
            return None
        return None

    def add_retweet(self, rt):
        if self.rt:
            for i in self.retweets:
                if i.user_id == rt:
                    i.count += 1
                    return None
            self.retweets.append(Retweet(user_id=rt, count=1))
            return None
        return None


class Users:
    def __init__(self, hashtags, retweets):
        self.users = []
        self.hashtags = hashtags
        self.retweets = retweets

    def __repr__(self):
        return f"Users({self.users})"

    def find_user(self, user_id):
        for i in self.users:
            if i.user_id == user_id:
                return i
        user = User(user_id=user_id, ht=self.hashtags, rt=self.retweets)
        self.users.append(user)
        return user
