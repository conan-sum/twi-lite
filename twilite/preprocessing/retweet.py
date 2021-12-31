class Retweet:
    def __init__(self, user_id="", count=0):
        self.user_id = user_id
        self.count = count

    def __repr__(self):
        return f"Retweet({self.user_id}, {self.count})"


class Retweets:
    def __init__(self):
        self.retweets = []

    def __repr__(self):
        return f"Retweets({self.retweets})"

    def top(self, k=5):
        return None

    def add_retweet(self, rt):
        for i in self.retweets:
            if i.user_id == rt:
                i.count += 1
                return None
        retweet = Retweet(user_id=rt, count=1)
        self.retweets.append(retweet)
        return None
