class Hashtag:
    def __init__(self, hashtag="", count=0):
        self.hashtag = hashtag
        self.count = count

    def __repr__(self):
        return f"Hashtag({self.hashtag}, {self.count})"


class Hashtags:
    def __init__(self):
        self.hashtags = []

    def __repr__(self):
        return f"Hashtags({self.hashtags})"

    def top(self, k=5):
        return None

    def add_hashtag(self, ht):
        for i in self.hashtags:
            if i.hashtag == ht:
                i.count += 1
                return None
        hashtag = Hashtag(hashtag=ht, count=1)
        self.hashtags.append(hashtag)
        return None