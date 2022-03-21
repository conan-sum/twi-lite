def query(ft):
    queries = {
        'hashtag': "SELECT author_id, hashtags FROM tweets WHERE hashtags IS NOT NULL",
        'retweet': "SELECT author_id, ref_tweet_id FROM tweets WHERE ref_tweet_type = 'retweeted' "
                   "AND ref_tweet_id IS NOT NULL",
    }
    return queries.get(ft)
