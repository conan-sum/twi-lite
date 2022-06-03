default_queries = {
    'create': {
        'db_config': "CREATE TABLE IF NOT EXISTS config (id INT AUTO_INCREMENT PRIMARY KEY, feature VARCHAR(20));",
        'embeddings': "CREATE TABLE IF NOT EXISTS {i} (config_id INT, feature VARCHAR(50), xcord FLOAT(10), "
                      "ycord FLOAT(10), labels INT, FOREIGN KEY (config_id) REFERENCES config(id));"
    },
    'drop': {
        'db_config': "DROP TABLE IF EXISTS config;",
        'embeddings': "DROP TABLE IF EXISTS {i};"
    },
    'fetch': {
        'user_ht': "SELECT author_id, hashtag FROM hash_link;",
        'ht_user': "SELECT hashtag, author_id FROM hash_link;",
        'user_rt_tid': "SELECT author_id, ref_tweet_id FROM retweeted;",
        'user_rt_uid': "SELECT author_id, ref_author_id FROM retweeted;",
        'labels': "SELECT * FROM {table} WHERE config_id=(SELECT id FROM config WHERE feature='{table}' "
                  "ORDER BY id DESC LIMIT 1);"
    },
    'store': {
        'config_id': "INSERT INTO config (feature) VALUE ({feature});",
        'fetch_id': "SELECT id FROM config ORDER BY ID DESC LIMIT 1;",
        'row': "INSERT INTO {feature} VALUES (%s,%s,%s,%s,%s);"
    },
}