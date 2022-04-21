import mysql.connector
import pandas as pd


class Storage:
    def __init__(self, db, creds):
        self.creds = creds
        self.database = db
        self.queries = {
            'user_ht': "SELECT author_id, hashtag FROM hash_link;",
            'ht_user': "SELECT hashtag, author_id FROM hash_link;",
            'user_rt_tid': "SELECT author_id, ref_tweet_id FROM retweeted;",
            'user_rt_uid': "SELECT author_id, ref_author_id FROM retweeted;",
        }

    def __repr__(self):
        return f"database={self.database}"

    def connect(self):
        return mysql.connector.connect(
            host=self.creds["host"],
            user=self.creds["user"],
            passwd=self.creds["passwd"],
            database=self.database
        )

    def create_all(self):
        con = self.connect()
        cur = con.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS config (id INT AUTO_INCREMENT PRIMARY KEY, feature VARCHAR(20));")
        for i in self.queries.keys():
            cur.execute(f"CREATE TABLE IF NOT EXISTS {i} "
                        f"(config_id INT, feature VARCHAR(32), xcord FLOAT(10), ycord FLOAT(10), labels INT, "
                        f"FOREIGN KEY (config_id) REFERENCES config(id));")
        con.commit()
        con.close()
        return None

    def drop_all(self):
        con = self.connect()
        cur = con.cursor()
        for i in self.queries.keys():
            cur.execute(f"DROP TABLE IF EXISTS {i};")
        cur.execute("DROP TABLE IF EXISTS config;")
        con.commit()
        con.close()
        return None

    def fetch(self, feature):
        con = self.connect()
        cur = con.cursor()
        q = self.queries.get(feature)
        cur.execute(q)
        data = cur.fetchall()
        df = pd.DataFrame([(str(i), str(j)) for i, j in data], columns=['author_id', 'feature'])
        df = df.groupby(['author_id', 'feature']).size().reset_index()
        df.columns = ['author_id', 'feature', 'count']
        con.close()
        return df

    def fetch_labels(self, table):
        con = self.connect()
        cur = con.cursor()
        cur.execute(f"SELECT * FROM {table} WHERE config_id=(SELECT id FROM config WHERE feature='{table}' ORDER BY id DESC LIMIT 1);")
        data = cur.fetchall()
        df = pd.DataFrame(data, columns=['config_id', 'u_id', 'xcord', 'ycord', 'label'])
        df.drop(labels='config_id', axis=1, inplace=True)
        con.close()
        return df

    def fetch_annotations(self):
        con = self.connect()
        cur = con.cursor()
        cur.execute("SELECT author_id FROM annotations;")
        data = cur.fetchall()
        con.close()
        return [i[0] for i in data]

    def execute(self, query):
        con = self.connect()
        cur = con.cursor()
        cur.execute(query)
        data = cur.fetchall()
        con.close()
        return data

    def save_to_db(self, feature, df):
        con = self.connect()
        cur = con.cursor()
        cur.execute("INSERT INTO config (feature) VALUE (%s);", (feature,))
        cur.execute("SELECT id FROM config ORDER BY ID DESC LIMIT 1;")
        config_id = cur.fetchone()[0]
        data = df.to_numpy()
        for row in data:
            cur.execute(f"INSERT INTO {feature} VALUES (%s,%s,%s,%s,%s);",
                        (config_id, row[0], round(row[1], 4), round(row[2], 4), row[3]))
        con.commit()
        con.close()
        return None
