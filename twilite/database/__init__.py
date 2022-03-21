import mysql.connector
from twilite.database.queries import query
import pandas as pd


class Storage:
    def __init__(self, db, creds):
        def connect(database):
            return mysql.connector.connect(
                host=creds["host"],
                user=creds["user"],
                passwd=creds["passwd"],
                database=database
            )
        self.con = connect(db)

    def fetch(self, feature):
        cur = self.con.cursor()
        q = query(feature)
        cur.execute(q)
        data = cur.fetchall()
        df = pd.DataFrame([(str(i), str(j)) for i, j in data], columns=['author_id', feature])
        df = df.explode(feature)
        df = df.groupby(['author_id', feature]).size().reset_index()
        df.columns = ['author_id', 'feature', 'count']

    def find_config(self):
        pass

    def store_config(self):
        pass

    def save_to_db(self, feature, df):
        cur = self.con.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {feature};")
        cur.execute(f"CREATE TABLE {feature} (author_id VARCHAR(50), xcord FLOAT(10), ycord FLOAT(10), labels INT);")
        data = df.to_numpy()
        for row in data:
            cur.execute(f"INSERT INTO {feature} VALUES (%s,%s,%s);", (row[0], round(row[1], 4), round(row[2], 4), row[3]))
        self.con.commit()
        return None
