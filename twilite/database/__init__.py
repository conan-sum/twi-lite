"""
An object that handles all database related operations
"""
import mysql.connector
import pandas as pd
from .query import default_queries


class Storage:
    def __init__(self, db, creds, queries=default_queries):
        self.creds = creds
        self.database = db
        self.queries = queries

    def __repr__(self):
        return f"Storage(database={self.database})"

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
        sql = self.queries['create']
        cur.execute(sql['db_config'])
        for i in self.queries.keys():
            cur.execute(sql['embeddings'].format(i=i))
            cur.execute(sql['embeddings']+'_subcluster'.format(i=i))
        con.commit()
        con.close()
        return None

    def drop_all(self):
        con = self.connect()
        cur = con.cursor()
        sql = self.queries['drop']
        for i in self.queries.keys():
            cur.execute(sql['embeddings'].format(i=i))
        cur.execute(sql['db_config'])
        con.commit()
        con.close()
        return None

    def fetch(self, feature):
        con = self.connect()
        cur = con.cursor()
        sql = self.queries['fetch']
        cur.execute(sql[feature])
        data = cur.fetchall()
        df = pd.DataFrame([(str(i), str(j)) for i, j in data], columns=['author_id', 'feature'])
        df = df.groupby(['author_id', 'feature']).size().reset_index()
        df.columns = ['author_id', 'feature', 'count']
        con.close()
        return df

    def fetch_labels(self, table):
        con = self.connect()
        cur = con.cursor()
        sql = self.queries['fetch']
        cur.execute(sql['labels'].format(table=table))
        data = cur.fetchall()
        df = pd.DataFrame(data, columns=['config_id', 'u_id', 'xcord', 'ycord', 'label'])
        df.drop(labels='config_id', axis=1, inplace=True)
        con.close()
        return df

    def save_to_db(self, feature, df):
        con = self.connect()
        cur = con.cursor()
        sql = self.queries['store']
        cur.execute(sql['config_id'].format(feature=feature))
        cur.execute(sql['fetch_id'])
        config_id = cur.fetchone()[0]
        data = df.to_numpy()
        for row in data:
            cur.execute(sql['row'].format(feature=feature),
                        (config_id, row[0], round(row[1], 4), round(row[2], 4), row[3]))
        con.commit()
        con.close()
        return None
