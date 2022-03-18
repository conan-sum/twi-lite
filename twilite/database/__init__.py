class Storage:
    def __init__(self, connection):
        self.con = connection

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
