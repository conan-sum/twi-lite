import mysql.connector
import json


def connect(db: str):
    creds = json.load(open("twilite/utils/creds.json"))
    return mysql.connector.connect(
        host=creds["host"],
        user=creds["user"],
        passwd=creds["passwd"],
        database=db
    )