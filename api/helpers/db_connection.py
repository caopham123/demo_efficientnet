import psycopg2
import numpy as np
from datetime import datetime

DB_NAME = "db_goods_prediction"
DB_USER = "postgres"
DB_PASS = "postgres"
DB_HOST = "127.0.0.1"
DB_PORT = 5434


class QueryMember:
    def get_db_connection(self):
        try:
            conn = psycopg2.connect(database=DB_NAME, user= DB_USER, 
                                    password= DB_PASS, host= DB_HOST, port=DB_PORT)
            return conn
        except Exception as e:
            raise e