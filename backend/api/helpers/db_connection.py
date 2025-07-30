import psycopg2
import numpy as np
from datetime import datetime

DB_NAME = "db_goods_prediction"
DB_USER = "postgres"
DB_PASS = "postgres"
DB_HOST = "127.0.0.1"
DB_PORT = 5434
TB_PREDICTION= "prediction"

class QueryMember:
    def get_db_connection(self):
        try:
            conn = psycopg2.connect(database=DB_NAME, user= DB_USER, 
                                    password= DB_PASS, host= DB_HOST, port=DB_PORT)
            return conn
        except Exception as e:
            raise e
        
    def create_new_prediction(self, product:str|None, score:float|None, image:str|None):
        try:
            conn= self.get_db_connection()
            cursor= conn.cursor()
            inserted_query= """
            INSERT INTO prediction (production, score, image)
            VALUES (%s, %s, %s)
                            """
            cursor.execute(inserted_query, (product, score, image))
            conn.commit()
            cursor.close()
            conn.commit()
            return True
        except Exception as e:
            raise e