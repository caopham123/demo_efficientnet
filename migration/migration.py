import psycopg2
from datetime import datetime

DB_NAME = "db_goods_prediction"
DB_USER = "postgres"
DB_PASS = "postgres"
DB_HOST = "127.0.0.1"
DB_PORT = 5434

try:
    conn = psycopg2.connect(database=DB_NAME, user= DB_USER, 
                            password= DB_PASS, host= DB_HOST, port=DB_PORT)
    cur= conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS prediction( 
                    id SERIAL PRIMARY KEY,
                    production TEXT NOT NULL,
                    score NUMERIC NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW()),
                    modified_at TIMESTAMP WITH TIME ZONE,
                    isDelete BOOLEAN DEFAULT FALSE,
                    image TEXT
                ); """)
    print("Created table member successfully!")

    
    conn.commit()
    cur.close()
    conn.close()
except Exception as e:
    raise e