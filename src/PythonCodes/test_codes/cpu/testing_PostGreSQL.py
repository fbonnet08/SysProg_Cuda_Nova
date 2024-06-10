import psycopg2

conn = psycopg2.connect(
    database="MolRefAnt_DB_PostGreSQL",
    user="frederic",
    password="postgre23",
    host='127.0.0.1',
    port='5432'
)

cursor = conn.cursor()

cursor.execute("select * from \"MolRefAnt_DB_PostGreSQL\".\"MolRefAnt_DB\".analytics_data;")
data_analytics_data = cursor.fetchone()

cursor.execute("select * from \"MolRefAnt_DB_PostGreSQL\".\"MolRefAnt_DB\".datetable;")
data_datetable = cursor.fetchone()

print("data_analytics_data ---> ", data_analytics_data)
print("data_datetable      ---> ", data_datetable)

conn.close()