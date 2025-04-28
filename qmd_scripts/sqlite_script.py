import pandas as pd
import numpy as np
import plotly.express as px
import plotnine as pn
import seaborn as sns
import matplotlib.pyplot as plt
# import duckdb as duck
import sqlite3
# from sqlalchemy import create_engine, text

#duckdb
engine = create_engine('duckdb:///database/f1_gp.duckdb', connect_args = {'read_only': False, 'config': {'threads': 4}})
conn = engine.connect()

# result = conn.execute(text('SELECT * FROM cardata LIMIT 10;'))
# for row in result:
#     print(row)

# shows all the tables in the database
conn.execute(text("SHOW TABLES;")).fetchall()

conn.execute(text('SELECT * FROM pos LIMIT 10;')).fetchall()
conn.execute(text('SELECT RPM, Speed, nGear, Throttle, Brake, driver_number, session, time_sec FROM cardata LIMIT 10;')).fetchall()

conn.close()


# SQLite

# Connecting to Database
conn = sqlite3.connect('database/f1_gp.sqlite')
curs = conn.cursor()

for row in curs.execute('SELECT * FROM weather LIMIT 10;'):
    print(row)