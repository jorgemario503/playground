# import libraries
import pyodbc
import pandas as pd

def sql_pandas(server, database, query):
    # SQL SET UP
    with open(r'c:\Users\Jorge.Melendez\Documents\powerbi.txt') as f:
        pwd = f.readlines()
    server = server
    database = database 
    username = 'rpt_powerbi'
    password = pwd[0]
    cnxn = pyodbc.connect('DRIVER={SQL Server Native Client 11.0};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
    data = pd.read_sql(query, cnxn)

    return data
