import schedule
import subprocess
import time

print('Scheduler ETL, reports and alerts: ON |',str(time.ctime()))

# REPORTES
def run_etl_api():
    subprocess.run(['python', r'c:\Users\Jorge.Melendez\Documents\Python\ETL\etl_helpdesk.py'])
    print('INSERT bi.master_helpdesk completed on:',str(time.ctime()))
    subprocess.run(['python', r'c:\Users\Jorge.Melendez\Documents\Python\ETL\etl_cosmos.py'])
    print('INSERT multiple cosmos tables completed on:',str(time.ctime()))

def run_recuperaciones():
    subprocess.run(['python', 'reporte_recuperaciones-listas.py'])
    print('Reportes de listas de cobros enviados:',str(time.ctime()))

def run_comercial():
    subprocess.run(['python', 'reporte_comercial.py'])
    print('Reportes de comercial enviados:',str(time.ctime()))

# ALERTAS
def run_alertas():
    subprocess.run(['python', 'alerta_opto.py'])
    print('Alertas de opto enviadas:',str(time.ctime()))

# SCHEDULES
def run_daily_reports():
    run_recuperaciones()
    run_comercial()
    run_alertas()

def run_daily_etl():
    run_etl_api()

# JOBS
for i in ["02:00"]:
    schedule.every().monday.at(i).do(run_daily_etl)
    schedule.every().tuesday.at(i).do(run_daily_etl)
    schedule.every().wednesday.at(i).do(run_daily_etl)
    schedule.every().thursday.at(i).do(run_daily_etl)
    schedule.every().friday.at(i).do(run_daily_etl)
    schedule.every().saturday.at(i).do(run_daily_etl)
    schedule.every().sunday.at(i).do(run_daily_etl)

for i in ["06:15"]:
    schedule.every().monday.at(i).do(run_daily_reports)
    schedule.every().tuesday.at(i).do(run_daily_reports)
    schedule.every().wednesday.at(i).do(run_daily_reports)
    schedule.every().thursday.at(i).do(run_daily_reports)
    schedule.every().friday.at(i).do(run_daily_reports)
    schedule.every().saturday.at(i).do(run_daily_reports)

while True:
    schedule.run_pending()
    time.sleep(600) # wait 10 min before checking the schedule again
