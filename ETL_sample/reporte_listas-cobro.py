# Reload libraries
from importlib import reload 
import playground.alertas_ETL.api_sql_pandas as api_sql_pandas
api_sql_pandas= reload(api_sql_pandas)
import playground.alertas_ETL.api_gmail_xls as api_gmail_xls
api_gmail_xls= reload(api_gmail_xls)

# import libraries
from playground.alertas_ETL.api_sql_pandas import sql_pandas
from playground.alertas_ETL.api_gmail_xls import send_email
from emoji import emojize

# GET YESTERDAY
from datetime import date
today = date.today()

# SEND EMAIL
emails_dev='emails@emails.com'
emails='email1@email.com,email2@email.com'
emails_final=emails+','+emails_dev

# LISTAS DE COBROS
query_premora = '''
select *
from DWHOPTIMA.bi.rpt_recuperaciones_premora
'''
query_par1_30 = '''
select *
from DWHOPTIMA.bi.rpt_recuperaciones_mora130
'''
query_par30 = '''
select *
from DWHOPTIMA.bi.rpt_recuperaciones_mora30
'''
query_refi = '''
select *
from DWHOPTIMA.bi.rpt_recuperaciones_refi
'''
query_visita = '''
select *
from DWHOPTIMA.bi.rpt_recuperaciones_visita130
'''

# GET REPORTS
import time
start_premora = time.time()
try:
    data1 = sql_pandas('10.3.11.27', 'DWHOPTIMA', query_premora)
    print('SQL succesfull')
    data1.to_excel(r'c:\Users\Jorge.Melendez\Documents\Python\ALERTAS\archivos\lista_premora.xlsx', sheet_name='lista', index=False)
    print('Excel succesfull')
    saldos_premora = data1['saldo_capital'].sum()
    subject_premora = emojize(':loudspeaker:')+'ADS: Lista de cobros PREMORA'
    body_premora = f'Listado de cobros PREMORA, con estado EN GRACIA al día de hoy ({today}): {len(data1)} referencias por ${saldos_premora.astype(str)}'
    send_email(emails_final,subject_premora,email_body=body_premora,file_name='lista_premora.xlsx') 
except:
    print('reporte_listas-cobro.py error on DWHOPTIMA.bi.rpt_recuperaciones_premora SQL read')
    send_email(emails_dev,'ERROR ON SQL REPORTE LISTAS DE COBRO PREMORA',f'Error: no SQL read for DWHOPTIMA.bi.rpt_recuperaciones_premora on file reporte_listas-cobro.py for: {today}','lista_premora.xlsx')
end_premora = time.time()
print('tiempo de reporte de premora:',end_premora-start_premora)

start_mora130 = time.time()
try:
    data2 = sql_pandas('10.3.11.27', 'DWHOPTIMA', query_par1_30)
    print('SQL succesfull')
    data2.to_excel(r'c:\Users\Jorge.Melendez\Documents\Python\ALERTAS\archivos\lista_mora130.xlsx', sheet_name='lista', index=False)
    print('Excel succesfull')
    saldos = data2['saldo_capital'].sum()
    subject_par1_30 = emojize(':loudspeaker:')+'ADS: Lista de cobros PAR1-30'
    body_par1_30 = f'Listado de cobros mora 1-30 actualizada al día de hoy ({today}): {len(data2)} referencias por ${saldos.astype(str)}'
    send_email(emails_final,subject_par1_30,email_body=body_par1_30,file_name='lista_mora130.xlsx')
except:
    print('reporte_listas-cobro.py error on DWHOPTIMA.bi.rpt_recuperaciones_mora130 SQL read')
    send_email(emails_dev,'ERROR ON SQL REPORTE LISTAS DE COBRO PAR1-30',f'Error: no SQL read for DWHOPTIMA.bi.rpt_recuperaciones_mora130 on file reporte_listas-cobro.py for: {today}','lista_mora130.xlsx')
end_mora130 = time.time()
print('tiempo de reporte de mora 1-30:',end_mora130-start_mora130)

start_mora30 = time.time()
try:
    data3 = sql_pandas('10.3.11.27', 'DWHOPTIMA', query_par30)
    print('SQL succesfull')
    data3.to_excel(r'c:\Users\Jorge.Melendez\Documents\Python\ALERTAS\archivos\lista_mora30.xlsx', sheet_name='lista', index=False)
    print('Excel succesfull')
    subject_deterioro30 = emojize(':loudspeaker:')+'ADS: Lista de cobros PAR30+'
    body_deterioro30 = f'Listado de cobros de PAR30+ y rollback en el mes, con mora actualizada al día de hoy ({today}): {len(data3)} referencias'
    send_email(emails_final,subject_deterioro30,body_deterioro30,file_name='lista_mora30.xlsx') 
except:
    print('reporte_listas-cobro.py error on DWHOPTIMA.bi.rpt_recuperaciones_mora30 SQL read')
    send_email(emails_dev,'ERROR ON SQL REPORTE LISTAS DE COBRO PAR30+',f'Error: no SQL read for DWHOPTIMA.bi.rpt_recuperaciones_mora30 on file reporte_listas-cobro.py for: {today}','lista_mora30.xlsx')
end_mora30 = time.time()
print('tiempo de reporte de mora 30+:',end_mora30-start_mora30)

start_refi = time.time()
try:
    data4 = sql_pandas('10.3.11.27', 'DWHOPTIMA', query_refi)
    print('SQL succesfull')
    data4.to_excel(r'c:\Users\Jorge.Melendez\Documents\Python\ALERTAS\archivos\lista_refi.xlsx', sheet_name='lista', index=False)
    print('Excel succesfull')
    subject_refi = emojize(':loudspeaker:')+'ADS: Lista de cobros REFI'
    body_refi = f'Listado de cobros REFI (EN GRACIA y mora 1-30), actualizada al día de hoy ({today}): {len(data4)} referencias'
    send_email(emails_final,subject_refi,body_refi,file_name='lista_refi.xlsx') 
except:
    print('reporte_listas-cobro.py error on DWHOPTIMA.bi.rpt_recuperaciones_refi SQL read')
    send_email(emails_dev,'ERROR ON SQL REPORTE LISTAS DE COBRO REFI',f'Error: no SQL read for DWHOPTIMA.bi.rpt_recuperaciones_refi on file reporte_listas-cobro.py for: {today}','lista_refi.xlsx')
end_refi = time.time()
print('tiempo de reporte de refi:',end_refi-start_refi)

start_visita = time.time()
try:
    data5 = sql_pandas('10.3.11.27', 'DWHOPTIMA', query_visita)
    print('SQL succesfull')
    referencias = data5['referencia'].nunique()
    data5.to_excel(r'c:\Users\Jorge.Melendez\Documents\Python\ALERTAS\archivos\lista_visita.xlsx', sheet_name='lista', index=False)
    print('Excel succesfull')
    subject_visita = emojize(':loudspeaker:')+'ADS: Lista de cobros visita PAR1-30'
    body_visita = f'Listado de cobros visita (mora 1-30), actualizada al día de hoy ({today}): {referencias} referencias'
    send_email(emails_final,subject_visita,body_visita,file_name='lista_visita.xlsx') 
except:
    print('reporte_listas-cobro.py error on DWHOPTIMA.bi.rpt_recuperaciones_visita SQL read')
    send_email(emails_dev,'ERROR ON SQL REPORTE LISTAS DE COBRO VISITA PAR1-30',f'Error: no SQL read for DWHOPTIMA.bi.rpt_recuperaciones_visita on file reporte_listas-cobro.py for: {today}','lista_visita.xlsx')
end_visita = time.time()
print('tiempo de reporte de visita 1-30:',end_visita-start_visita)
