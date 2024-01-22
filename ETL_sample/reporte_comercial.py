# import libraries
from playground.alertas_ETL.api_sql_pandas import sql_pandas
from playground.alertas_ETL.api_gmail_xls import send_email
from emoji import emojize

# GET YESTERDAY
from datetime import date
from datetime import timedelta
today = date.today()
yesterday = today - timedelta(days = 1)

# SEND EMAILS
emails_dev='emails@emails.com'
emails='email1@email.com,email2@email.com'
emails_final=emails+','+emails_dev

# REPORTES PARA DIVISIÓN COMERCIAL
query_desem_pagos ='''
select *
from DWHOPTIMA.bi.rpt_comercial_pagos_desembolsos
'''

# GET REPORTS
import time
start = time.time()
try:
    data1 = sql_pandas('10.3.11.27', 'DWHOPTIMA', query_desem_pagos)
    print('sql success')
    data1.to_excel(r'c:\Users\Jorge.Melendez\Documents\Python\ALERTAS\archivos\desembolsos_pagos.xlsx', index=False)
    print('excel success')
    no_desembolsos = len(data1[data1['tipo_transaccion']=='desembolso'])
    no_pagos = len(data1[data1['tipo_transaccion']=='pago'])
    subject_desem_pagos = emojize(':loudspeaker:')+'ADS: Reporte de desembolsos y pagos'
    body_desem_pagos = 'Listado de referencias con desembolsos y pagos (Número de desembolso:'+str(no_desembolsos)+'- Número de pagos:'+str(no_pagos)
    send_email(emails_final,subject_desem_pagos,body_desem_pagos,file_name='desembolsos_pagos.xlsx')

except:
    print('reporte_listas-cobro.py error on DWHOPTIMA.bi.rpt_comercial_pagos_desembolsos SQL read')
    send_email(emails_dev,'ERROR ON SQL REPORTE_COMERCIAL.PY','Error: no SQL read on DWHOPTIMA.bi.rpt_comercial_pagos_desembolsos for: '+str(yesterday),'desembolsos_pagos.xlsx')

end = time.time()
print('report time:',end-start)