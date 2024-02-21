# Import libraries
import requests
import json
import pandas as pd
import psycopg2
import os
import sys

# Define path of files (default is to get directory where this file is being invoked)
location = os.path.dirname(os.path.abspath(sys.argv[0]))

# FUNCTION TO GET DATA FROM BLS API AND SAVE IT AS TXT FILES
def get_data(series_id, end_year, start_year, **kwargs):
    """
    Pass in a list of BLS timeseries to fetch data and return the series in JSON format. 
    Arguments can also be provided as kwargs:
        - registrationKey (API key from BLS website)
        - catalog (True or False)
        - calculations (True or False)
        - annualaverage (True or False)
    If the registrationKey is not passed in, you'll face request restrictions.
    """
    # Set up API parameters and update using kwargs
    headers = {'Content-type': 'application/json'}
    api_url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'
    data = {"seriesid": series_id
               , "endyear": end_year
               , "startyear": start_year
              }
    # Update data parameter with added kwargs
    data.update(kwargs)
    data = json.dumps(data)
    # Fetch the response from the BLS API
    response = requests.post(api_url, data=data, headers=headers)
    json_data = json.loads(response.text)
    # Save the output as a txt file
    txt_file_location = []
    for series in json_data['Results']['series']:
        df = pd.DataFrame(columns=["series_id","year","period","value","footnotes"])
        seriesID = series['seriesID']
        if series['data']:
            for item in series['data']:
                year = item['year']
                period = item['period']
                value = int(item['value'])
                footnotes=""
                for footnote in item['footnotes']:
                    if footnote:
                        footnotes = footnotes + footnote['text'] + ','
                if 'M01' <= period <= 'M12':
                    df.loc[len(df)] = [seriesID,year,period,value,footnotes[0:-1]]
            # File destination
            file_location = location+'/'+seriesID + '_' + str(start_year) + '_' + str(end_year) + '.txt'
            txt_file_location.append(file_location)
            df.to_csv(file_location, sep='\t', index=False)
    return txt_file_location

# FUNCTION TO GET THE VALUES OF THE VARIABLES
def find_value(key):
    """
    Fetch the variables.txt file and return the needed variables to run this script. 
    PostgreSQL:
        - database
        - hostname
        - username
        - password
    CES API:
        - api_key
    """
    # File import_config.txt location (default is same as script directory)
    import_config = location+'/'+"import_config.txt"
    try:
        with open(import_config) as f:
            for line in f:
                k, v = map(str.strip, line.split("="))
                if k == key:
                    return v
    except:
        print('No variable value returned')
        return None

# MAKE THE REQUEST AND UPLOAD TO PostgreSQL
# Define database connection string and cursor
conn = psycopg2.connect(dbname=find_value('database')
                        , host=find_value('hostname')
                        , user=find_value('username')
                        , password=find_value('password')
                        )
cursor = conn.cursor()
# Drop target table if exists
cursor.execute('''
DROP TABLE IF EXISTS ces_schema.ces_data;
CREATE TABLE ces_schema.ces_data (
seriesid varchar(100)
, year integer
, period varchar(3)
, value integer
, footnotes varchar(1000)
);
''')
# Commit the transaction
conn.commit()

# CES series and API Key
series_dict = {'CES9000000010': 'Women employees, thousands, government, seasonally adjusted'
               , 'CES0500000001': 'All employees, thousands, total private, seasonally adjusted'
               , 'CES0500000006': 'Production and nonsupervisory employees, thousands, total private, seasonally adjusted'
              }
api_key = find_value('api_key')

# Run the function using a loop that loads the files to PostgreSQL and substracts years for the next iteration
# Adjust variables as needed
end_year = 1924
loops = 100
file_names = []
for i in range(1,loops):
    # start_date is hardcode with a subtraction of 9 years 
    file_names = get_data(list(series_dict.keys()), end_year, end_year-9, registrationKey=api_key)
    # check if there are values for the series
    if file_names:
        # Load data into PostgreSQL tables
        for file in file_names:
            # Open the file as a stream
            with open(file, 'rb') as f:
                next(f) # Skipping header row
                cursor.copy_expert("COPY ces_schema.ces_data FROM STDIN", f) 
            conn.commit()
        # Substract 10 years to the end_year parameter value
        end_year -= 10
    # breaks the loop if the response has no values
    else:
        print('No more series')
        break
# Close the connection
conn.close()
