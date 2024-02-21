# POSTGRESQL SERVER PROJECT

## Description
Set up a PostgreSQL Server with 2 endpoints:
    (1) api_women_in_government
    (2) api_production_nonsupervisory_ratio

## Steps
*Step 1*: Create database and schema (you can either create it using psql or via a postgreSQL UI)
- CREATE DATABASE ces_time_series;
- CREATE SCHEMA ces_schema;

*Step 2*: Run python file import_ces_data.py to upload CES data to ces_schema.ces_data
- Before running the python file you need to update import_config.txt with your postgreSQL username, password and BLS API key (you can also adjust the name of your database and the hostname)
- You can either run the python file directly OR from psql/postgreSQL by running the file ces_data.sql (which also includes the next steps 3 and 4), just make sure that you have the PL/Python extension _plpython3u_ and that the file directory is updated.

*Step 3*: Create views in ces_schema that will serve as the REST endpoints.
- api_women_in_government filters the women employees time series ID
- api_production_nonsupervisory_ratio calculates the ratio of nonsupervisory employees by supervisory employees (the rest of the employees not considered nonsupervisory)

*Step 4*: Create role for anonymous web requests and a dedicated role to connect to the database

*Step 5*: Run the server postgrest tutorial.conf (if installed from a package manager)
- Default db-uri is "postgres://authenticator:mysecretpassword@localhost:5432/ces_time_series"
- Default schema is "ces_data"
- Default db-anon-role is "web_anon"
- Modify if needed.

## Resources
Data set explanation: https://download.bls.gov/pub/time.series/ce/ce.txt
Serie IDs explanation: https://download.bls.gov/pub/time.series/ce/ce.series
Data: https://download.bls.gov/pub/time.series/ce/
PostgREST: https://postgrest.org/en/v12/tutorials/tut0.html
