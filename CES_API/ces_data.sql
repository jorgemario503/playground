-- Step 2: PL/Python to import CES data
CREATE OR REPLACE FUNCTION api_ces_data() RETURNS void AS $$
import subprocess
subprocess.run(["python", "import_ces_data.py"])
$$ LANGUAGE plpython3u;
-- Import data using python
SELECT api_ces_data();

-- Step 3: Create views 
-- Endpoint: Evolution of women in government
CREATE OR REPLACE VIEW ces_schema.api_women_in_government AS
SELECT
	concat(year,' ',TO_CHAR(TO_DATE(right(period,2)::text,'MM'),'Month')) as date
	, value as valueInThousands
FROM ces_schema.ces_data
WHERE seriesid='CES9000000010'
;
-- Endpoint: Evolution of production and nonsupervisory employees / supervisory employees ratio
CREATE OR REPLACE VIEW ces_schema.api_production_nonsupervisory_ratio AS
SELECT
	concat(a.year,' ',TO_CHAR(TO_DATE(right(a.period,2)::text,'MM'),'Month')) as date
	, b.value/(a.value-b.value) as ratio
FROM ces_schema.ces_data a
LEFT JOIN ces_schema.ces_data b
on a.year=b.year and a.period=b.period and a.seriesid='CES0500000001' and b.seriesid='CES0500000006'
;

-- Step 4: Create roles 
-- For anonymous web requests 
CREATE ROLE web_anon nologin;
GRANT usage on schema ces_schema to web_anon;
GRANT SELECT on ces_schema.api_women_in_government, ces_schema.api_production_nonsupervisory_ratio to web_anon;
-- For connecting to database and switching to the web_anon role
CREATE ROLE authenticator noinherit login password 'mysecretpassword';
GRANT web_anon to authenticator;

