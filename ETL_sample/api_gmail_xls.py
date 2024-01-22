import os.path
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
import base64
# send attachment
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# EMAIL
# If modifying these scopes, delete the file token.json.
SCOPES = ['https://mail.google.com/']
def send_email(emails, subject, email_body, file_name):
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(r'ALERTAS/credentials/token.json'):
        creds = Credentials.from_authorized_user_file(r'ALERTAS/credentials/token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                r'ALERTAS/credentials/credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(r'ALERTAS/credentials/token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        # Call the Gmail API
        service = build('gmail', 'v1', credentials=creds)

        # Create the email content
        message = MIMEMultipart()

        message['To'] = emails
        message['From'] = 'Data Science'
        message['Subject'] = subject

        # Attach your file
        file_path = r'ALERTAS/archivos/'+file_name
        file_name = file_name

        # part = MIMEBase('application', 'octet-stream') # csv
        part = MIMEBase('application', 'vnd.openxmlformats-officedocument.spreadsheetml.sheet') #xlsx
        part.set_payload(open(file_path, 'rb').read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="{file_name}"')
        message.attach(part)

        # Encode the message as bytes
        text = MIMEText(email_body)
        message.attach(text)
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        message = (service.users().messages().send(userId='me', body={'raw': raw_message}).execute())
        print(f"Message sent: {message['id']}")

    except HttpError as error:
        # send message
        message = (service.users().messages().send
                        (userId="me", body='Error').execute())
        print(F'Message Id: {message["id"]}')

    return message
