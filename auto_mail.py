import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# List available models (optional)
# print([m.name for m in genai.list_models()])

# Use a correct, supported model name
model = genai.GenerativeModel('models/gemini-2.5-flash')  # <- corrected name with "models/" prefix

word_limit = 50
tone = "proffesional"
subject = "Testing mail 1"
description = "a mail just for testing of auto_mail_app"
receiver = "" # Target
sender = "" # Sender's name
to = "" # Target's email

if(word_limit == 0):
    word_limit = "suitable"

prompt = f"write a mail to {receiver}, subject: {subject}, tone: {tone}, word limit: {word_limit}, sender:{sender}, description of mail:{description}"

# response = model.generate_content(f"Write a mail in about {word_limit} words, regarding {subject}")
response = model.generate_content(prompt)


# print(response.text)

import os.path
import base64
from email.message import EmailMessage
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying these SCOPES, delete token.json
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def gmail_authenticate():
    creds = None
    # Load token if it exists
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If not valid, refresh or authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save token for next time
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def create_message(to, subject, message_text):
    message = EmailMessage()
    message.set_content(message_text)
    message['To'] = to
    message['Subject'] = subject

    # Encode the message
    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {'raw': encoded_message}

def send_email(service, user_id, message):
    try:
        sent_message = service.users().messages().send(userId=user_id, body=message).execute()
        print(f"Message sent! ID: {sent_message['id']}")
        return sent_message
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == '__main__':
    creds = gmail_authenticate()
    service = build('gmail', 'v1', credentials=creds)

    message = create_message(to, subject, response.text)
    send_email(service, 'me', message)
