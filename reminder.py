# Currently only works for indian time zone, add functionality for multiple timezones

import datetime
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_calendar_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('calendar', 'v3', credentials=creds)

def create_event(service, summary, start_datetime):
    # Do NOT localize or attach tzinfo
    end_datetime = start_datetime + datetime.timedelta(hours=1)

    event = {
        'summary': summary,
        'start': {
            'dateTime': start_datetime.isoformat(),  # naive datetime (no +05:30)
            'timeZone': 'Asia/Kolkata',
        },
        'end': {
            'dateTime': end_datetime.isoformat(),
            'timeZone': 'Asia/Kolkata',
        },
    }
    print(f"{start_datetime} {start_datetime.isoformat()}")

    event_result = service.events().insert(calendarId='primary', body=event).execute()
    print(f"✅ Event created: {event_result.get('htmlLink')}")

def main():
    service = get_calendar_service()

    print("Enter Event Details:")
    name = input("Event name: ").strip()
    date_input = input("Date (YYYY-MM-DD): ").strip()
    time_input = input("Time (HH:MM, 24-hour format): ").strip()

    # Fixed the timezone error by hardcoding

    hour = int(time_input.split(':')[0]) + 5
    minute = int(time_input.split(':')[1]) + 30

    def increase_date_by_one(date_str):
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        
        next_day = date_obj + datetime.timedelta(days=1)
        
        return next_day.strftime("%Y-%m-%d")

    if minute >= 60:
        hour += 1
        minute -= 60

    if hour >= 24:
        hour -= 24
        date_input = increase_date_by_one(date_input)

    # Changing to two digit compulsorily
    
    # Logic 
    '''
    hour_str = str(hour)
    minute_str = str(minute)

    if not(hour // 10 >= 1):
        hour_str = '0' + str(hour)

    if not(minute // 10 >= 1):
        minute_str = '0' + str(minute)
    '''
    # Readable code
    hour_str = f"{hour:02}"
    minute_str = f"{minute:02}"

    time_input = f"{hour_str}:{minute_str}"
    
    #---x--- Time zone Fix ---x---

    try:
        # Do NOT localize here — keep it naive
        start_datetime = datetime.datetime.strptime(f"{date_input} {time_input}", "%Y-%m-%d %H:%M")
    except ValueError:
        print("❌ Invalid date or time format. Please try again.")
        return

    create_event(service, name, start_datetime)

if __name__ == '__main__':
    main()

