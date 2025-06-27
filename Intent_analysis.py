import asyncio
import typing
import dotenv 
dotenv.load_dotenv()
import os
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

import typing
import asyncio
import json
from datetime import datetime
import time
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import Union
import datetime as dt

now = dt.datetime.now()

class ReminderEntities(BaseModel):
    task: typing.Optional[str] = Field(None, description="Task name to remind about")
    datetime: typing.Optional[str] = Field(None, description="standard date and time format, e.g. 2025-06-13T23:59:00")

class TimerEntities(BaseModel):
    duration: typing.Optional[str] = Field(None,description="time duration for timer in hh:mm:ss e.g. 00:36:00")
    task_name: typing.Optional[str] = Field(None,description="task name")

class AssignTaskEntities(BaseModel):
    person: typing.Optional[str] = Field(None, description="person name")
    deadline: typing.Optional[str] = Field(None, description="date and time in standard format e.g. 2025-06-13T23:59:00")
    work: typing.Optional[str] = Field(None, description="task name to be assigned to")

class GreetingEntities(BaseModel): pass
class UnknownEntities(BaseModel): pass

INTENT_FIELDS = {
    "reminder": ["task", "datetime"],
    "timer": ["duration", "task_name"],
    "assignTask": ["person", "deadline", "work"],
    "greeting": [],
    "unknown": []
}

# 2. Output model (entities as Union)
class IntentResponse(BaseModel):
    intent: str
    entities: Union[
        ReminderEntities,
        TimerEntities,
        AssignTaskEntities,
        GreetingEntities,
        UnknownEntities
    ]
    finished: bool
    statement: str

# 3. Agent configuration
intent_agent = Agent(
    model="google-gla:gemini-2.0-flash",
    result_type=IntentResponse,
    instructions=f"""
You are an intent recognition agent and conversational assistant.

For **any** user input:
1. Identify the intent.
2. Extract **entities** required for that intent. (Use empty/null for missing ones.)
3. Determine **finished** (True if all required entities are present).
4. **Generate a clear, natural, conversational "statement"** responding to the user based on the detected intent and extracted information.

Example format:

{{
  "intent": "reminder",
  "entities": {{
    "task": "visit doctor",
    "datetime": "2025-06-13T23:59:00"
  }},
  "finished": True,
  "statement": "Alright, I've set a reminder for you to visit the doctor at 8:33 on 14/06/2025."
}}

Supported intents: {INTENT_FIELDS}

Current datetime: {now}
Date format: dd-mm-yyyy. Time format: hh:mm:ss.
"""
)

async def process_request(user_input: str) -> dict:
    start_time = time.time()
    request_datetime = datetime.now().isoformat()

    try:
        result = await intent_agent.run(user_input)
        processing_time = time.time() - start_time

        return {
            "intent": result.output.intent,
            "entities": result.output.entities.model_dump(),
            "finished": result.output.finished,
            "statement": result.output.statement,
            "datetime_of_request": request_datetime,
            "raw_input": user_input,
            "processing_time": round(processing_time, 3),
            "confidence": 0.95,
            "agent_version": "gemini-2.0-flash",
            "input_summary": result.output.intent + ": " + json.dumps(result.output.entities.model_dump())
        }

    except Exception as e:
        return {
            "error": str(e),
            "datetime_of_request": request_datetime,
            "processing_time": round(time.time() - start_time, 3)
        }

async def main():
    # user_input="assign task of implementing logic to Shubham that should be finished till midnight today"
    # user_input = "hey set a timer for 67 minutes from now on for my workout"
    # user_input = "hey assign this task to today for its completion "
    # user_input = "hey assign this task to him for its completion after this hour "
    # user_input = "hey assign this task of making agents to scarab for its completion till tomorrow "
    # user_input="set a reminder to make an appointment to doctor"
    user_input=input("You: ")
    result = await process_request(user_input)

    if "error" in result:
        print("Agent error:", result["error"])
        return

    if result["finished"]:
        print(result)
    else:
        while not result["finished"]:
            print(f"Agent: {result['statement']} \nYou: ", end="")
            datainput = result["raw_input"] + ". " + input()
            newresult = await process_request(datainput)
            if "error" in newresult:
                print("Agent error:", newresult["error"])
                return
            result = newresult
            # data=json.dumps(result, indent=2)
            # json_data=json.loads(data)
            # print(json_data)
            # print(type(result))
            # print(type(data))
            # print(type(json_data))
            print(result)

if __name__ == "__main__":
    asyncio.run(main())
