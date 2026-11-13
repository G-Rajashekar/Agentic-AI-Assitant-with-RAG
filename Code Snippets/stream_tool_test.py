import os
from types import SimpleNamespace
from openai import OpenAI
from dotenv import load_dotenv
from tools import Tools

load_dotenv()

client=OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

stream=client.chat.completions.create(
    mode=os.getenv("MODEL"),
    messages=[
        {
            "role":"user",
            "content":"what is the current time and weather in my location"
        }
    ],
    tools=Tools,
    stream=True
)

calls={}

for chunk in stream:
    if  not chunk.choices:
        continue 

    delta=chunk.choices[0].delta

    for tc in (delta.tool_calls or []):

        obj=calls.setdefault(
            tc.index,
            SimpleNamespace(id=None,function=SimpleNamespace(function_name="",function_args=""))
        )

        if tc.id:
            obj.id=tc.id

        if tc.function and tc.function.name:
            obj.function.function_name+=tc.function.name

        if tc.function and tc.function.arguments:
            obj.function.function_args+=tc.function.arguments

calls_list=[]

for key in sorted(calls):
    calls_list.append(calls[key])