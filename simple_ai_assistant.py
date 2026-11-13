from openai import OpenAI
from dotenv import load_dotenv
import os
from tools import Tools,run_tool
import json
from Utilities.utilities import log
from types import SimpleNamespace
load_dotenv()

client=OpenAI(

    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

messages=[
    {
    "role":"system",
    "content":"You are a helpful and humble assistant and you work for this company. Answer general questions that is programming, science, advice from your own knowledge. Use get_current_time, get_city_weather and get_user_city only for the current time, the weather or the user city. For any question about Zervex policies, benefits, leave, medical cover, HR, hiring, tech process, incident reports or travel expenses, you must first call search_knowledge_base. Answer company questions only from the text that search_knowledge_base returns. Do not use outside knowledge for company facts. After the answer, cite the source like [file_name #chunk_id]. If the returned text does not contain the answer, say exactly like this: 'I could not find this in the knowledge base'. Keep the answers under 100 words."
    }
]

MAX_TOOL_CALLS=5

def stream_chat(history):
    stream=client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=history,
        tools=Tools,
        stream=True
    )

    full_response=""
    calls={}
    label_printed=False

    try:
        for chunk in stream:
            if not chunk.choices:
                continue 

            delta=chunk.choices[0].delta

            for tc in (delta.tool_calls or []):
                obj=calls.setdefault(
                    tc.index,
                    SimpleNamespace(id=None,function=SimpleNamespace(name="",arguments=""))
                )

                if tc.id:
                    obj.id=tc.id
                if tc.function and tc.function.name:
                    obj.function.name+=tc.function.name
                if tc.function and tc.function.arguments:
                    obj.function.arguments+=tc.function.arguments

            if delta.content:
                piece=delta.content
                if not label_printed:
                    print("Assistant: ",end="",flush=True)
                    label_printed=True
                print(piece,end="",flush=True)
                full_response+=piece

    except KeyboardInterrupt:
        print("\n{Stopped}")
        calls={}
         

    if (label_printed):
        print()

    calls_list=[]

    for key in sorted(calls):
        calls_list.append(calls[key])

    return full_response,calls_list
        



def build_tool_calls_payload(tool_calls):
   payload=[]

   for tool_call in tool_calls:
      tool_call_dict={
         "id":tool_call.id,
         "type":"function",
         "function":{
            "name":tool_call.function.name,
            "arguments":tool_call.function.arguments
         }
      }

      payload.append(tool_call_dict)

   return payload

while True:

    user_input=input("Raj: ")

    if user_input.lower() in ["exit","quit","bye"]:
        print("Exiting the assistant. GoodBye!")
        break

    messages.append({
        "role":"user",
        "content":user_input
    })

    for _ in range(MAX_TOOL_CALLS):
       

       full_response,tool_calls=stream_chat(messages)
        

       if not tool_calls:
            messages.append(
                {
                    "role":"assistant",
                    "content":full_response
                }
            )
            
            break

       messages.append(
          {
             "role":"assistant",
             "content":full_response,
             "tool_calls":build_tool_calls_payload(tool_calls)
          }
       )

       log(f"Step {_+1}: model requested {len(tool_calls)} tool calls")

       for tool_call in tool_calls:

            result=run_tool(tool_call)

            messages.append(
                {
                    "role":"tool",
                        "tool_call_id":tool_call.id,
                    "content":str(result)
                }
            )

    else:
        fall_back="sorry, I couldn't  finish within max tool call limit"
        messages.append(
            {
            "role":"assistant",
            "content":fall_back
            }
        )