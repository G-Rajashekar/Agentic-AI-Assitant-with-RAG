from datetime import datetime
import requests
import json
from Utilities.utilities import log
from rag import get_top_k_chunks

def get_current_time():

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_city_weather(city):

    url=f"https://wttr.in/{city}?format=3"
    response=requests.get(url,timeout=10)
    response.raise_for_status()
    return response.text

def get_user_city():

    return "khammam"

def search_knowledge_base(query):
    results=get_top_k_chunks(query,2)

    log(f" knowledge Base search: {query}")

    for r in results:
        log(f"  {r["score"]:.2f}-{r["source"]}#{r["chunk_id"]}")

    blocks=[]

    for r in results:
        source=r["source"]
        chunk_id=r["chunk_id"]
        text=r["text"]

        header=f"[{source} #{chunk_id}]"
        block=f"{header}\n{text}"
        blocks.append(block)

    return "\n\n".join(blocks)


Tools=[
    {
        "type":"function",
        "function":{
            "name":"get_current_time",
            "description":"Return the current date and time in a formatted string",
            "parameters":{"type":"object","properties":{}}
        }
    },
    {
        "type":"function",
        "function":{
            "name":"get_city_weather",
            "description":"Returns the current weather for a given city",
            "parameters":{
                "type":"object",
                "properties":{
                    "city":{
                        "type":"string",
                        "description":"The name of the city to get the weather for"
                    }
                },
                "required":["city"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"get_user_city",
            "description":"Returns the user current location city",
            "parameters":{"type":"object","properties":{}}
        }
    },
    {
        "type":"function",
        "function":{
            "name":"search_knowledge_base",
            "description":"Searches the zorvex technologies company knowledge base:leva,medical,hr,hiring,tech,report and travel expenses policies,use it for any question about company policies or documents. return the most relevant passages with their sources",
            "parameters":{
                "type":"object",
                "properties":{
                    "query":{
                        "type":"string",
                        "description":"The users question, rewritten as a short,self-contained search query",
                    }
                },
                "required":["query"]
            }
        }
    }
]

FUNCTION_NAME_TO_FUNCTION_MAP={
    "get_current_time":get_current_time,
    "get_city_weather":get_city_weather,
    "get_user_city":get_user_city,
    "search_knowledge_base":search_knowledge_base
}


def run_tool(tool_call):
    function_name=tool_call.function.name
    function_args=json.loads(tool_call.function.arguments or "{}")
    log(f" {function_name}({function_args})")
    result=FUNCTION_NAME_TO_FUNCTION_MAP[function_name](**function_args)
    log(f"  {result}")

    return str(result)