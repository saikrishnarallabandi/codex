from invoke_llm import InvokeGPT
import json
import sys 
import pandas as pd
import ast
import asyncio
import traceback

llm = InvokeGPT()
f = open('./log.out', 'w')
f.write("Starting call_gpt.py" + '\n')
f.close() 

def wrap_tool_definition(tool):
    if tool.get("type") == "function" and "name" in tool:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
        }
    return tool

async def error_event_generator(category, message, tb=None):
    yield {
        "type": "error",
        "category": category,
        "message": message,
        **({"traceback": tb} if tb else {})
    }

async def main():
    input_data = sys.stdin.read()
    #print(f"stdin received: ", input_data, file=sys.stderr)
    f = open('log.in', 'w')
    f.write("input is " + input_data)
    f.close()
    f = open('./log.out', 'w')
    f.close()
    #input_data_json = pd.DataFrame(input_data)
    #input_data_json.to_json('log.json', indent=4)
    messages = json.loads(input_data)
    
    chat_input = repr(messages.get('input'))
    chat_input_ast = ast.literal_eval(chat_input)
    with open('./log.out', 'a') as f:
        f.write('AST helped ' + '\n')
        
    instructions = messages.get('instructions')
    user_content = chat_input_ast[0]['content'][0]['text']   
    tools = messages.get('tools')
    tools = [wrap_tool_definition(tool) for tool in tools]
    
    chat_messages = [
        {
            "role": "system",
            "content": instructions
        },
        {
            "role": "user",
            "content": user_content
        }
    ] 

    f = open('./log.out', 'w')
    f.write(str(messages))
    f.close()

    try:
        response_gen = llm.get_response(chat_messages, tools=tools, stream=True)
    except Exception as e:
        tb = traceback.format_exc()
        async for event in error_event_generator("internal", str(e), tb):
            print(json.dumps(event), flush=True)
        return
    try:
        # Stream output: print each chunk as a JSON event, flush after each
        if response_gen is not None:
            async for chunk in response_gen:
                print(json.dumps(chunk), flush=True)
        else:
            print(json.dumps({
                "role": "assistant",
                "type": "output_text",
                "content": {"text": "[ERROR] No response from the model."}
            }), flush=True)
    except Exception as e:
        tb = traceback.format_exc()
        async for event in error_event_generator("internal", str(e), tb):
            print(json.dumps(event), flush=True)

if __name__ == "__main__":
    asyncio.run(main())