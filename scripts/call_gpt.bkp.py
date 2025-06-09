import sys
import json
import asyncio
from invoke_llm import InvokeGPT

llm = InvokeGPT()
f = open('log.in', 'w')
f.close()
f = open('log.out', 'w')
f.close() 
model_name = 'cody'

def convert_input_messages(raw_input):
    return [
        {
            "role": msg["role"],
            "content": next((c["text"] for c in msg["content"] if c["type"] == "input_text"), "")
        }
        for msg in raw_input
    ]

def wrap_tool_definition(tool):
    if tool.get("type") == "function" and "name" in tool:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {})
            }
        }
    return tool

def log(msg):
    with open("log.out", "a") as f:
        f.write(msg + '\n')

async def main():
    # Step 1: Read and parse input JSON
    input_data_str = sys.stdin.read()
    with open('log.in', 'w') as f:
        f.write("Input JSON:\n" + input_data_str + '\n')

    try:
        input_data = json.loads(input_data_str)
        log("[✓] Parsed JSON successfully")
    except Exception as e:
        log(f"[ERROR] JSON parsing failed")
        return

    log("[✓] Parsed input_data")

    # Step 2: Extract and convert messages
    input_messages_raw = input_data.get("input", [])
    messages = convert_input_messages(input_messages_raw)
    log(f"[✓] Converted input messages:\n{json.dumps(messages, indent=2)}")

    # Step 3: Always inject system message from instructions
    instructions = input_data.get("instructions", "").strip()
    system_message = {
        "role": "system",
        "content": instructions
    }
    messages.insert(0, system_message)
    log(f"[✓] Injected system message:\n{json.dumps(system_message, indent=2)}")

    # Step 4: Wrap tools if present
    tools = input_data.get("tools", [])
    wrapped_tools = [wrap_tool_definition(tool) for tool in tools]
    log(f"[✓] Wrapped tools:\n{json.dumps(wrapped_tools, indent=2)}")

    # Step 5: Prepare LLM call
    model = input_data.get("model", "gpt-4")
    tool_choice = input_data.get("tool_choice", "auto")
    log(f"[✓] Prepared model={model}, tool_choice={tool_choice}")
    log(f"[✓] Messages to be sent:\n{json.dumps(messages, indent=2)}")

    try:
        response_gen = await llm.get_response(
            messages,
            tools=wrapped_tools,
            stream=True,
            tool_choice=tool_choice,
            model=model
        )
        log("[✓] LLM get_response executed")
        
        log(repr(response_gen))

        # Step 6: Stream each event to stdout and log
        async for event in response_gen:
            '''
            if hasattr(event, "model"):
                event.model = model_name
            if hasattr(event.choices[0], "delta"):
                yield {
                    "model": "cody-0.1",
                    "type": "delta",
                    "delta": event.choices[0].delta,
                    "finish_reason": event.choices[0].finish_reason
                }
            ''' 
            log(f"[→] Event: {repr(event)}")
            print(json.dumps(event), flush=True)

    except Exception as e:
        log(f"[ERROR] During LLM call or streaming: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
