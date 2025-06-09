import sys
import json
import asyncio
from invoke_llm import InvokeGPT
import collections.abc

llm = InvokeGPT()
model_name = "cody"

# Reset logs
open("log.out", "w").close()

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
    input_data_str = sys.stdin.read()
    with open("log.in", "w") as f:
        f.write("Input JSON:\n" + input_data_str + "\n")
    log("[✓] Read input data from stdin")
    log(input_data_str + "\n")    

    try:
        input_data = json.loads(input_data_str)
        log("[✓] Parsed JSON successfully")
    except Exception as e:
        log(f"[ERROR] JSON parsing failed: {str(e)}")
        return

    input_messages_raw = input_data.get("input", [])
    messages = convert_input_messages(input_messages_raw)
    instructions = input_data.get("instructions", "").strip()
    messages.insert(0, { "role": "system", "content": instructions })

    tools = input_data.get("tools", [])
    wrapped_tools = [wrap_tool_definition(tool) for tool in tools]

    model = input_data.get("model", "gpt-4")
    tool_choice = input_data.get("tool_choice", "auto")

    log(f"[✓] Sending request to model={model}")
    log(f"[✓] Messages:\n{json.dumps(messages, indent=2)}")

    try:
        response_stream = await llm.get_response(
            messages,
            tools=wrapped_tools,
            stream=True,
            tool_choice=tool_choice,
            model=model
        )

        log("[✓] Started response stream")

        full_content = ""

        async for chunk in response_stream:
            if hasattr(chunk, "to_dict"):
                chunk = chunk.to_dict()

            log(f"[→] Chunk: {repr(chunk)}")

            choices = chunk.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})

            # ✅ Handle tool calls safely
            if "tool_calls" in delta:
                for tool_call_raw in delta["tool_calls"]:
                    tool_call = tool_call_raw.to_dict() if hasattr(tool_call_raw, "to_dict") else tool_call_raw
                    tool_id = tool_call.get("id", "tool_call")
                    function = tool_call.get("function", {})
                    tool_name = function.get("name", "")
                    arguments = function.get("arguments", "")

                    print(json.dumps({
                        "type": "response.output_item.done",
                        "item": {
                            "type": "function_call",
                            "id": tool_id,
                            "name": tool_name,
                            "arguments": arguments
                        }
                    }), flush=True)

            # ✅ Handle content
            content_piece = delta.get("content")
            if content_piece is not None:
                full_content += content_piece
                #print(json.dumps({
                #    "type": "response.output_item.done",
                #    "item": {
                #        "type": "message",
                #        "role": "assistant",
                #        "content": [
                #            { "type": "output_text", "text": content_piece }
                #        ]
                #    }
                #}), flush=True)

        if not full_content:
            raise Exception("No content accumulated from stream")

        # Final signal that the response is complete
        print(json.dumps({
            "type": "response.completed",
            "response": {
                "id": "final",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            { "type": "output_text", "text": full_content }
                        ]
                    }
                ]
            }
        }), flush=True)

    except Exception as e:
        log(f"[ERROR] During LLM call or output formatting: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
