import sys
import json
import asyncio
import uuid
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

def gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

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
        stream = await llm.get_response(
            messages,
            tools=wrapped_tools,
            stream=True,
            tool_choice=tool_choice,
            model=model,
        )

        log("[✓] Started response stream")

        resp_id = gen_id("resp")

        def emit(evt):
            print(json.dumps(evt), flush=True)
            log(f"[→] {evt.get('type')}")

        emit({"type": "response.created", "response": {"id": resp_id, "status": "in_progress"}})
        emit({"type": "response.in_progress", "response": {"id": resp_id, "status": "in_progress"}})

        tool_calls = {}
        text_content = ""
        final_output = []

        async for chunk in stream:
            if hasattr(chunk, "to_dict"):
                chunk = chunk.to_dict()

            log(f"[→] Chunk: {repr(chunk)}")

            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            for tc in delta.get("tool_calls", []):
                idx = tc.get("index", 0)
                call = tool_calls.setdefault(idx, {
                    "id": tc.get("id", gen_id("call")),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": "",
                })
                if tc.get("function", {}).get("name"):
                    call["name"] = tc["function"]["name"]
                if tc.get("function", {}).get("arguments"):
                    call["arguments"] += tc["function"]["arguments"]

            if "content" in delta and delta["content"] is not None:
                text_content += delta["content"]

            if finish_reason == "tool_calls":
                for tc in tool_calls.values():
                    emit({
                        "type": "response.output_item.done",
                        "item": {
                            "type": "function_call",
                            "id": tc["id"],
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    })
                    final_output.append({
                        "type": "function_call",
                        "id": tc["id"],
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    })

            if finish_reason == "stop":
                emit({
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text_content}],
                    },
                })
                final_output.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text_content}],
                })

        if not final_output:
            raise Exception("No output received from model")

        emit({
            "type": "response.completed",
            "response": {"id": resp_id, "status": "completed", "output": final_output},
        })

    except Exception as e:
        log(f"[ERROR] During LLM call or output formatting: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
