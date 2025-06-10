#!/usr/bin/env python3
import json
import os
import sys
import asyncio
import uuid
from invoke_llm import InvokeGPT

# Reset logs
open("log.out", "w").close()

def convert_input_messages(raw_input):
    messages = []
    for item in raw_input:
        if item.get("type") == "message":
            parts = item.get("content", [])
            text = "".join(
                p.get("text", "")
                for p in parts
                if isinstance(p, dict) and p.get("type") == "input_text"
            )
            messages.append({"role": item.get("role"), "content": text})
        elif item.get("type") == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": item.get("output", ""),
                }
            )
    return messages


def log(msg: str) -> None:
    with open("log.out", "a") as f:
        f.write(msg + "\n")

def gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

async def main():
    open("log.out", "w").close()

    if not os.environ.get("OPENAI_API_KEY"):
        log("[ERROR] OPENAI_API_KEY not set")
        return

    data_str = sys.stdin.read()
    if not data_str:
        sys.stderr.write("Expected request JSON on stdin\n")
        return
    log("[✓] Read input data from stdin")
    log(json.dumps(data_str, indent=4))
    
    try:
        request = json.loads(data_str)
        log("[✓] Parsed JSON successfully")
    except Exception as e:
        log(f"[ERROR] JSON parsing failed: {e}")
        sys.stderr.write("Expected request JSON on stdin\n")
        return
    instructions = request.get("instructions", "")
    messages = convert_input_messages(request.get("input", []))
    if instructions:
        messages.insert(0, {"role": "system", "content": instructions})
    log("[✓] Built message list")
    log(f"Messages: {json.dumps(messages, indent=4)}")

    wrapped_tools = request.get("tools")
    tool_choice = request.get("tool_choice", "auto")

    model = "gpt-4o-mini"
    llm = InvokeGPT(model=model)


    try:
        stream = llm.get_response(
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

        for chunk in stream:
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

            if finish_reason == "stop" and text_content:
                emit({
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "id": gen_id("msg"),
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text_content}],
                    },
                })
                final_output.append({
                    "type": "message",
                    "id": gen_id("msg"),
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
