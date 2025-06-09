#!/usr/bin/env python3
import json
import sys
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

import subprocess

from invoke_llm import InvokeGPT

def log(msg: str) -> None:
    with open("log.out", "a") as f:
        f.write(msg + "\n")

def gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

async def main():
    open("log.out", "w").close()
    data_str = sys.stdin.read()
    if not data_str:
        sys.stderr.write("Expected request JSON on stdin\n")
        return
    log("[✓] Read input data from stdin")
    log(data_str)
    try:
        request = json.loads(data_str)
        log("[✓] Parsed JSON successfully")
    except Exception as e:
        log(f"[ERROR] JSON parsing failed: {e}")
        sys.stderr.write("Expected request JSON on stdin\n")
        return
    instructions = request.get("instructions", "")
    messages = [
        {"role": "system", "content": instructions}
    ]
    for item in request.get("input", []):
        if item.get("role") == "user":
            for part in item.get("content", []):
                if isinstance(part, dict) and part.get("type") == "input_text":
                    messages.append({"role": "user", "content": part.get("text", "")})
                    break
    log("[✓] Built message list")

    gpt = InvokeGPT()
    chat_resp = await gpt.get_response(
        messages,
        tools=request.get("tools"),
        model=request.get("model"),
    )
    log("[✓] Received base reply")

    resp_id = "resp_mock"
    msg_id = "msg_1"

    call = chat_resp["choices"][0]["message"]["tool_calls"][0]
    func_id = call["id"]
    call_id = call["id"]
    args = call["function"]["arguments"]

    async def emit(evt):
        log(f"[→] {evt.get('type')}")
        print(json.dumps(evt), flush=True)
        await asyncio.sleep(0.05)

    await emit({"type": "response.created", "response": {"id": resp_id, "status": "in_progress"}})
    await emit({"type": "response.in_progress", "response": {"id": resp_id, "status": "in_progress"}})

    await emit({
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {"type": "function_call", "id": func_id, "status": "in_progress", "call_id": call_id, "name": "shell", "arguments": ""},
    })
    await emit({
        "type": "response.function_call_arguments.delta",
        "item_id": func_id,
        "output_index": 0,
        "content_index": 0,
        "delta": args,
    })
    await emit({
        "type": "response.function_call_arguments.done",
        "item_id": func_id,
        "output_index": 0,
        "content_index": 0,
        "arguments": args,
    })
    await emit({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {"type": "function_call", "id": func_id, "status": "completed", "call_id": call_id, "name": "shell", "arguments": args},
    })

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
