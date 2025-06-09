#!/usr/bin/env python3
import json
import sys
import asyncio
import subprocess

from invoke_llm import InvokeGPT

def log(msg: str) -> None:
    with open("log.out", "a") as f:
        f.write(msg + "\n")

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
        call_args = json.loads(args)
        cmd = call_args.get("command", [])
        workdir = call_args.get("workdir")
        timeout = call_args.get("timeout")
    except Exception:
        cmd = []
        workdir = None
        timeout = None

    if isinstance(cmd, list):
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                timeout=(timeout / 1000) if isinstance(timeout, (int, float)) else None,
                capture_output=True,
                text=True,
            )
            result = proc.stdout.strip()
        except Exception as e:
            result = str(e)
    else:
        result = ""

    messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
    chat_resp2 = await gpt.get_response(
        messages,
        tools=request.get("tools"),
        model=request.get("model"),
    )
    text = chat_resp2["choices"][0]["message"]["content"]
    log(f"[✓] Built final text: {text}")

    await emit({
        "type": "response.output_item.added",
        "output_index": 1,
        "item": {"type": "message", "id": msg_id, "status": "in_progress", "role": "assistant", "content": [{"type": "output_text", "text": ""}]},
    })
    await emit({
        "type": "response.output_text.delta",
        "item_id": msg_id,
        "output_index": 1,
        "content_index": 0,
        "delta": text,
    })
    await emit({
        "type": "response.output_text.done",
        "item_id": msg_id,
        "output_index": 1,
        "content_index": 0,
        "text": text,
    })
    await emit({
        "type": "response.output_item.done",
        "output_index": 1,
        "item": {"type": "message", "id": msg_id, "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": text}]},
    })

    await emit({
        "type": "response.completed",
        "response": {
            "id": resp_id,
            "status": "completed",
            "model": chat_resp2["model"],
            "output": [
                {"type": "function_call", "id": func_id, "status": "completed", "call_id": call_id, "name": "shell", "arguments": args},
                {"type": "message", "id": msg_id, "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": text}]},
            ],
            "parallel_tool_calls": False,
        },
    })
    log("[✓] Response completed")

if __name__ == "__main__":
    asyncio.run(main())
