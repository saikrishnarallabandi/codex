#!/usr/bin/env python3
import json
import os
import sys
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
                if isinstance(p, dict)
                and p.get("type") in {"input_text", "output_text"}
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


def build_messages(request):
    """Return a message list from the request payload."""
    if "messages" in request:
        return request["messages"]

    instructions = request.get("instructions", "")
    messages = convert_input_messages(request.get("input", []))
    if instructions:
        messages.insert(0, {"role": "system", "content": instructions})
    return messages


def log(msg: str) -> None:
    with open("log.out", "a") as f:
        f.write(msg + "\n")

def gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def main():
    open("log.out", "w").close()

    if not os.environ.get("OPENAI_API_KEY"):
        log("[ERROR] OPENAI_API_KEY not set")
        return

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
    messages = build_messages(request)
    log("[✓] Built message list")

    wrapped_tools = request.get("tools")
    tool_choice = request.get("tool_choice", "auto")

    model = "gpt-4o-mini"
    llm = InvokeGPT(model=model)


    try:
        response = llm.get_response(
            messages,
            tools=wrapped_tools,
            stream=False,
            tool_choice=tool_choice,
            model=model,
        )

        log("[✓] Received response")

        resp_id = gen_id("resp")

        def emit(evt):
            print(json.dumps(evt), flush=True)
            log(f"[→] {evt.get('type')}")

        emit({"type": "response.created", "response": {"id": resp_id, "status": "in_progress"}})
        emit({"type": "response.in_progress", "response": {"id": resp_id, "status": "in_progress"}})

        if hasattr(response, "to_dict"):
            response = response.to_dict()

        log(f"[→] Response: {repr(response)}")

        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})

        tool_calls = message.get("tool_calls", [])
        content = message.get("content")

        final_output = []

        for tc in tool_calls:
            emit({
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": tc.get("id", gen_id("call")),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", ""),
                },
            })
            final_output.append({
                "type": "function_call",
                "id": tc.get("id", gen_id("call")),
                "name": tc.get("function", {}).get("name", ""),
                "arguments": tc.get("function", {}).get("arguments", ""),
            })

        if content:
            emit({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "id": gen_id("msg"),
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                },
            })
            final_output.append({
                "type": "message",
                "id": gen_id("msg"),
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
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
    main()
