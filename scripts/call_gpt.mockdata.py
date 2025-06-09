import json
import sys
import asyncio

async def main():
    input_data = sys.stdin.read()
    data = json.loads(input_data)

    # Simulate: 1. tool call
    tool_call_event = {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "id": "call_shell_123",
            "name": "shell",
            "arguments": json.dumps({
                "command": ["echo", "hi"],
                "timeout": 1000
            })
        }
    }
    print(json.dumps(tool_call_event), flush=True)
    await asyncio.sleep(0.1)

    
    # Simulate: 2. normal message
    assistant_message_event = {
        "type": "response.output_item.done",
        "item": {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Shell command issued."
                }
            ]
        }   
    }

    print(json.dumps(assistant_message_event), flush=True)
    await asyncio.sleep(0.1)

    # Simulate: 3. final completion event
    # Correctly structured final event
    event3 = {
        "type": "response.completed",
        "response": {
            "id": "mock-response-id",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Shell command issued."
                        }
                    ]
                }
            ]
        }
    }

    print(json.dumps(event3), flush=True)
    
if __name__ == "__main__":
    asyncio.run(main())
