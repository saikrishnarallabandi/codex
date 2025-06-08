import httpx
import requests
import openai
import os
import json
import traceback

OLLAMA_URL = "http://localhost:11434/api/chat"


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

class InvokeLLama:
    def __init__(self, messages=None):
        self.messages = []
    
    async def get_response(self, messages=None):
        if messages is None:
            messages = self.messages
        else:
            self.messages = messages

        payload = {
            "model": "llama3:latest",
            "messages": messages,
            "stream": False
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(OLLAMA_URL, json=payload)

        if response.status_code == 200:
            data = response.json()
            return data["message"]["content"]
        else:
            print("Error:", response.text)

class InvokeGPT:
    def __init__(self, messages=None, model="gpt-4o-mini"):
        self.messages = messages or []
        self.model = model
        # Set OpenAI API key from environment if not already set
        if not openai.api_key:
            openai.api_key = os.environ.get("OPENAI_API_KEY", "")

    async def get_response(self, messages=None, tools=None, stream=False):
        if messages is None:
            messages = self.messages
        else:
            self.messages = messages
        if stream:
            async def error_event(category, message, tb=None):
                yield {
                    "type": "error",
                    "category": category,
                    "message": message,
                    **({"traceback": tb} if tb else {})
                }
            try:
                response = await openai.ChatCompletion.acreate(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    stream=stream
                )
                with open('./log.out', 'a') as f:
                    f.write("Response received: " + repr(response) + '\n')
                # Stream mode: yield each chunk as it arrives
                async for chunk in response:
                    choice = chunk.get('choices', [{}])[0]
                    # Handle function_call (tool call)
                    if 'function_call' in choice:
                        yield {
                            "type": "response.output_item.done",
                            "item": {
                                "type": "function_call",
                                "id": choice.get("id"),
                                "name": choice["function_call"].get("name"),
                                "arguments": choice["function_call"].get("arguments", "")
                            }
                        }
                    # Handle output text (streaming mode returns 'delta' key with 'content')
                    if 'delta' in choice and 'content' in choice['delta']:
                        yield {
                            "type": "response.output_item.done",
                            "item": {
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {"type": "output_text", "text": choice['delta']['content']}
                                ]
                            }
                        }
            except Exception as e:
                tb = traceback.format_exc()
                async for event in error_event("internal", str(e), tb):
                    yield event
        else:
            async def error_event(category, message, tb=None):
                yield {
                    "type": "error",
                    "category": category,
                    "message": message,
                    **({"traceback": tb} if tb else {})
                }
            try:
                response = await openai.ChatCompletion.acreate(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    stream=stream
                )
                with open('./log.out', 'a') as f:
                    f.write("Response received: " + repr(response) + '\n')
                # Non-stream mode: extract the text content from the response
                if response and 'choices' in response and response['choices']:
                    yield {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": response['choices'][0]['message']['content']}
                            ]
                        }
                    }
                else:
                    yield {
                        "type": "error",
                        "category": "internal",
                        "message": "No response from the model."
                    }
            except Exception as e:
                tb = traceback.format_exc()
                async for event in error_event("internal", str(e), tb):
                    yield event