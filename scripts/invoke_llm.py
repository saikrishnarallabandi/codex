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


    def get_response(self, messages=None, tools=None, stream=False, tool_choice="auto", model=None):

        with open("log.out", "a") as f:
            f.write("Starting get_response in InvokeGPT\n")
        if messages is None:
            messages = self.messages
        else:
            self.messages = messages

        wrapped_tools = [wrap_tool_definition(t) for t in tools] if tools else None

        response = openai.chat.completions.create(
            model=self.model if model is None else model,
            messages=messages,
            tools=wrapped_tools,
            tool_choice=tool_choice,
            stream=stream,
        )
        with open("log.out", "a") as f:
            f.write("Response received from OpenAI\n")

        return response