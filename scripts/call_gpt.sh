#!/bin/bash

# filepath: /home2/srallaba/projects/project_codex/repos/codex/scripts/call_gpt.sh

# Example JSON input
read -r -d '' REQUEST <<EOF
{
  "instructions": "You are a helpful assistant.",
  "input": [
    {
      "role": "user",
      "content": [
        { "type": "input_text", "text": "What is the weather in Paris?" }
      ]
    }
  ],
  "tools": [
    {
      "type": "function",
      "name": "shell",
      "description": "Run a shell command",
      "parameters": {
        "type": "object",
        "properties": {
          "command": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["command"]
      }
    }
  ],
  "model": "gpt-4o-mini"
}
EOF

# Call the Python script with the JSON input
python3 /home2/srallaba/projects/project_codex/repos/codex/scripts/call_gpt.py <<< "$REQUEST"
