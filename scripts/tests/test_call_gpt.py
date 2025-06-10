import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from call_gpt import convert_input_messages, build_messages

class ConvertInputMessagesTests(unittest.TestCase):
    def test_handles_output_text(self):
        raw_input = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            },
            {
                "type": "function_call_output",
                "call_id": "call1",
                "output": "result",
            },
        ]
        messages = convert_input_messages(raw_input)
        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "tool", "tool_call_id": "call1", "content": "result"},
            ],
        )


class BuildMessagesTests(unittest.TestCase):
    def test_prefers_messages_field(self):
        request = {
            "messages": [
                {"role": "system", "content": "hi"},
                {"role": "user", "content": "hello"},
            ]
        }
        self.assertEqual(build_messages(request), request["messages"])

    def test_falls_back_to_input(self):
        request = {
            "instructions": "sys",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
            ],
        }
        self.assertEqual(
            build_messages(request),
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}],
        )

if __name__ == "__main__":
    unittest.main()
