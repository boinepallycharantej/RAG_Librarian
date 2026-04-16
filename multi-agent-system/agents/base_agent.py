"""
Base agent class — handles the tool-use agentic loop with Claude.
"""
import anthropic
import json
from typing import Callable

client = anthropic.Anthropic()


class BaseAgent:
    """
    Runs an agentic loop: sends messages to Claude, executes tool calls,
    feeds results back, and repeats until Claude stops calling tools.
    """

    def __init__(self, name: str, system: str, tools: list, tool_handlers: dict[str, Callable]):
        self.name = name
        self.system = system
        self.tools = tools
        self.tool_handlers = tool_handlers  # {"tool_name": handler_fn}

    def run(self, user_message: str, model: str = "claude-opus-4-6", max_tokens: int = 4096) -> str:
        """
        Execute the agentic loop and return the final text response.
        """
        messages = [{"role": "user", "content": user_message}]

        while True:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=self.system,
                tools=self.tools,
                messages=messages,
            )

            # Collect tool calls from this response
            tool_calls = [b for b in response.content if b.type == "tool_use"]

            if not tool_calls or response.stop_reason == "end_turn":
                # No more tool calls — extract final text and return
                text_blocks = [b.text for b in response.content if b.type == "text"]
                return "\n".join(text_blocks).strip()

            # Append assistant response to message history
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool call and collect results
            tool_results = []
            for tc in tool_calls:
                handler = self.tool_handlers.get(tc.name)
                if handler is None:
                    result = f"Error: unknown tool '{tc.name}'"
                else:
                    try:
                        result = handler(**tc.input)
                        if not isinstance(result, str):
                            result = json.dumps(result)
                    except Exception as e:
                        result = f"Tool error: {e}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})
