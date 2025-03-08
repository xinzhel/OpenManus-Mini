from typing import Dict, List, Literal, Optional, Union
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from app.schema import Message

class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __init__(
        self, llm_config: dict = None
    ):
        
        self.model = llm_config.model
        self.max_tokens = llm_config.max_tokens
        self.temperature = llm_config.temperature
        self.api_key = llm_config.api_key
        self.base_url = llm_config.base_url
        self.client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url
        )

    @staticmethod
    def format_messages(messages: List[Union[dict, Message]]) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            if isinstance(message, dict):
                # If message is already a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")
                formatted_messages.append(message)
            elif isinstance(message, Message):
                # If message is a Message object, convert it to dict
                formatted_messages.append(message.to_dict())
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ["system", "user", "assistant", "tool"]:
                raise ValueError(f"Invalid role: {msg['role']}")
            if "content" not in msg and "tool_calls" not in msg:
                raise ValueError(
                    "Message must contain either 'content' or 'tool_calls'"
                )

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        temperature: Optional[float] = None,
    ) -> str:
        # Format system and user messages
        if system_msgs:
            system_msgs = self.format_messages(system_msgs)
            messages = system_msgs + self.format_messages(messages)
        else:
            messages = self.format_messages(messages)

        # Streaming request
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=temperature or self.temperature,
            stream=True,
        )

        collected_messages = []
        for chunk in response:
            chunk_message = chunk.choices[0].delta.content or ""
            collected_messages.append(chunk_message)
            print(chunk_message, end="", flush=True)

        print()  # Newline after streaming
        full_response = "".join(collected_messages).strip()
        if not full_response:
            raise ValueError("Empty response from streaming LLM")
        return full_response


    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 60,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        temperature: Optional[float] = None,
        **kwargs,
    ):
        # Validate tool_choice
        if tool_choice not in ["none", "auto", "required"]:
            raise ValueError(f"Invalid tool_choice: {tool_choice}")

        # Format messages
        if system_msgs:
            system_msgs = self.format_messages(system_msgs)
            messages = system_msgs + self.format_messages(messages)
        else:
            messages = self.format_messages(messages)

        # Set up the completion request
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=self.max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            timeout=timeout,
            **kwargs,
        )

        # Check if response is valid
        if not response.choices or not response.choices[0].message:
            print(response)
            raise ValueError("Invalid or empty response from LLM")

        return response.choices[0].message

 
