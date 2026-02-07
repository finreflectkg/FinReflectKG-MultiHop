"""
Simple LLM Client for Multi-hop QA Generation

All configs loaded from config.yaml.

Supports models with reasoning in:
- Separate fields: msg.reasoning / msg.reasoning_content (Qwen3-32B, Nemotron-Nano-30B)
- Content with <think> tags: (Qwen3-8B, Qwen3-235B, Nemotron-Super-49B)
- Raw thinking in content: (Nemotron-Nano-9B)
- Harmony API format: analysisXXXassistantfinalYYY (GPT-OSS with reasoning)

Reasoning control methods:
- enable_thinking: true/false in extra_body (most models)
- system_prompt_prefix: "/no_think" prepended to system prompt (Nemotron-Super-49B, Nemotron-Nano-9B)
- Harmony API with ReasoningEffort.HIGH (GPT-OSS reasoning mode)
"""

import re
import yaml
import time
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APIStatusError
import tiktoken

# Harmony imports for GPT-OSS reasoning
try:
    from openai_harmony import (
        load_harmony_encoding,
        HarmonyEncodingName,
        Role,
        Message,
        Conversation,
        SystemContent,
        DeveloperContent,
        ReasoningEffort,
    )
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False

# Anthropic imports for Claude models
try:
    from anthropic import AnthropicFoundry
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


def load_config() -> Dict[str, Any]:
    """Load config.yaml and substitute environment variables."""
    import os

    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config_str = f.read()

    # Substitute environment variables (${VAR_NAME} syntax)
    def substitute_env_vars(text: str) -> str:
        pattern = r'\$\{([^}]+)\}'
        def replacer(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))  # Keep original if not found
        return re.sub(pattern, replacer, text)

    config_str = substitute_env_vars(config_str)
    return yaml.safe_load(config_str)


@dataclass
class LLMResponse:
    """LLM response with content and optional reasoning."""
    content: str
    reasoning: Optional[str] = None

    def __str__(self) -> str:
        """For backwards compatibility, str() returns content."""
        return self.content or ""


def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    retryable_errors: tuple = (503, 500, 429, 502, 504)
):
    """Execute function with exponential backoff retry on transient errors.

    Args:
        func: Callable to execute
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds (doubles each retry)
        max_delay: Maximum delay between retries
        retryable_errors: HTTP status codes to retry on

    Returns:
        Result from func()

    Raises:
        Last exception if all retries exhausted
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except RateLimitError as e:
            # 429 - Rate limited
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                print(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
        except APIStatusError as e:
            # Check if status code is retryable (503, 500, 502, 504)
            last_exception = e
            if hasattr(e, 'status_code') and e.status_code in retryable_errors:
                if attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    print(f"Server error {e.status_code}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
            else:
                # Non-retryable error, raise immediately
                raise
        except APIConnectionError as e:
            # Connection errors - retry
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                print(f"Connection error, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
        except APIError as e:
            # Generic API error - check if retryable
            last_exception = e
            if hasattr(e, 'status_code') and e.status_code in retryable_errors:
                if attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    print(f"API error {e.status_code}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
            else:
                raise

    # All retries exhausted
    raise last_exception


def extract_think_tags(content: str) -> Tuple[Optional[str], str]:
    """Extract <think> tags from content and return (reasoning, clean_content)."""
    if not content:
        return None, ""

    # Check for <think> tags
    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL | re.IGNORECASE)
    if think_match:
        reasoning = think_match.group(1).strip()
        clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
        return reasoning, clean_content

    # Check for </think> without opening tag (Nemotron-Nano-9B pattern)
    if '</think>' in content.lower():
        parts = re.split(r'</think>', content, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()

    return None, content


def extract_harmony_format(text: str) -> Tuple[Optional[str], str]:
    """Extract reasoning and answer from Harmony format (analysisXXXassistantfinalYYY)."""
    if not text:
        return None, ""

    # Pattern: analysis<reasoning>assistantfinal<answer>
    # Handle variations in the format
    patterns = [
        r'analysis(.*?)assistantfinal(.*?)$',
        r'analysis(.*?)assistant.*?final(.*?)$',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            reasoning = match.group(1).strip()
            answer = match.group(2).strip()
            # Return whatever we extracted - even if one part is empty
            # reasoning can be empty (simple questions), answer should exist
            return reasoning if reasoning else None, answer

    # Fallback - no harmony format found
    return None, text.strip()


class LLMClient:
    """Simple OpenAI-compatible LLM client with reasoning support."""

    def __init__(self, model_name: Optional[str] = None):
        config = load_config()
        llm_config = config["llm"]

        # Use default model if not specified
        if model_name is None:
            model_name = llm_config["default_model"]

        model_cfg = llm_config["models"][model_name]

        # Determine client type (openai or anthropic)
        self.client_type = model_cfg.get("client_type", "openai")

        if self.client_type == "anthropic":
            # Initialize Anthropic client
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")

            self.client = AnthropicFoundry(
                api_key=model_cfg["api_key"],
                base_url=model_cfg.get("base_url")
            )
        else:
            # Initialize OpenAI client (default)
            self.client = OpenAI(
                base_url=model_cfg["base_url"],
                api_key=model_cfg["api_key"],
                timeout=380.0  # ~6 minute timeout - prevents hanging forever
            )

        self.model = model_cfg["model"]
        self.model_name = model_name
        self.max_tokens = model_cfg.get("max_tokens", 16000)
        self.temperature = model_cfg.get("temperature", 0.3)
        self.extra_body = model_cfg.get("extra_body", None)

        # Context limit for dynamic max_tokens calculation
        self.context_limit = model_cfg.get("context_limit", 32000)

        # System prompt prefix for Nemotron models (e.g., "/no_think")
        self.system_prompt_prefix = model_cfg.get("system_prompt_prefix", None)

        # Harmony API for GPT-OSS reasoning mode
        self.use_harmony_api = model_cfg.get("use_harmony_api", False)
        if self.use_harmony_api and HARMONY_AVAILABLE:
            self.harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        # Tiktoken encoder for accurate token counting
        # Use cl100k_base which is used by GPT-4, works well for most models
        try:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._tokenizer = None

    def _estimate_tokens(self, text: str) -> int:
        """Count tokens using tiktoken for accuracy."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # Fallback to conservative estimate (3 chars per token)
        return len(text) // 3

    def _calculate_dynamic_max_tokens(self, system_prompt: str, user_prompt: str) -> int:
        """Calculate max_tokens dynamically based on input length and context limit."""
        input_tokens = self._estimate_tokens(system_prompt + user_prompt)
        # Use 15% buffer + 1000 tokens to account for tokenizer differences
        # Tiktoken (cl100k_base) can differ significantly from model tokenizers
        buffer = max(1000, int(input_tokens * 0.15))
        available = self.context_limit - input_tokens - buffer

        # Use minimum of available space and configured max_tokens
        # Ensure at least 500 tokens for output
        dynamic_max = max(500, min(available, self.max_tokens))
        return dynamic_max

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None
    ) -> str:
        """Call LLM and return response text (backwards compatible).

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Optional temperature override

        Returns:
            Response content as string (for backwards compatibility)
        """
        response = self.complete_with_reasoning(system_prompt, user_prompt, temperature)
        return response.content or ""

    def complete_with_reasoning(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None
    ) -> LLMResponse:
        """Call LLM and return response with reasoning separated.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Optional temperature override

        Returns:
            LLMResponse with content and reasoning fields
        """
        # Use Anthropic API for Claude models
        if self.client_type == "anthropic":
            return self._complete_with_anthropic(system_prompt, user_prompt, temperature)

        # Use Harmony API for GPT-OSS reasoning mode
        if self.use_harmony_api and HARMONY_AVAILABLE:
            return self._complete_with_harmony(system_prompt, user_prompt, temperature)

        # Apply system prompt prefix if configured (for Nemotron models)
        final_system_prompt = system_prompt
        if self.system_prompt_prefix:
            final_system_prompt = f"{self.system_prompt_prefix} {system_prompt}"

        # NOTE: max_tokens commented out - let server auto-calculate based on remaining context
        # This avoids tokenizer mismatch issues between tiktoken and model tokenizers
        # dynamic_max_tokens = self._calculate_dynamic_max_tokens(final_system_prompt, user_prompt)

        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # "max_tokens": dynamic_max_tokens,  # Let server handle this
            "temperature": temperature if temperature is not None else self.temperature
        }

        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        # Call with retry on transient errors
        result = retry_with_backoff(
            lambda: self.client.chat.completions.create(**kwargs),
            max_retries=3,
            base_delay=2.0
        )
        msg = result.choices[0].message
        content = msg.content or ""

        # Extract reasoning - try separate fields first
        reasoning = None
        if hasattr(msg, 'reasoning') and msg.reasoning:
            reasoning = msg.reasoning
        elif hasattr(msg, 'reasoning_content') and msg.reasoning_content:
            reasoning = msg.reasoning_content

        # If no separate reasoning field, check for <think> tags in content
        if reasoning is None and content:
            extracted_reasoning, clean_content = extract_think_tags(content)
            if extracted_reasoning:
                reasoning = extracted_reasoning
                content = clean_content

        return LLMResponse(
            content=content,
            reasoning=reasoning
        )

    def _complete_with_harmony(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None
    ) -> LLMResponse:
        """Call GPT-OSS using Harmony API for reasoning mode.

        Uses completions.create() with Harmony format for detailed reasoning.
        """
        # Build Harmony conversation
        system_message = SystemContent.new().with_reasoning_effort(ReasoningEffort.HIGH)
        developer_message = DeveloperContent.new().with_instructions(system_prompt)

        conversation = Conversation.from_messages([
            Message.from_role_and_content(Role.SYSTEM, system_message),
            Message.from_role_and_content(Role.DEVELOPER, developer_message),
            Message.from_role_and_content(Role.USER, user_prompt),
        ])

        formatted_prompt = self.harmony_encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )

        # Call completions API (not chat.completions)
        # NOTE: GPT-OSS Harmony API requires max_tokens (unlike chat API)
        # Call with retry on transient errors
        response = retry_with_backoff(
            lambda: self.client.completions.create(
                model=self.model,
                prompt=formatted_prompt,
                max_tokens=self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
            ),
            max_retries=3,
            base_delay=2.0
        )

        raw_text = response.choices[0].text or ""

        # Parse Harmony format: analysisXXXassistantfinalYYY
        reasoning, content = extract_harmony_format(raw_text)

        return LLMResponse(
            content=content,
            reasoning=reasoning
        )

    def _complete_with_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None
    ) -> LLMResponse:
        """Call Anthropic API (Claude models via AnthropicFoundry).

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Optional temperature override

        Returns:
            LLMResponse with content (no separate reasoning for Claude)
        """
        # Call with retry on transient errors
        response = retry_with_backoff(
            lambda: self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            ),
            max_retries=3,
            base_delay=2.0
        )

        # Extract content from response
        content = ""
        if response.content and len(response.content) > 0:
            # Anthropic returns a list of content blocks
            content = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])

        return LLMResponse(
            content=content,
            reasoning=None  # Claude doesn't separate reasoning
        )


# Quick test
if __name__ == "__main__":
    client = LLMClient()
    response = client.complete(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say hello in one sentence."
    )
    print(response)
