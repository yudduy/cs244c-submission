# LLM client — OpenAI API via openai SDK.
# Supports: gpt-4o, gpt-5.2, gpt-5.3-codex (Responses API), o3-mini, o4-mini.
import json
import os
import re
from pathlib import Path

import backoff
import openai

# Auto-load .env from project root
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists() and not os.environ.get("OPENAI_API_KEY"):
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()

MAX_OUTPUT_TOKENS = 4096

# Default model for CEGIS evolution
DEFAULT_MODEL = "gpt-5.3-codex"

# Models that require the Responses API instead of Chat Completions
_RESPONSES_API_MODELS = {"gpt-5.3-codex", "gpt-5.2-codex", "gpt-5-codex"}


def create_client(model: str):
    """Create OpenAI client for the specified model.
    Returns (client, model_name) tuple."""
    return openai.OpenAI(), model


def _uses_responses_api(model: str) -> bool:
    """Check if model needs the Responses API instead of Chat Completions."""
    return model in _RESPONSES_API_MODELS or "codex" in model.lower()


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APITimeoutError),
    max_time=120,
)
def get_response_from_llm(
        msg,
        client,
        model,
        system_message,
        print_debug=False,
        msg_history=None,
        temperature=0.7,
):
    if msg_history is None:
        msg_history = []

    if _uses_responses_api(model):
        # Codex models: use Responses API
        content = _call_responses_api(
            client, model, system_message, msg, msg_history, temperature,
        )
        new_msg_history = msg_history + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": content},
        ]
    elif model.startswith("o1-") or model.startswith("o3-") or model.startswith("o4-"):
        # o-series: no system role, temperature must be 1
        new_msg_history = msg_history + [
            {"role": "user", "content": system_message + "\n\n" + msg}
        ]
        response = client.chat.completions.create(
            model=model,
            messages=[*new_msg_history],
            temperature=1,
            n=1,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [
            {"role": "assistant", "content": content}
        ]
    else:
        # gpt-4o, gpt-5.2: standard Chat Completions API
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        if "gpt-5" in model:
            token_param = {"max_completion_tokens": MAX_OUTPUT_TOKENS}
        else:
            token_param = {"max_tokens": MAX_OUTPUT_TOKENS}
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            **token_param,
            n=1,
            stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [
            {"role": "assistant", "content": content}
        ]

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        print(f"Model: {model}")
        print(f'User: {msg[:200]}...')
        print(f'Assistant: {content[:200]}...')
        print("*" * 21 + " LLM END " + "*" * 21)
        print()
    return content, new_msg_history


def _call_responses_api(client, model, system_message, msg, msg_history, temperature):
    """Call OpenAI Responses API for codex models."""
    # Build input: system instructions + conversation history + new message
    input_parts = []
    if system_message:
        input_parts.append({
            "role": "developer",
            "content": system_message,
        })
    for m in msg_history:
        input_parts.append({"role": m["role"], "content": m["content"]})
    input_parts.append({"role": "user", "content": msg})

    response = client.responses.create(
        model=model,
        input=input_parts,
        temperature=temperature,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return response.output_text


def extract_json_between_markers(llm_output):
    """Extract JSON from ```json ... ``` blocks. Fallback to regex."""
    inside_json_block = False
    json_lines = []

    for line in llm_output.split('\n'):
        striped_line = line.strip()
        if striped_line.startswith("```json"):
            inside_json_block = True
            continue
        if inside_json_block and striped_line.startswith("```"):
            inside_json_block = False
            break
        if inside_json_block:
            json_lines.append(line)

    if not json_lines:
        # Try parsing the entire output as JSON (LLM sometimes returns raw JSON)
        stripped = llm_output.strip()
        if stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                stripped_clean = re.sub(r"[\x00-\x1F\x7F]", "", stripped)
                try:
                    return json.loads(stripped_clean)
                except json.JSONDecodeError:
                    pass
        # Greedy fallback: find the outermost { ... } block
        match = re.search(r"\{.*\}", llm_output, re.DOTALL)
        if match:
            candidate = match.group(0).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                candidate_clean = re.sub(r"[\x00-\x1F\x7F]", "", candidate)
                try:
                    return json.loads(candidate_clean)
                except json.JSONDecodeError:
                    pass
        return None

    json_string = "\n".join(json_lines).strip()
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
        try:
            return json.loads(json_string_clean)
        except json.JSONDecodeError:
            return None


def extract_code_block(llm_output):
    """Extract Python code from ```python ... ``` blocks."""
    inside_code_block = False
    code_lines = []

    for line in llm_output.split('\n'):
        striped_line = line.strip()
        if striped_line.startswith("```python"):
            inside_code_block = True
            continue
        if inside_code_block and striped_line.startswith("```"):
            inside_code_block = False
            break
        if inside_code_block:
            code_lines.append(line)

    if not code_lines:
        return None
    return "\n".join(code_lines)
