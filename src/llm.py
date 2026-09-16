"""Provider-agnostic LLM client.

OpenRouter, Groq and NVIDIA NIM all expose the same OpenAI-compatible
/chat/completions shape, so one request path covers all three. Pick one with
LLM_PROVIDER; the matching <PROVIDER>_API_KEY must be set.

    setx OPENROUTER_API_KEY sk-or-...     &  set LLM_PROVIDER=openrouter
    setx GROQ_API_KEY       gsk_...       &  set LLM_PROVIDER=groq
    setx NVIDIA_API_KEY     nvapi-...     &  set LLM_PROVIDER=nvidia

Override the model with LLM_MODEL. Responses are cached on disk keyed by
(provider, model, prompt) so re-running the evaluation costs nothing and gives
byte-identical numbers.
"""
import hashlib, json, os, re, time
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dotenv(path=None):
    """Read KEY=value lines from .env at the repo root into os.environ.

    Set here rather than relying on the shell: on Windows `setx` only reaches
    newly launched processes and an `export` does not survive between tool
    invocations, so a file the process reads itself is the one mechanism that
    works the same way everywhere. Real environment variables always win, so
    exporting a key still overrides the file.
    """
    path = path or os.path.join(_ROOT, ".env")
    if not os.path.exists(path):
        return
    seen = set()
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if not key:
                continue
            if key in seen:
                # A duplicated key in .env silently resolves to whichever line
                # came first, which is exactly the kind of thing you debug for
                # twenty minutes. Say so.
                print(f"[llm] warning: {key} appears more than once in "
                      f"{os.path.basename(path)}; using the first value")
                continue
            seen.add(key)
            if key not in os.environ:
                os.environ[key] = val


load_dotenv()

PROVIDERS = {
    "openrouter": {
        "base": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.3-70b-instruct",
    },
    "groq": {
        "base": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        # Groq retired the Llama 3.3 endpoints; gpt-oss-120b is the strongest
        # general instruct model it currently serves. Check the live list with
        #   GET https://api.groq.com/openai/v1/models
        "default_model": "openai/gpt-oss-120b",
    },
    "nvidia": {
        "base": "https://integrate.api.nvidia.com/v1/chat/completions",
        "key_env": "NVIDIA_API_KEY",
        "default_model": "meta/llama-3.3-70b-instruct",
    },
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", ".llm_cache")


class LLMUnavailable(RuntimeError):
    """Raised when no provider is configured, so callers can fall back."""


def active_provider():
    """Resolve the provider to use: LLM_PROVIDER, else the first key that is set."""
    name = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if name:
        if name not in PROVIDERS:
            raise LLMUnavailable(f"unknown LLM_PROVIDER={name!r}; pick one of {list(PROVIDERS)}")
        if not os.environ.get(PROVIDERS[name]["key_env"]):
            raise LLMUnavailable(f"LLM_PROVIDER={name} but {PROVIDERS[name]['key_env']} is not set")
        return name
    for n, cfg in PROVIDERS.items():
        if os.environ.get(cfg["key_env"]):
            return n
    raise LLMUnavailable(
        "no LLM configured -- set one of " +
        ", ".join(c["key_env"] for c in PROVIDERS.values()))


def is_available():
    try:
        active_provider()
        return True
    except LLMUnavailable:
        return False


def model_name():
    p = active_provider()
    return os.environ.get("LLM_MODEL") or PROVIDERS[p]["default_model"]


def _cache_path(key):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, key + ".json")


def _retry_delay(resp, attempt):
    """How long to wait after a 429.

    Free tiers here are limited by tokens-per-minute, not requests, and the
    reset can be ~60s -- far longer than a naive 1/2/4/8s backoff will wait. So
    honour the server's own numbers when it sends them and only fall back to
    exponential backoff when it does not. Guessing shorter than the reset window
    just burns retries and turns a rate limit into a fake model failure.
    """
    hdr = resp.headers if resp is not None else {}
    for key in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        raw = hdr.get(key)
        if not raw:
            continue
        try:
            if raw.replace(".", "", 1).isdigit():      # plain seconds
                return min(float(raw) + 1.0, 90.0)
            m = re.fullmatch(r"(?:(\d+)m)?([\d.]+)s", raw.strip())  # e.g. "1m12.4s"
            if m:
                mins = float(m.group(1) or 0)
                return min(mins * 60 + float(m.group(2)) + 1.0, 90.0)
        except (ValueError, AttributeError):
            pass
    return min(2 ** attempt, 60)


def chat(prompt, system=None, temperature=0.0, max_tokens=800,
         reasoning_effort="low", retries=8):
    """Single-turn completion. Returns the assistant text.

    Deterministic by default (temperature 0) and disk-cached, because the
    evaluation harness must give the same numbers on a re-run.

    `reasoning_effort` matters more than it looks. Reasoning models such as
    gpt-oss emit their chain of thought into a separate `reasoning` field that
    still counts against max_tokens, so a long prompt can burn the entire budget
    thinking and return an EMPTY content string with finish_reason="stop" -- a
    silent failure that looks like a valid answer. "low" cuts completion tokens
    ~40% here, which both fixes the truncation and eases the tokens-per-minute
    rate limit. Providers that do not know the parameter ignore it.
    """
    provider = active_provider()
    cfg = PROVIDERS[provider]
    model = model_name()

    key = hashlib.sha256(json.dumps(
        [provider, model, system, prompt, temperature, max_tokens,
         reasoning_effort],
        sort_keys=True).encode()).hexdigest()[:32]
    path = _cache_path(key)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)["text"]
        if cached.strip():
            return cached
        os.remove(path)   # never trust a cached empty answer

    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort

    last = None
    for attempt in range(retries):
        resp = None
        try:
            resp = requests.post(cfg["base"], json=body, timeout=120, headers={
                "Authorization": f"Bearer {os.environ[cfg['key_env']]}",
                "Content-Type": "application/json",
            })
            if resp.status_code == 429 or resp.status_code >= 500:
                last = f"{resp.status_code} {resp.text[:160]}"
                delay = _retry_delay(resp, attempt)
                print(f"      [{provider} {resp.status_code}; waiting {delay:.0f}s]",
                      flush=True)
                time.sleep(delay)
                continue
            resp.raise_for_status()
            choice = resp.json()["choices"][0]
            text = (choice["message"].get("content") or "").strip()
            if not text:
                # Truncated by the reasoning budget. Retry with more room rather
                # than caching the empty string and scoring it as a model error.
                last = (f"empty content (finish_reason="
                        f"{choice.get('finish_reason')})")
                body["max_tokens"] = min(int(body["max_tokens"] * 2), 4000)
                print(f"      [empty reply; retrying with "
                      f"max_tokens={body['max_tokens']}]", flush=True)
                continue
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"provider": provider, "model": model,
                           "prompt": prompt, "text": text}, f, ensure_ascii=False)
            return text
        except requests.RequestException as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(_retry_delay(resp, attempt))
    raise LLMUnavailable(f"{provider} failed after {retries} attempts: {last}")


def chat_json(prompt, system=None, **kw):
    """chat() plus tolerant JSON extraction (models like to wrap JSON in prose)."""
    text = chat(prompt, system=system, **kw)
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        t = t[4:] if t.lower().startswith("json") else t
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in response: {text[:200]}")
    return json.loads(t[start:end + 1])


if __name__ == "__main__":
    if not is_available():
        print("no provider configured; set OPENROUTER_API_KEY / GROQ_API_KEY / NVIDIA_API_KEY")
    else:
        print(f"provider={active_provider()} model={model_name()}")
        print("reply:", chat("Reply with the single word: ok"))
