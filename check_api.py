"""Check the Claude API connection (direct Anthropic or a gateway such as MIT Parley).

Usage (key comes from the macOS Keychain; it is never printed):
  ANTHROPIC_BASE_URL=https://parley.api.mit.edu \
  ANTHROPIC_API_KEY="$(security find-generic-password -a "$USER" -s anthropic-api-key -w)" \
  python ~/rebot_lerobot/llm_arm/check_api.py [model]

Prints: which endpoint is used, the models it lists, and whether a tiny request works
with and without the optional features agent.py uses.
"""

import _env  # noqa: F401  (re-launches inside the rebot env if needed)

import os
import sys

import anthropic

MODEL = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-5-5"

key = os.environ.get("ANTHROPIC_API_KEY", "")
print(f"endpoint : {os.environ.get('ANTHROPIC_BASE_URL') or 'https://api.anthropic.com (default)'}")
print(f"key      : {'set, ' + str(len(key)) + ' chars' if key else 'NOT SET'}")
print(f"model    : {MODEL}\n")

client = anthropic.Anthropic()

print("1) models the endpoint lists:")
try:
    ids = [m.id for m in client.models.list()]
    print("   " + (", ".join(ids) if ids else "(none listed)"))
except Exception as e:
    print(f"   could not list models: {type(e).__name__}: {str(e)[:200]}")

MSG = [{"role": "user", "content": "Reply with the single word OK."}]


def attempt(label, fn):
    try:
        r = fn()
        text = "".join(b.text for b in r.content if b.type == "text").strip()
        print(f"   {label:28s} OK  -> {text[:40]!r} (stop: {r.stop_reason})")
    except Exception as e:
        print(f"   {label:28s} FAIL -> {type(e).__name__}: {str(e)[:200]}")


print("\n2) tiny requests:")
attempt("plain", lambda: client.messages.create(model=MODEL, max_tokens=1024, messages=MSG))
attempt("with effort", lambda: client.messages.create(
    model=MODEL, max_tokens=1024, messages=MSG, output_config={"effort": "low"}))
attempt("with refusal fallback (beta)", lambda: client.beta.messages.create(
    model=MODEL, max_tokens=1024, messages=MSG, output_config={"effort": "low"},
    betas=["server-side-fallback-2026-07-01"], fallbacks="default"))
