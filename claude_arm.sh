#!/bin/zsh
# Launch the Claude arm agent through MIT Parley, with the key read from the macOS Keychain
# (never printed or stored in a file). Any arguments are passed to agent.py, e.g.:
#   ~/rebot_lerobot/llm_arm/claude_arm.sh --dry-run
#   ~/rebot_lerobot/llm_arm/claude_arm.sh            # real arm, approve each motion
#   ~/rebot_lerobot/llm_arm/claude_arm.sh --auto     # real arm, no per-motion approval
export ANTHROPIC_BASE_URL="https://parley.api.mit.edu"
ANTHROPIC_API_KEY="$(security find-generic-password -a "$USER" -s anthropic-api-key -w)" || {
  echo "No key in the Keychain (service 'anthropic-api-key'). Store it with:"
  echo "  security add-generic-password -U -a \"\$USER\" -s anthropic-api-key -w"
  exit 1
}
export ANTHROPIC_API_KEY
exec /opt/homebrew/Caskroom/miniforge/base/envs/rebot/bin/python "$(dirname "$0")/agent.py" "$@"
