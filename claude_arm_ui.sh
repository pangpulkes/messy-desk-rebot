#!/bin/zsh
# Open the reBot arm control window (Claude via MIT Parley; key read from the macOS Keychain).
#   ~/rebot_lerobot/llm_arm/claude_arm_ui.sh              # real arm
#   ~/rebot_lerobot/llm_arm/claude_arm_ui.sh --dry-run    # plan only, arm not connected
export ANTHROPIC_BASE_URL="https://parley.api.mit.edu"
ANTHROPIC_API_KEY="$(security find-generic-password -a "$USER" -s anthropic-api-key -w)" || {
  echo "No key in the Keychain (service 'anthropic-api-key')."; exit 1; }
export ANTHROPIC_API_KEY
exec /opt/homebrew/Caskroom/miniforge/base/envs/rebot/bin/python "$(dirname "$0")/ui_server.py" "$@"
