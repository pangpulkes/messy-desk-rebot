"""Import this first: if we're not running inside the `rebot` conda env, re-launch with its Python.

Lets `python ~/rebot_lerobot/llm_arm/<script>.py` work from any terminal tab, even without
`conda activate rebot`.
"""

import os
import sys

REBOT_PYTHON = "/opt/homebrew/Caskroom/miniforge/base/envs/rebot/bin/python"

if os.path.realpath(sys.executable) != os.path.realpath(REBOT_PYTHON) and os.path.exists(REBOT_PYTHON):
    os.execv(REBOT_PYTHON, [REBOT_PYTHON, *sys.argv])
