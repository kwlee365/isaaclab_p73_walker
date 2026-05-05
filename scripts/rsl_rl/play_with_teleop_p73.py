"""Compatibility wrapper.

We keep a single source of truth for P73 teleop under:
  `scripts/tools/p73_command_control/play_with_teleop_p73.py`

This file remains as a convenience entrypoint to avoid breaking older commands.
"""

from __future__ import annotations

import os
import runpy


def main() -> None:
    here = os.path.dirname(__file__)
    tool_script = os.path.abspath(os.path.join(here, "..", "tools", "p73_command_control", "play_with_teleop_p73.py"))
    runpy.run_path(tool_script, run_name="__main__")


if __name__ == "__main__":
    main()

