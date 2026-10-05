#!/usr/bin/env python3

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

import sys
from pathlib import Path
sys.path.insert(0, str((_workspace_root())))
from infrastructure.shared.experimental.simple3d.simple3d_pipeline import main
if __name__ == "__main__":
    raise SystemExit(main("link5"))
