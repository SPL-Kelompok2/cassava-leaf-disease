import shutil
import sys
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

needs_java = pytest.mark.skipif(shutil.which("java") is None, reason="Java tidak terpasang (dibutuhkan Pellet)")
