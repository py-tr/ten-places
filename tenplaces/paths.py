"""The three paths every module resolves against: the checkout, the SO-101 model it clones, and the output tree."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SO101_XML = ROOT / "third_party" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.xml"
OUT = ROOT / "out"
