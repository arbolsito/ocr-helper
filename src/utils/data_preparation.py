import re
from os import Path

LABEL_PATTERN = re.compile(r"*_\d")

def parse_label_from_filename(p:Path, pattern:)->str:
    