"""AVA label names and categories (from the official AVA v2.2 label map rebuilt by tools/setup_lart.py)."""
import re
from functools import lru_cache
from pathlib import Path

from . import CACHE_HOME

_PBTXT = Path(CACHE_HOME) / ".cache" / "lart" / "ava_action_list_v2.2.pbtxt"
_ITEM = re.compile(r'label \{\s*name: "([^"]+)"\s*label_id: (\d+)\s*label_type: (\w+)\s*\}')


@lru_cache(None)
def _items():
    return [(int(i), n, t) for n, i, t in _ITEM.findall(_PBTXT.read_text())]


class _Names(dict):
    def __missing__(self, k):
        return f"class {k}"


AVA80 = _Names({i: n for i, n, _ in _items()}) if _PBTXT.exists() else _Names()


def category_map():
    """{class name: PERSON_MOVEMENT | OBJECT_MANIPULATION | PERSON_INTERACTION} (the paper's PM / OM / PI)."""
    return {n: t for _, n, t in _items()}


def per_class_name(metric_key):
    """'PascalBoxes_PerformanceByCategory/AP@0.5IOU/bend/bow (at the waist)' -> 'bend/bow (at the waist)'.
    (Class names contain '/', so splitting on '/' would truncate them.)"""
    return metric_key.split("IOU/", 1)[1]
