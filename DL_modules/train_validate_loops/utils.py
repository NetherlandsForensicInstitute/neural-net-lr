import os

from DL_modules.utils import ResultTLM


def get_output_path(path: str) -> str:
    return os.path.join(path, f'{get_next_entry(path)}/')


def get_next_entry(path: str) -> int:
    os.makedirs(path, exist_ok=True)
    entries = os.listdir(path)
    next_entry = 1
    if dirs := [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]:
        next_entry += max(_try_convert_to_int(d) for d in dirs)
    return next_entry


def _try_convert_to_int(s: str) -> int | bool:
    try:
        return int(s)
    except ValueError:
        return False


EmptyResultTLM = ResultTLM({}, {}, {}, {})
