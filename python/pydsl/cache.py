import functools
import os
import os.path
from pathlib import Path
from typing import Callable

@functools.cache
def get_cache_dir():
    path = os.environ.get('PYDSL_CACHE_DIR', None)
    if path is not None:
        return Path(path)
    return Path.home() / ".pydsl" / "cache"


class CompilationCache:
    def __init__(self, name, hash_code):
        cache_dir = get_cache_dir() / hash_code
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self.name = name

    def get_file_path(self, suffix: str):
        return self.cache_dir / (self.name + "." + suffix)

    def get_or_create(self, suffix: str, ctor: Callable[[], bytes]):
        file_path = self.get_file_path(suffix)
        try:
            file = open(file_path, "rb")
            return file.read()
        except FileNotFoundError:
            pass
        else:
            raise
        contents = ctor()
        if isinstance(contents, str):
            contents = contents.encode('utf-8')
        with open(file_path, "wb") as file:
            file.write(contents)
        return contents
