import os
from pathlib import Path
from dataclasses import dataclass
'''
FlagCX distributed runtime configuration module.

This module is responsible for:
1. Locating the local FlagCX runtime installation directory;
2. Configuring the FlagCX bitcode and shared library paths required
   by Triton/TLE distributed execution;
3. Providing extern library mappings for compilation/runtime stages.

Expected directory layout:
  ~/.flagtree/flagcx/
      ├── libflagcx_device.bc
      └── libflagcx.so

Main components:

- FlagCXConfig:
    Resolves and validates the FlagCX runtime paths, including:
      - libflagcx_device.bc
      - libflagcx.so

Typical usage:

  dist = Distributed()
  extern_libs = dist.get_extern_libs()

  triton.compile(..., extern_libs=extern_libs)
'''


@dataclass
class FlagCXConfig:
    bitcode_path: str
    shared_lib_path: str

    def __init__(self):
        self.default_libdir = self._get_bitcode_path()
        self.bitcode_path = str(self.default_libdir / 'libflagcx_device.bc')
        self.shared_lib_path = str(self.default_libdir / 'libflagcx.so')
        if not os.path.exists(self.bitcode_path):
            raise FileNotFoundError(f"FlagCX bitcode not found at {self.bitcode_path}")

    def _get_bitcode_path(self):
        path = Path.home() / ".flagtree" / "flagcx"
        if not path.exists():
            raise FileNotFoundError(f"FlagCX directory not found at {path}")
        return path


class Distributed:

    def __init__(self):
        self.is_use_flagcx = os.environ.get("USE_FLAGCX", "ON") == "ON"
        self.extern_libs = {}
        if self.is_use_flagcx:
            self.extern_libs["libflagcx"] = FlagCXConfig().bitcode_path

    def get_extern_libs(self):
        return self.extern_libs
