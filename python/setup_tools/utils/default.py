import subprocess
from pathlib import Path
import shutil

class FlagcxRegistrar:
    def __init__(self, external):
        self.bitcode_name = "libflagcx_device.bc"
        self._set_path(external)

    def _set_path(self, external):
        submodule = external['backend']
        flagtree_cache = external['cache']
        flagtree_config = external['configs']
        self.flagcx_src_dir = submodule.dst_path
        self.flagtree_dir = flagtree_config.flagtree_root_dir
        self.bitcode_src_path = Path(self.flagcx_src_dir) / "build" / "lib" / self.bitcode_name
        self.bitcode_cache_path = Path(flagtree_cache.dir_path) / "flagcx" / self.bitcode_name
        flagtree_cache._create_subdir(subdir_name="flagcx")
    
    def _compile_and_cache(self):
        if self.bitcode_cache_path.exists():
            print("libflagcx_device.bc already exists, skipping compilation.")
        elif self.bitcode_src_path.exists():
            print("libflagcx_device.bc already exists in build directory, copying to cache...")
            shutil.copy(self.bitcode_src_path, self.bitcode_cache_path)
            print(f"libflagcx_device.bc copied from {self.bitcode_src_path} to cache at {self.bitcode_cache_path}")
        else:
            print(f"Compiling libflagcx_device.bc in {self.flagcx_src_dir}...")
            subprocess.run(["make", "-C", "bindings/ir/nvidia"], cwd=self.flagcx_src_dir, check=True)
            if not self.bitcode_src_path.exists():
                raise FileNotFoundError(f"Expected bitcode file not found: {self.bitcode_src_path}")
            print("libflagcx_device.bc compilation completed.")
            shutil.copy(self.bitcode_src_path, self.bitcode_cache_path)
            print(f"libflagcx_device.bc copied from {self.bitcode_src_path} to cache at {self.bitcode_cache_path}")


    def _copy_required_files(self):

        dst = Path(self.flagtree_dir) / "python" / "triton" / "experimental" / "tle" / "language" / "flagcx_wrapper.py"
        src = Path(self.flagcx_src_dir) / "plugin" / "interservice" / "flagcx_wrapper.py"
        shutil.copy(src, dst)
        print(f"flagcx_wrapper.py copied from {src} to {dst}")
        dst = Path(self.flagtree_dir) / "python" / "triton" / "experimental" / "tle" / "language" / "include"
        src = Path(self.flagcx_src_dir) / "flagcx" / "include" 
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"FlagCX headers copied from {src} to {dst}")


    def run(self):
        self._compile_and_cache()
        self._copy_required_files()



def handle_flagcx(*args, **kwargs):
    registrar = FlagcxRegistrar(kwargs)
    registrar.run()