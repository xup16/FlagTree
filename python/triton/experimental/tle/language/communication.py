try:
    from .flagcx_wrapper import (
        FLAGCXLibrary,
        flagcxDevCommRequirements,
        flagcxUniqueId,
        FLAGCX_WIN_COLL_SYMMETRIC,
    )
    import os
    import tempfile
    import torch
    import torch.distributed as dist
    from torch.cuda.memory import CUDAPluggableAllocator
    from torch.utils.cpp_extension import load_inline
    from pathlib import Path
    enabled = True
except ImportError:
    enabled = False

mem_pool = None

flagcx_allocator_source = """
#include "flagcx.h"
extern "C" {

void* flagcx_alloc_plug(size_t size, int device, void* stream) {
  void* ptr = nullptr;
  flagcxResult_t err = flagcxMemAlloc(&ptr, size);
  if (err != flagcxSuccess) {
    return nullptr;
  }
  return ptr;
}

void flagcx_free_plug(void* ptr, size_t size, int device, void* stream) {
  if (ptr != nullptr) {
    flagcxMemFree(ptr);
  }
}

}
"""

FLAGCX_INCLUDE_PATH = os.environ.get("FLAGCX_INCLUDE_PATH",
                                     os.path.abspath(os.path.join(os.path.dirname(__file__), "include")))


def get_flagcx_mem_pool():
    libflagcx_dir = str(Path.home() / ".flagtree" / "flagcx")
    """Compile and return a PyTorch MemPool that uses flagcxMemAlloc."""
    out_dir = tempfile.gettempdir()
    lib_name = "flagcx_allocator"

    load_inline(
        name=lib_name,
        cpp_sources=flagcx_allocator_source,
        with_cuda=True,
        extra_ldflags=[
            f"-L{libflagcx_dir}",
            f"-Wl,-rpath,{libflagcx_dir}",
            "-lflagcx",
        ],
        verbose=True,
        is_python_module=False,
        build_directory=out_dir,
        extra_include_paths=[FLAGCX_INCLUDE_PATH],
    )

    allocator_wrapper = CUDAPluggableAllocator(
        f"{out_dir}/{lib_name}.so",
        "flagcx_alloc_plug",
        "flagcx_free_plug",
    )
    return torch.cuda.MemPool(allocator_wrapper.allocator())


def initialize_flagcx_communication():
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:28510", world_size=1, rank=0)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    print(f"[Rank {rank}] Starting LSA test (world_size={world_size})")

    # Initialize FlagCX
    libflagcx_path = Path().home() / ".flagtree" / "flagcx" / "libflagcx.so"
    flagcx = FLAGCXLibrary(so_file=libflagcx_path)

    if rank == 0:
        unique_id = flagcx.flagcxGetUniqueId()
        id_bytes = bytes(unique_id.contents.internal)
    else:
        id_bytes = b"\x00" * 256

    # Broadcast unique_id bytes via torch distributed
    id_tensor = torch.frombuffer(bytearray(id_bytes), dtype=torch.uint8).cuda()
    dist.broadcast(id_tensor, src=0)
    id_bytes = id_tensor.cpu().numpy().tobytes()

    if rank != 0:
        unique_id = flagcx.unique_id_from_bytes(id_bytes)

    # Init FlagCX communicator
    comm = flagcx.flagcxCommInitRank(world_size, unique_id, rank)
    print(f"[Rank {rank}] FlagCX comm initialized")

    if rank == 0:
        unique_id = flagcx.flagcxGetUniqueId()
        id_bytes = bytes(unique_id.contents.internal)
    else:
        id_bytes = b"\x00" * 256

    # Broadcast unique_id bytes via torch distributed
    id_tensor = torch.frombuffer(bytearray(id_bytes), dtype=torch.uint8).cuda()
    dist.broadcast(id_tensor, src=0)
    id_bytes = id_tensor.cpu().numpy().tobytes()

    if rank != 0:
        unique_id = flagcx.unique_id_from_bytes(id_bytes)

    # Init FlagCX communicator
    comm = flagcx.flagcxCommInitRank(world_size, unique_id, rank)
    print(f"[Rank {rank}] FlagCX comm initialized")

    # with torch.cuda.use_mem_pool(flagcx_pool):
    #     buf_tensor = torch.full((N,), float(rank), dtype=torch.float32, device="cuda")
    # raise RuntimeError(f"buf_tensor: {buf_tensor}, ptr: {buf_tensor.data_ptr():#x}")
    # buf_ptr = buf_tensor.data_ptr()

    # buf_ptr = buf_tensor.data_ptr()

    # # Register buffer with symmetric window for LSA
    # win = flagcx.flagcxCommWindowRegister(comm, buf_ptr, buf_size,
    #                                        flags=FLAGCX_WIN_COLL_SYMMETRIC)
    # print(f"[Rank {rank}] Window registered (symmetric)")

    # # Create DevComm with 1 intra barrier
    # reqs = flagcxDevCommRequirements()
    # reqs.intraMulticast = False
    # reqs.barrierCount = 0
    # reqs.intraBarrierCount = 1
    # reqs.interBarrierCount = 0
    # reqs.intraLLA2ABlockCount = 0
    # reqs.intraLLA2ASlotCount = 0
    # reqs.interForceEnable = False
    # reqs.interContextCount = 4
    # reqs.interSignalCount = 0
    # reqs.interCounterCount = 0

    # dev_comm = flagcx.flagcxDevCommCreate(comm, reqs)
    # print(f"[Rank {rank}] DevComm created")

    # # Create DevMem (with window)
    # dev_mem = flagcx.flagcxDevMemCreate(comm, buf_ptr, buf_size, win)
    # print(f"[Rank {rank}] DevMem created")

    # # Get device pointers for Triton
    # dev_comm_dptr = flagcx.flagcxDevCommGetDevicePtr(dev_comm)
    # dev_mem_dptr = flagcx.flagcxDevMemGetDevicePtr(dev_mem)
    # print(f"[Rank {rank}] Device pointers: comm={dev_comm_dptr.value:#x}, "
    #       f"mem={dev_mem_dptr.value:#x}")

    # # Synchronize all ranks before kernel launch
    # dist.barrier()


if enabled:
    # ...
    initialize_flagcx_communication()
    mem_pool = get_flagcx_mem_pool()
