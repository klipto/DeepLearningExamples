from mpi4py import MPI
import horovod.torch as hvd
import os

def local_init():
    import torch
    torch.distributed.init_process_group(backend="nccl",
                                         init_method="file:///tmp/distributed_test",
                                         world_size=num_devices,
                                         rank=local_rank())

def local_reduce_mean_async_(tensor, root=0):
    import torch
    tensor.div_(local_size())
    handle = torch.distributed.all_reduce(tensor, async_op=True)
    return handle

def local_broadcast_async_(tensor, root=0):
    import torch
    handle = torch.distributed.broadcast(tensor, src=root, async_op=True)
    return handle

def local_broadcast_sync_(tensor, root=0):
    import torch
    torch.distributed.broadcast(tensor, src=root, async_op=False)

def local_rank():
    return world_rank() % num_devices

def local_size():
    return num_devices

def world_rank():
    return MPI.COMM_WORLD.rank

def world_size():
    return MPI.COMM_WORLD.size


num_devices = 4
#newcomm = MPI.COMM_WORLD.Split(MPI.COMM_WORLD.rank % num_devices, MPI.COMM_WORLD.rank)
#hvd.init(comm=newcomm)
if world_rank() % num_devices == 0:
    hvd.init(comm=[i for i in range(world_size()) if i % num_devices == 0])
os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank())

