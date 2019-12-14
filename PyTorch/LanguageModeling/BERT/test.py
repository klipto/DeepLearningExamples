import os
num_devices = 4
from mpi4py import MPI
import horovod.torch as hvd

print("initializsing",flush=True)
newcomm = MPI.COMM_WORLD.Split(MPI.COMM_WORLD.rank % num_devices, MPI.COMM_WORLD.rank)
hvd.init(comm=newcomm)
#newgrp = MPI.COMM_WORLD.group.Incl([0,4])
#newcomm = MPI.COMM_WORLD.Create_group(newgrp)
#if MPI.COMM_WORLD.rank in [0,4]:
#    hvd.init(comm=[0,4])
    
print("done initializsing",flush=True)
os.environ['CUDA_VISIBLE_DEVICES'] = str(MPI.COMM_WORLD.rank % num_devices)
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl",
                        init_method="file:///tmp/distributed_test2",
                        world_size=num_devices,
                        rank=MPI.COMM_WORLD.rank % num_devices)

local_rank = MPI.COMM_WORLD.rank % num_devices
tensors = [torch.FloatTensor([MPI.COMM_WORLD.rank]).cuda() for i in range(MPI.COMM_WORLD.size)]

local_allreduce_handles = []
for tensor in tensors:
    local_allreduce_handle = dist.all_reduce(tensor, async_op=True)
    local_allreduce_handles.append(local_allreduce_handle)

assert len(local_allreduce_handles) == len(tensors)

hvd_handles = []
for index, (tensor, local_allreduce_handle) in enumerate(zip(tensors, local_allreduce_handles)):
    local_allreduce_handle.wait()
    handle = None
    if index % num_devices == local_rank:
        name = 'tensor_%i'%index
        handle = hvd.allreduce_async_(tensor,op=hvd.Sum,name=name)
    hvd_handles.append(handle)

local_bcast_handles = []
for index, (tensor, hvd_handle) in enumerate(zip(tensors, hvd_handles)):
    #if index % num_devices == local_rank:
    if hvd_handle is not None:
        tensor = hvd.synchronize(hvd_handle)
    bcast_handle = dist.broadcast(tensor,src=index%num_devices,async_op=True)
    local_bcast_handles.append(bcast_handle)
    
for handle in local_bcast_handles:
    handle.wait()

for index, tensor in enumerate(tensors):
    print(MPI.COMM_WORLD.rank, index, tensor.item())
    
#if MPI.COMM_WORLD.rank in [0,4]:
#    #print("i am root", MPI.COMM_WORLD.rank, newcomm.rank, newcomm.size, flush=True)
#    print(MPI.COMM_WORLD.rank, tensor)
#    hvd.allreduce_(tensor, op=hvd.Adasum)
#    print(MPI.COMM_WORLD.rank, tensor)
#dist.broadcast_multigpu([tensor], src=local_rank, src_tensor=0)#, async_op=True)

