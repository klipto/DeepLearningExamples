from mpi4py import MPI
import horovod.torch as hvd
import os
import subprocess
import math
import torch
import torch.distributed
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
#from fused_adam_local import FusedAdam
from apex.optimizers import FusedAdam
from apex.multi_tensor_apply import multi_tensor_applier
import amp_C
import time
from apex import amp

#from azureml.core.run import Run
#amlrun = Run.get_context()


get_num_gpus = lambda: str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID') if any(os.access(os.path.join(path, 'nvidia-smi'), os.X_OK) for path in os.environ["PATH"].split(os.pathsep)) else 0
import os
num_devices = int(os.environ['HOROVOD_NUM_GPUS_PER_LOCAL_GROUP'])
num_gpus = get_num_gpus()
assert num_devices <= num_gpus
assert num_gpus % num_devices == 0
torch_group = None

def local_init():
    global torch_group
    torch.distributed.init_process_group(backend="nccl",
                                         init_method="file:///tmp/distributed_test",
                                         world_size=num_gpus,
                                         rank=MPI.COMM_WORLD.rank % num_gpus)
    if num_devices != num_gpus:
        for group_num in range(num_gpus//num_gpus):
            group_ids = range(group_num*num_gpus, (group_num+1)*num_gpus)
            cur_group = torch.distributed.new_group(ranks=group_ids)
            if torch.distributed.get_rank()//num_gpus == group_num:
                torch_group = cur_group
                                                            
def local_device():
    return world_rank() % num_gpus

def local_reduce_sum_(tensor, rank):
    torch.distributed.reduce(tensor, dst = rank, async_op=False, group=torch_group)
    return None

def local_reduce_sum_async_(tensor, rank):
    return torch.distributed.reduce(tensor, dst = rank, async_op=True, group=torch_group)
    return None

def local_allreduce_sum_async_(tensor, root=0):
    handle = torch.distributed.all_reduce(tensor, async_op=True, group=torch_group)
    return handle

def local_allreduce_sum_(tensor, root=0):
    handle = torch.distributed.all_reduce(tensor, async_op=False, group=torch_group)
    return handle

def local_allreduce_mean_async_(tensor, root=0):
    import torch
    tensor.div_(local_size())
    handle = torch.distributed.all_reduce(tensor, async_op=True, group=torch_group)
    return handle

def local_broadcast_async_(tensor, root=0):
    import torch
    handle = torch.distributed.broadcast(tensor, src=root, async_op=True, group=torch_group)
    return handle

def local_broadcast_sync_(tensor, root=0):
    import torch
    torch.distributed.broadcast(tensor, src=root, async_op=False, group=torch_group)

def local_rank():
    return world_rank() % num_devices

def local_size():
    return num_devices

def world_rank():
    return MPI.COMM_WORLD.rank

def world_size():
    return MPI.COMM_WORLD.size

from contextlib import contextmanager

def find_duplicates(lst):
    seen = set()
    dups = set()
    for el in lst:
        if el in seen:
            dups.add(el)
        seen.add(el)
    return dups

from apex.fp16_utils.loss_scaler import DynamicLossScaler    
class DistributedAdasumOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer, compression,
                 backward_passes_per_step=1):
        params = [p
                  for param_group in optimizer.param_groups
                  for p in param_group['params']]
        super(self.__class__, self).__init__(params, dict(DistributedAdasumOptimizer.__dict__))
        
        self._compression = compression
        self.optimizer = optimizer

        named_parameters = [('allreduce.noname.%s' % i, v)
                            for param_group in self.param_groups
                            for i, v in enumerate(param_group['params'])]

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        dups = find_duplicates([k for k, _ in named_parameters])
        if len(dups) > 0:
            raise ValueError('Parameter names in named_parameters must be unique. '
                             'Found duplicates: %s' % ', '.join(dups))

        all_param_ids = {id(v)
                         for param_group in self.param_groups
                         for v in param_group['params']}
        named_param_ids = {id(v) for k, v in named_parameters}
        unnamed_param_ids = all_param_ids - named_param_ids
        if len(unnamed_param_ids):
            raise ValueError('named_parameters was specified, but one or more model '
                             'parameters were not named. Python object ids: '
                             '%s' % ', '.join(str(id) for id in unnamed_param_ids))

        self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        self.backward_passes_per_step = backward_passes_per_step
        self._allreduce_delay = []
        self._handles = []
        self._grad_accs = []
        self._requires_update = []
        self._synchronized = False
        self._should_synchronize = True

        self.cpu_buffer = torch.zeros(1,dtype=torch.float32,device='cpu',requires_grad=False).pin_memory()
        self._buffer = torch.cuda.FloatTensor([0.0, 0.0])
        self.one = torch.cuda.FloatTensor([1.0])
        self.overflow_buf = torch.cuda.IntTensor([0])
        self._is_first = True


        total = 0
        param_index = 0
        for param_group in self.optimizer.param_groups:
            assert len(param_group['params']) == 1, "requires 1 parameter per group"
            for p in param_group['params']:
                total += p.numel()
                param_index += 1
                
        self.owner = [None] * param_index
        split_size = total / num_devices
        param_index = 0
        total = 0
        curr_rank = 0
        
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                if total >= split_size:
                    total = 0
                    curr_rank += 1
                    if curr_rank == num_devices:
                        curr_rank -= 1
                total += p.numel()
                self.owner[param_index] = curr_rank
                param_index += 1
                
        self._register_hooks()
        amp._amp_state.loss_scalers[0]._loss_scale = 2**15
        
    def _register_hooks(self):
        self.optimizer._amp_lazy_init() # why they lazily init is beyond me... its such a headache.
        param_index = 0
        for param_group in self.optimizer.param_groups:
            assert len(param_group['params']) == 1, "requires 1 parameter per group"
            for p in param_group['params']:
                assert p.requires_grad

                master = torch.empty_like(p, requires_grad=False).float()
                master.grad = torch.zeros_like(master, requires_grad=False)
                master_grad = master.grad

                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                self._allreduce_delay.append(self.backward_passes_per_step)
                self._handles.append((None, None))
                scalar = DynamicLossScaler(init_scale=2**18)
                grad_acc.register_hook(self._make_hook(param_index, scalar, param_group, p,  master, master_grad))
                self._grad_accs.append(grad_acc)
                param_index += 1

    def _make_hook(self, param_index, scalar, group, p,  master, master_grad):
        def hook(*ignore):
            if self._allreduce_delay[param_index] <= 0:
                raise AssertionError(
                    "Gradients were computed more than "
                    "backward_passes_per_step times before call "
                    "to step(). Increase backward_passes_per_step to "
                    "accumulate gradients locally.")
            assert not p.grad.requires_grad
            handle, ctx = None, None
            self._allreduce_delay[param_index] -= 1
            if self._allreduce_delay[param_index] == 0:                
                handle, ctx = self._allreduce_grad_async(param_index, scalar, group, p,  master, master_grad)
                self._allreduce_delay[param_index] = self.backward_passes_per_step
            self._handles[param_index] = (handle, ctx)
        return hook

    
    def _allreduce_grad_async(self, param_index, scalar, group, p,  master, master_grad):
        handle = local_allreduce_sum_async_(p.grad)
        if self._is_first:
            master.data.copy_(p.data)
        return handle, (p, param_index, scalar,  group, master, master_grad)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()                          

        # finish backward
        amp_scale = amp._amp_state.loss_scalers[0]
        scale = 1.0 / (num_devices * amp_scale.loss_scale())
        fp16_grads, fp32_grads = [], []

        my_param_groups = []
        for handle, (p, param_index, scalar,  group, master, master_grad) in reversed(self._handles):
            group['params'] = [master]
            my_param_groups.append(group)
            fp16_grads.append(p.grad)
            fp32_grads.append(master.grad)
            handle.wait()

        amp_C.multi_tensor_scale(65536,
                                 amp_scale._overflow_buf,
                                 [fp16_grads, fp32_grads],
                                 scale)

        had_overflow = amp_scale.update_scale()
        if not had_overflow:
            tmp = self.optimizer.param_groups
            self.optimizer.param_groups = my_param_groups
            self.optimizer.step()
            self.optimizer.param_groups = tmp
        else:
            print("Rank {} had overflow: new scale {} {}".format(
                world_rank(), amp_scale.loss_scale(), had_overflow), flush=True)
            
        # start adasum
        hvd_handles = [None] * len(self._handles)
        for _, (p, param_index, scalar,  group, master, master_grad) in self._handles:            
            handle = None
            owner = self.owner[param_index]
            if local_rank() == owner:
                if not had_overflow:
                    torch.sub(master, p.data, out = master_grad)
                else:
                    master_grad.zero_()
                name = 'allreduce_%i' % param_index
                handle = hvd.allreduce_async_(master_grad.data, name=name, op=hvd.Adasum)
            hvd_handles[param_index] = handle                

        # finish adasum
        for _, (p, param_index, scalar,  group, master, master_grad) in self._handles:
            group['params'] = [p]
            owner = self.owner[param_index]
            if local_rank() == owner:                                                                                                                                                           
                hvd.synchronize(hvd_handles[param_index])

        amp_scale.clear_overflow_state()
        self.optimizer.zero_grad()
        self._is_first = False
                
        for _, (p, param_index, scalar,  group, master, master_grad) in self._handles:
            # current += delta
            owner = self.owner[param_index]
            if local_rank() == owner:
                p.data.add_(master_grad)
            local_broadcast_sync_(p.data,root=owner)
            master.data.copy_(p.data, non_blocking=True)
                    
        return loss


    def zero_grad(self):
        if self._handles:
            raise AssertionError("optimizer.zero_grad() was called after loss.backward() "
                                 "but before optimizer.step() or optimizer.synchronize(). "
                                 "This is prohibited as it can cause a race condition.")
        return self.optimizer.zero_grad()
