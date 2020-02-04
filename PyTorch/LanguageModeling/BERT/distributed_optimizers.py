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
group_id = None
def local_init():
    global torch_group, group_id
    torch.distributed.init_process_group(backend="nccl",
                                         init_method="file:///tmp/distributed_test",
                                         world_size=num_gpus,
                                         rank=MPI.COMM_WORLD.rank % num_gpus)
    for group_num in range(num_gpus//num_devices):
        group_ids = range(group_num*num_devices, (group_num+1)*num_devices)
        cur_group = torch.distributed.new_group(ranks=group_ids)
        if torch.distributed.get_rank()//num_devices== group_num:
            torch_group = cur_group
            group_id = group_num
                                                            
def local_device():
    return world_rank() % num_gpus

def local_reduce_sum_(tensor, rank):
    torch.distributed.reduce(tensor, dst = rank+group_id*num_devices, async_op=False, group=torch_group)
    return None

def local_reduce_sum_async_(tensor, rank):
    return torch.distributed.reduce(tensor, dst = rank+group_id*num_devices, async_op=True, group=torch_group)
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
    handle = torch.distributed.broadcast(tensor, src=root+group_id*num_devices, async_op=True, group=torch_group)
    return handle

def local_broadcast_sync_(tensor, root=0):
    import torch
    torch.distributed.broadcast(tensor, src=root+group_id*num_devices, async_op=False, group=torch_group)

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
                 backward_passes_per_step=1, create_master=False):
        params = [p
                  for param_group in optimizer.param_groups
                  for p in param_group['params']]
        super(DistributedAdasumOptimizer, self).__init__(params, dict(DistributedAdasumOptimizer.__dict__))
        
        self._compression = compression
        self.optimizer = optimizer
        self.create_master = create_master
        
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
        #amp._amp_state.loss_scalers[0]._loss_scale = 2**15
        
    def _register_hooks(self):
        self.optimizer._amp_lazy_init() # why they lazily init is beyond me... its such a headache.
        param_index = 0
        for param_group in self.optimizer.param_groups:
            assert len(param_group['params']) == 1, "requires 1 parameter per group"
            for p in param_group['params']:
                assert p.requires_grad
                start = None
                master = None
                if self.create_master or self.owner[param_index] == local_rank():
                    master = torch.empty_like(p, requires_grad=False).float()
                    master.grad = torch.empty_like(master, requires_grad=False)
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                self._allreduce_delay.append(self.backward_passes_per_step)
                self._handles.append((None, None))
                scalar = DynamicLossScaler(init_scale=2**18)
                grad_acc.register_hook(self._make_hook(param_index, scalar, param_group, p, start, master))
                self._grad_accs.append(grad_acc)
                param_index += 1

    def _make_hook(self, param_index, scalar, group, p, start, master):
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
                handle, ctx = self._allreduce_grad_async(param_index, scalar, group, p, start, master)
            self._handles[param_index] = (handle, ctx)
        return hook

    
    def _allreduce_grad_async(self, param_index, scalar, group, p, start, master):
        owner = self.owner[param_index]
        handle = local_reduce_sum_async_(p.grad, owner)
        if local_rank() == owner:
            if self._is_first:
                master.data.copy_(p.data)
        return handle, (p, param_index, scalar, start, group, master)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()                          

        # finish backward
        amp_scale = amp._amp_state.loss_scalers[0]
        scale = 1.0 / (num_devices * amp_scale.loss_scale())
        my_param_groups = []        
        fp16_grads, fp32_grads = [], []
        norm_sq = 0.0
        for handle, (p, param_index, scalar, start, group, master) in reversed(self._handles):
            owner = self.owner[param_index]
            if local_rank() == owner:
                group['params'] = [master]
                my_param_groups.append(group)
                fp16_grads.append(p.grad)
                fp32_grads.append(master.grad)
                handle.wait()
            else:
                handle.wait()

        amp_C.multi_tensor_scale(65536,
                                 amp_scale._overflow_buf,
                                 [fp16_grads, fp32_grads],
                                 scale)
        
        norm_sq = 0.0
        for handle, (p, param_index, scalar, start, group, master) in self._handles:
            owner = self.owner[param_index]
            if local_rank() == owner:
                norm_sq += master.grad.data.norm(p=2)**2
        local_allreduce_sum_(norm_sq)
        #had_overflow = not (torch.isfinite(norm_sq).item())
        self.cpu_buffer.copy_(norm_sq, non_blocking=True)
        
        clip_coef = 1.0 / (torch.sqrt(norm_sq) + 1e-6)
        clip_coef = torch.min(clip_coef, self.one)
        amp_C.multi_tensor_scale(65536,
                                 amp_scale._overflow_buf,
                                 [fp32_grads, fp32_grads],
                                 clip_coef)
        
        torch.cuda.current_stream().synchronize()
        had_overflow = not math.isfinite(self.cpu_buffer.item())
        # grad cip and step
        if had_overflow == False:
            tmp = self.optimizer.param_groups
            self.optimizer.param_groups = my_param_groups
            self.optimizer.step()
            self.optimizer.param_groups = tmp

        # start adasum
        #with Timer("start_adasum"):
        hvd_handles = [None] * len(self._handles)
        for _, (p, param_index, scalar, start, group, master) in self._handles:
            owner = self.owner[param_index]
            handle = None
            if local_rank() == owner:                    
                if had_overflow == False:
                    master.data.sub_(p.data)#, out=p.grad.data)
                    torch.mul(master.data, scalar.loss_scale, out=p.grad.data)
                else:
                    p.grad.data.zero_()
                name = 'allreduce_%i' % param_index
                handle = hvd.allreduce_async_(p.grad.data, name=name, op=hvd.Adasum)
            hvd_handles[param_index] = handle                

        # finish adasum
        #with Timer("finish_adasum"):        
        fp16_grads = []
        fp32_masters = []
        for _, (p, param_index, scalar, start, group, master) in self._handles:
            owner = self.owner[param_index]
            self._allreduce_delay[param_index] = self.backward_passes_per_step
            if local_rank() == owner:
                group['params'] = [p]                    

                fp16_grads.append(p.grad)
                fp32_masters.append(master)
                
                hvd.synchronize(hvd_handles[param_index])

        scalar = self._handles[0][1][2]
        self.overflow_buf.zero_()
        amp_C.multi_tensor_scale(65536,
                                 self.overflow_buf,
                                 [fp16_grads, fp32_masters],
                                 1.0 / scalar.loss_scale)

        #layer_had_overflow = self.overflow_buf.item()
        #tmp = 1.0 - self.overflow_buf
        self.cpu_buffer.copy_(self.overflow_buf, non_blocking=True)
        for _, (p, param_index, scalar, start, group, master) in self._handles:
            owner = self.owner[param_index]
            if local_rank() == owner:
                #if layer_had_overflow == 0:
                p.data.add_(master.data)
                #master.data.copy_(p.data, non_blocking=True)
                local_broadcast_sync_(p.data, root=owner)                
            else:
                local_broadcast_sync_(p.data, root=owner)

            #local_broadcast_handles.append(local_broadcast_async_(p.data, root=owner))
                    
        #with Timer("cleanup"):
        amp_scale._has_overflow = had_overflow
        amp_scale.update_scale()        
        if had_overflow:
            print("Rank {} had overflow: new scale {} ".format(
                world_rank(), amp_scale.loss_scale()), flush=True)

        amp_scale.clear_overflow_state()        
        self.optimizer.zero_grad()
        self._is_first = False

        #for handle in local_broadcast_handles:
        #    handle.wait()

        torch.cuda.current_stream().synchronize()
        layer_had_overflow = self.cpu_buffer.item()

        if layer_had_overflow == 1:
            print("Layer overflow", world_rank(),
                  scalar.loss_scale, flush=True)
            for _, (p, param_index, scalar, start, group, master) in self._handles:
                owner = self.owner[param_index]
                if local_rank() == owner:
                    p.data.copy_(master.data)
        else:
            for _, (p, param_index, scalar, start, group, master) in self._handles:
                owner = self.owner[param_index]
                if local_rank() == owner:
                    master.data.copy_(p.data, non_blocking=True)
                            
        scalar.update_scale(layer_had_overflow == 1)
        
        return loss


    def zero_grad(self):
        if self._handles:
            raise AssertionError("optimizer.zero_grad() was called after loss.backward() "
                                 "but before optimizer.step() or optimizer.synchronize(). "
                                 "This is prohibited as it can cause a race condition.")
        return self.optimizer.zero_grad()



class DistributedAdasumForLambOptimizer(DistributedAdasumOptimizer):
    def __init__(self, optimizer, compression,
                 backward_passes_per_step=1):
        super(DistributedAdasumForLambOptimizer, self).__init__(optimizer, compression, backward_passes_per_step, True)

    def _allreduce_grad_async(self, param_index, scalar, group, p, start, master):
        handle = local_allreduce_sum_async_(p.grad)
        if self._is_first:
            master.data.copy_(p.data)
        return handle, (p, param_index, scalar, start, group, master)

    def step(self, closure=None):
        assert closure is None, "Not supported"
        
        # finish backward
        amp_scale = amp._amp_state.loss_scalers[0]
        scale = 1.0 / (num_devices * amp_scale.loss_scale())
        fp16_grads, fp32_grads = [], []
        master_param_groups = []        
        for handle, (p, param_index, scalar, start, group, master) in reversed(self._handles):
            fp16_grads.append(p.grad)
            fp32_grads.append(master.grad)
            group['params'] = [master]
            master_param_groups.append(group)
            handle.wait()

        amp_C.multi_tensor_scale(65536,
                                 amp_scale._overflow_buf,
                                 [fp16_grads, fp32_grads],
                                 scale)
        
        had_overflow = amp_scale._overflow_buf.item() > 0
        if had_overflow == False:
            tmp = self.optimizer.param_groups
            self.optimizer.param_groups = master_param_groups
            self.optimizer.step()
            self.optimizer.param_groups = tmp

        # start adasum
        #with Timer("start_adasum"):
        hvd_handles = [None] * len(self._handles)
        for _, (p, param_index, scalar, start, group, master) in self._handles:
            owner = self.owner[param_index]
            handle = None
            if local_rank() == owner:                    
                if had_overflow == False:
                    master.data.sub_(p.data)
                    torch.mul(master.data, scalar.loss_scale, out=p.grad.data)
                else:
                    p.grad.data.zero_()
                name = 'allreduce_%i' % param_index
                handle = hvd.allreduce_async_(p.grad.data, name=name, op=hvd.Adasum)
            group['params'] = [p] # reset
            hvd_handles[param_index] = handle                

        # finish adasum
        for _, (p, param_index, scalar, start, group, master) in self._handles:
            owner = self.owner[param_index]
            self._allreduce_delay[param_index] = self.backward_passes_per_step
            if local_rank() == owner:
                hvd.synchronize(hvd_handles[param_index])

        local_handles = []
        for _, (p, param_index, scalar, start, group, master) in self._handles:
            owner = self.owner[param_index]
            if local_rank() == owner:
                self.overflow_buf.zero_()                
                amp_C.multi_tensor_scale(65536,
                                         self.overflow_buf,
                                         [[p.grad.data], [master.data]],
                                         1.0 / scalar.loss_scale)
                layer_had_overflow = self.overflow_buf.item() > 0
                if not layer_had_overflow:
                    p.data.add_(master.data)                    
                else:
                    print("Layer {} had overflow: new scale {}: rank: {}".format(
                        param_index, scalar.loss_scale, world_rank()), flush=True)
                scalar.update_scale(layer_had_overflow)                
            handle = local_broadcast_async_(p.data, root=owner)            
            local_handles.append((p, master, handle))

        #with Timer("cleanup"):
        amp_scale._has_overflow = had_overflow
        amp_scale.update_scale()        
        if had_overflow:
            print("Rank {} had overflow: new scale {} ".format(
                world_rank(), amp_scale.loss_scale()), flush=True)
        amp_scale.clear_overflow_state()        
        self.optimizer.zero_grad()
        self._is_first = False        

        for p, master, handle in local_handles:
            handle.wait()
            master.data.copy_(p.data) 
