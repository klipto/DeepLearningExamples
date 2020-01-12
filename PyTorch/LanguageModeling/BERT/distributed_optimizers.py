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

def local_init():
    import torch
    torch.distributed.init_process_group(backend="nccl",
                                         init_method="file:///tmp/distributed_test",
                                         world_size=num_devices,
                                         rank=local_rank())

def local_reduce_sum_(tensor, rank):
    torch.distributed.reduce(tensor, dst = rank, async_op=False)
    return None

def local_allreduce_mean_async_(tensor, root=0):
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

get_num_gpus = lambda: str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID') if any(os.access(os.path.join(path, 'nvidia-smi'), os.X_OK) for path in os.environ["PATH"].split(os.pathsep)) else 0
num_devices = get_num_gpus()

from contextlib import contextmanager

def find_duplicates(lst):
    seen = set()
    dups = set()
    for el in lst:
        if el in seen:
            dups.add(el)
        seen.add(el)
    return dups

def post_backward_models_are_masters(scaler, param, stashed_grad):
    grads_have_scale, stashed_have_scale, out_scale = scaler.loss_scale(), 1.0, 1.0
    grads_needing_unscale = []
    grads_needing_unscale_with_stash = []
    stashed = []

    if param.grad is None and stashed_grad is not None:
        param.grad = stashed_grad
    elif param.grad is not None and stashed_grad is None:
        grads_needing_unscale.append(param.grad)
    elif param.grad is not None and stashed_grad is not None:
        grads_needing_unscale_with_stash.append(param.grad)
        stashed.append(stashed_grad)

    # unscale() implements grads*(1/scale), so "scale" should be grads_have_scale/out_scale.
    if len(grads_needing_unscale) > 0:
        scaler.unscale(
            grads_needing_unscale,
            grads_needing_unscale,
            None, # unused_scale, currently present to avoid API breakage elsewhere
            models_are_masters=True,
            scale_override=grads_have_scale/out_scale)
        
    if len(grads_needing_unscale_with_stash) > 0:
        scaler.unscale_with_stashed(
            grads_needing_unscale_with_stash,
            stashed,
            grads_needing_unscale_with_stash,
            scale_override=(grads_have_scale, stashed_have_scale, out_scale))
        
    # Clear the stash.
    #for i in range(len(stashed_grads)):
    #    stashed_grads[i] = None
                                                                                                
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
        #self._allreduce_delay = {v: self.backward_passes_per_step
        #                         for _, v in sorted(named_parameters)}
        self._allreduce_delay = []
        self._handles = []
        self._grad_accs = []
        self._requires_update = []
        self._synchronized = False
        self._should_synchronize = True
        self._is_first = True

        self._register_hooks()
        self.one = torch.cuda.FloatTensor([1.0])
        self.overflow_buf = torch.cuda.IntTensor([0])

    def _register_hooks(self):
        self.optimizer._amp_lazy_init() # why they lazily init is beyond me... its such a headache.
        param_index = 0
        for param_group in self.optimizer.param_groups:
            assert len(param_group['params']) == 1, "requires 1 parameter per group"
            for p in param_group['params']:
                assert p.requires_grad
                start = torch.empty_like(p, requires_grad=False).float()
                master = torch.empty_like(start, requires_grad=False)
                master.grad = torch.empty_like(start, requires_grad=False)
                #self._requires_update.append(p)
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                self._allreduce_delay.append(self.backward_passes_per_step)
                self._handles.append((None, None))
                scalar = DynamicLossScaler(init_scale=2**15)
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
        owner = param_index % num_devices
        local_reduce_sum_(p.grad, owner)

        if self._is_first:
            start.data.copy_(p.data)
        
        handle = True
        if local_rank() == owner:
            amp_scale = amp._amp_state.loss_scalers[0]
            amp_C.multi_tensor_scale(65536,
                                     amp_scale._overflow_buf,
                                     [[p.grad], [master.grad]],
                                     1.0 / (num_devices * amp_scale.loss_scale()))
            master.data.copy_(p.data)
            
            norm = master.grad.norm()
            clip_coeff = 1.0 / (norm + 1e-6)
            clip_coeff = torch.min(clip_coeff, self.one)
            master.grad.data.mul_(clip_coeff)                        
            
            had_overflow = amp_scale._overflow_buf.item()
            if had_overflow == 0:
                tmp = self.optimizer.param_groups
                self.optimizer.param_groups = [group]
                group['params'] = [master]

                self.optimizer.step()
                group['params'] = [p]
                self.optimizer.param_groups = tmp
                
                master.data.sub_(start.data)
                master.data.mul_(scalar.loss_scale)
                p.grad.data.copy_(master.data)
            else:
                p.grad.data.zero_()
                
            name = 'allreduce_%i' % param_index
            handle = hvd.allreduce_async_(p.grad.data, name=name, op=hvd.Adasum)

        return handle, (p, param_index, scalar, start, p.grad, master)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()                          

        t0 = time.time()
        running = 0
        local_broadcast_handles = []
        for param_index in reversed(range(len(self._handles))):
            handle, ctx = self._handles[param_index]
            p, param_index, scalar, start, delta, tmp_buf  = ctx
            owner = param_index % num_devices
            if local_rank() == owner:
                self.overflow_buf.zero_()
                ta = time.time()
                hvd.synchronize(handle)
                tb = time.time()
                running += tb - ta
                amp_C.multi_tensor_scale(65536,
                                         self.overflow_buf,
                                         [[delta], [tmp_buf]],
                                         1.0 / scalar.loss_scale)
                had_overflow = self.overflow_buf.item()
                if had_overflow == 0:
                    start.data.add_(tmp_buf)
                p.data.copy_(start)
                local_broadcast_handles.append(local_broadcast_async_(p.data, root=owner))
                scalar.update_scale(had_overflow)
            else:
                local_broadcast_handles.append(local_broadcast_async_(p.data, root=owner))
            self._allreduce_delay[param_index] = self.backward_passes_per_step        
        t1 = time.time()
        for handle in local_broadcast_handles:
            handle.wait()                
        t2 = time.time()
        had_overflow = amp._amp_state.loss_scalers[0].update_scale()
        if had_overflow == 1:
            print("Rank {} had overflow: new scale {}".format(
                world_rank(), amp._amp_state.loss_scalers[0].loss_scale()), flush=True)
        t3 = time.time()        
        # keep all local amp scaler's in sync
        tmp_tensor = torch.cuda.FloatTensor([amp._amp_state.loss_scalers[0]._loss_scale])
        amp_handle = local_broadcast_async_(tmp_tensor, root=0)

        self._is_first = False            
        self.optimizer.zero_grad()
        #assert not amp._amp_state.opt_properties.master_weights

        amp_handle.wait()
        t4 = time.time()
        amp._amp_state.loss_scalers[0]._loss_scale = tmp_tensor.item()        
        amp._amp_state.loss_scalers[0].clear_overflow_state()
        t5 = time.time()

        if world_rank() == 0:
            print('RRR', t5-t0, t5-t4, t4-t3, t3-t2, t2-t1, t1-t0, running, flush=True)
        
        return loss

    def zero_grad(self):
        if self._handles:
            raise AssertionError("optimizer.zero_grad() was called after loss.backward() "
                                 "but before optimizer.step() or optimizer.synchronize(). "
                                 "This is prohibited as it can cause a race condition.")
        return self.optimizer.zero_grad()

