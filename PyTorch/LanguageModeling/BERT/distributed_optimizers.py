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

from apex import amp

def local_init():
    import torch
    torch.distributed.init_process_group(backend="nccl",
                                         init_method="file:///tmp/distributed_test",
                                         world_size=num_devices,
                                         rank=local_rank())

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

class DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer, compression,
                 backward_passes_per_step=1):
        params = [p
                  for param_group in optimizer.param_groups
                  for p in param_group['params']]
        super(self.__class__, self).__init__(params, dict(DistributedOptimizer.__dict__))
        self.optimizer = optimizer
        self._compression = compression

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
        self._allreduce_delay = {v: self.backward_passes_per_step
                                 for _, v in sorted(named_parameters)}
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self._synchronized = False
        self._should_synchronize = True
        if hvd.size() > 1:
            self._register_hooks()

    def set_backward_passes_per_step(self, passes):
        self.backward_passes_per_step = passes
        for p in self._allreduce_delay:
            self._allreduce_delay[p] = self.backward_passes_per_step

    def _register_hooks(self):
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _allreduce_grad_async(self, p):
        name = self._parameter_names.get(p)
        tensor = p.grad
        tensor_compressed, ctx = self._compression.compress(tensor)
        handle = hvd.allreduce_async_(tensor_compressed, op=hvd.Adasum, name=name)
        return handle, ctx

    def _make_hook(self, p):
        def hook(*ignore):
            if p in self._handles and self._handles[p][0] is not None:
                if self._allreduce_delay[p] <= 0:
                    raise AssertionError(
                        "Gradients were computed more than "
                        "backward_passes_per_step times before call "
                        "to step(). Increase backward_passes_per_step to "
                        "accumulate gradients locally.")
            assert not p.grad.requires_grad
            assert self._allreduce_delay[p] > 0
            handle, ctx = None, None
            self._allreduce_delay[p] -= 1
            if self._allreduce_delay[p] == 0:
                handle, ctx = self._allreduce_grad_async(p)
            self._handles[p] = (handle, ctx)
        return hook

    def synchronize(self):
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            handle, ctx = self._allreduce_grad_async(p)
            self._handles[p] = (handle, ctx)

        for p, value in self._handles.items():
            handle, ctx = value
            if handle is None:
                handle, ctx = self._allreduce_grad_async(p)
                self._handles[p] = (handle, ctx)
        for p, (handle, _) in self._handles.items():
            output = hvd.synchronize(handle)
            self._allreduce_delay[p] = self.backward_passes_per_step
            p.grad.copy_(self._compression.decompress(output, ctx).data)
        self._handles.clear()

        self._synchronized = True

    @contextmanager
    def skip_synchronize(self):
        """
        A context manager used to specify that optimizer.step() should
        not perform synchronization.

        It's typically used in a following pattern:

        .. code-block:: python

            optimizer.synchronize()
            with optimizer.skip_synchronize():
                optimizer.step()
        """
        self._should_synchronize = False
        try:
            yield
        finally:
            self._should_synchronize = True

    def step(self, closure=None):
        if self._should_synchronize:
            if self._synchronized:
                warnings.warn("optimizer.step() called without "
                              "optimizer.skip_synchronize() context after "
                              "optimizer.synchronize(). This can cause training "
                              "slowdown. You may want to consider using "
                              "optimizer.skip_synchronize() context if you use "
                              "optimizer.synchronize() in your code.")
            self.synchronize()
        self._synchronized = False
        return self.optimizer.step(closure)
    #return super(self.__class__, self).step(closure)

    def zero_grad(self):
        if self._handles:
            raise AssertionError("optimizer.zero_grad() was called after loss.backward() "
                                 "but before optimizer.step() or optimizer.synchronize(). "
                                 "This is prohibited as it can cause a race condition.")
        return self.optimizer.zero_grad()
        #return super(self.__class__, self).zero_grad()

class DistributedCpuAdasumOptimizer:
    def __init__(self, optimizer, gradient_clipping = 1.0, compression = hvd.Compression.none):
        self.optimizer = optimizer
        self.gradient_clipping = gradient_clipping
        self.compression = compression
        self._scalers = [
            DynamicLossScaler(init_scale=2**15)
            for p in amp.master_params(optimizer)
        ]
        
        with torch.no_grad():
            self.starting_model = [
                # .float because apex lazily initializes and at this point params are fp16
                # but we want start to be in fp32
                param.clone().detach().float() 
                for param in amp.master_params(optimizer)
            ]

    def reset_start(self):
        with torch.no_grad():
            for param, start in zip(amp.master_params(self.optimizer), self.starting_model):
                start.data.copy_(param)
        self.optimizer.zero_grad()
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # this code computes the following logic for the adasum optimizer.  We want to
        # include the impact of the adam optimizer and gradient clipping in the allreduce

        # If adam is a no-op (i.e., the optmizer is just sgd) and we do not have
        # gradient clipping, then this code computes *exactly* the same thing as applying
        # adasum to just the learning rate scaled gradient.
        
        ## start = current.copy()
        ## loss.backward()          # -> compute gradient
        ## clip_grad_norm_()        # -> apply gradient clipping on true gradient
        ## adam_opt.step()          # -> apply adam optmizer
        ##                          #    note current = start - \alpha.f(g) where
        ##                          #    f is adam logic and g is the clipped gradient
        ## delta = current - start  # -> (start - \alpha.f(g)) - start = -\alpha.f(g) 
        ## allreduce_(delta)        # -> call *adasum* allreduce
        ## current = start + delta  # -> update the model parameters
        
        # compute clipping of norm on amp's master version of the model
        total_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.gradient_clipping)

        # apply Adam optimizer locally
        self.optimizer.step()

        # now do an allreduce
        handles = []

        # compute delta = current - start
        for scaler, start, current in zip(self._scalers, self.starting_model, amp.master_params(self.optimizer)):
            current.data.sub_(start)
            current.data.mul_(scaler.loss_scale)
            delta, ctx = self.compression.compress(current)
            handle = hvd.allreduce_async_(delta.data, op=hvd.Adasum)
            handles.append((handle, delta, ctx))

        # while doing allreduce set gradient to 0
        self.optimizer.zero_grad()

        # update current = start + delta
        delta_norm_sq = 0
        for (handle, delta, ctx), scaler, start, current in zip(handles, self._scalers, self.starting_model, amp.master_params(self.optimizer)):
            hvd.synchronize(handle)
            delta = self.compression.decompress(delta, ctx)
            has_overflow = not (torch.isfinite(delta.data).all().item())
            if not has_overflow:
                delta.data.div_(scaler.loss_scale)
                #delta_norm_sq += delta.norm().item()
                start.data.add_(delta)
            # note if has_overflow we reset the model to start
            # effectively throwing away this step
            # since every mpi rank has the same delta, this
            # logic is the same on each rank so the model stays in sync
            current.data.copy_(start.data, True)
            scaler.update_scale(has_overflow)

        torch.cuda.synchronize()

        # if we did fp16 training, apex adds the following method
        # to the optimizer.  It does not exist in fp32 training
        if hasattr(self.optimizer,"_master_params_to_model_params"):
            # tell apex to push master parameters to those parameters it uses for fp16
            # training            
            self.optimizer._master_params_to_model_params()

        # return some statistics about this global step
        return total_norm, delta_norm_sq

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
        self._allreduce_delay = {v: self.backward_passes_per_step
                                 for _, v in sorted(named_parameters)}
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self._synchronized = False
        self._should_synchronize = True
        self._is_first = True

        self._starting_models = {}

        self._scalers = {}

        self._register_hooks()

    def set_backward_passes_per_step(self, passes):
        self.backward_passes_per_step = passes
        for p in self._allreduce_delay:
            self._allreduce_delay[p] = self.backward_passes_per_step

    def _register_hooks(self):
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _allreduce_delta(self,p):        
        handle = local_reduce_mean_async_(p.grad.data)
        
        if local_rank() == 0:
            group = None
            found = False
            for group in self.optimizer.param_groups:
                if any([p is v for v in group['params']]):
                    found = True
                    break
            assert group is not None
            assert found
            name = self._parameter_names.get(p)
            handle.wait()
            
            delta = self.optimizer.step_one(p, group)
            
            tensor_compressed, ctx = self._compression.compress(delta)
            handle = hvd.allreduce_async_(tensor_compressed.data, name=name, op=hvd.Adasum)
            return handle, ctx

        handle.wait()
        return handle, None

    def _allreduce_delta2(self,p):        
        handle = local_reduce_mean_async_(p.grad.data)
        
        if local_rank() == 0:
            stashed_params = []
            stashed_step = None
            for group in self.optimizer.param_groups:
                stashed_params.append(group['params'])
                # only want to step on p
                if any([p is v for v in group['params']]):
                    group['params'] = [p]
                    stashed_step = group.get('step', 0)
                else:
                    group['params'] = []

            start = self._starting_models[p]
            start.data.copy_(p)
            name = self._parameter_names.get(p)
            
            handle.wait()
            self.optimizer.step()
            p.data.sub_(start)

            # allreduce as before
            tensor_compressed, ctx = self._compression.compress(p)
            handle = hvd.allreduce_async_(tensor_compressed.data, name=name, op=hvd.Adasum)

            # reset stashed parameters
            for stashed, group in zip(stashed_params, self.optimizer.param_groups):
                group['params'] = stashed

            return handle, ctx

        handle.wait()
        return True, None #True -> Not None but we already waited
    
    def _allreduce_grad_async(self, p):
        #return self._allreduce_delta(p)
        #return self._allreduce_delta2(p)
        name = self._parameter_names.get(p)
        tensor = p.grad
        tensor_compressed, ctx = self._compression.compress(tensor)
        handle = local_allreduce_mean_async_(tensor_compressed.data)
        return handle, (ctx, tensor_compressed)

    def _make_hook(self, p):
        def hook(*ignore):
            if p in self._handles and self._handles[p][0] is not None:
                if self._allreduce_delay[p] <= 0:
                    raise AssertionError(
                        "Gradients were computed more than "
                        "backward_passes_per_step times before call "
                        "to step(). Increase backward_passes_per_step to "
                        "accumulate gradients locally.")
            assert not p.grad.requires_grad
            assert self._allreduce_delay[p] > 0
            handle, ctx = None, None
            self._allreduce_delay[p] -= 1
            if self._allreduce_delay[p] == 0:
                handle, ctx = self._allreduce_grad_async(p)
            self._handles[p] = (handle, ctx)
        return hook

    def synchronize(self):
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            handle, ctx = self._allreduce_grad_async(p)
            self._handles[p] = (handle, ctx)

        for p, value in self._handles.items():
            handle, ctx = value
            if handle is None:
                handle, ctx = self._allreduce_grad_async(p)
                self._handles[p] = (handle, ctx)

        for p, (handle, ctx) in self._handles.items():
            handle.wait()
            self._allreduce_delay[p] = self.backward_passes_per_step

        self._handles.clear()

    @contextmanager
    def skip_synchronize(self):
        raise AssertionError("Skipping synchronization is not supported when using Adasum optimizer.")

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        local_broadcast_handles = []
        if self._is_first:
            self._is_first = False
            for p in amp.master_params(self.optimizer):
                self._starting_models[p] = torch.zeros_like(p, requires_grad=False)
                self._starting_models[p].data.copy_(p.data)
                self._scalers[p] = DynamicLossScaler(init_scale=2**15)

        total_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 1.0)
        self.optimizer.step()
        
        handles = []
        for index, p in enumerate(amp.master_params(self.optimizer)):
            handle, ctx = None, None
            if index % num_devices == local_rank():
                name = self._parameter_names.get(p)
                scaler = self._scalers[p]
                start = self._starting_models[p]
                p.data.sub_(start)
                p.data.mul_(scaler.loss_scale)
                tensor_compressed, ctx = self._compression.compress(p)
                handle = hvd.allreduce_async_(tensor_compressed.data, name=name, op=hvd.Adasum)
            handles.append((handle, p, ctx))

        # keep all local amp scaler's in sync
        tmp_tensor = torch.cuda.FloatTensor([amp._amp_state.loss_scalers[0]._loss_scale])
        amp_handle = local_broadcast_async_(tmp_tensor, root=0)
        self.optimizer.zero_grad()
        amp_handle.wait()
        amp._amp_state.loss_scalers[0]._loss_scale = tmp_tensor.item()
            
        for index, (handle, p, ctx) in enumerate(handles):
            if index % num_devices == local_rank():
                start = self._starting_models[p]
                scaler = self._scalers[p]
                delta = hvd.synchronize(handle)
                has_overflow = not (torch.isfinite(delta.data).all().item())
                if not has_overflow:                
                    delta = self._compression.decompress(delta, ctx)
                    delta.data.div_(scaler.loss_scale)
                    start.data.add_(delta.data)
                p.data.copy_(start, True)
                local_broadcast_handles.append(local_broadcast_async_(p.data, root=index % num_devices))
                scaler.update_scale(has_overflow)
            else:
                local_broadcast_handles.append(local_broadcast_async_(p.data, root=index % num_devices))

        for handle in local_broadcast_handles:
            handle.wait()        
            
        # if we did fp16 training, apex adds the following method
        # to the optimizer.  It does not exist in fp32 training
        if hasattr(self.optimizer,"_master_params_to_model_params"):
            # tell apex to push master parameters to those parameters it uses for fp16
            # training            
            self.optimizer._master_params_to_model_params()
            
        return loss

    def zero_grad(self):
        if self._handles:
            raise AssertionError("optimizer.zero_grad() was called after loss.backward() "
                                 "but before optimizer.step() or optimizer.synchronize(). "
                                 "This is prohibited as it can cause a race condition.")
        return self.optimizer.zero_grad()

