# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch optimization for BERT model."""

import math
import torch
import horovod.torch as hvd
import dist
import torch.distributed
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
#from fused_adam_local import FusedAdam
from apex.optimizers import FusedAdam
from apex.multi_tensor_apply import multi_tensor_applier
import amp_C

from apex import amp

multi_tensor_l2norm = amp_C.multi_tensor_l2norm
lamb_compute_update = amp_C.multi_tensor_lamb_stage1_cuda
lamb_apply_update = amp_C.multi_tensor_lamb_stage2_cuda
scale = amp_C.multi_tensor_scale


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return max((x - 1. )/ (warmup - 1.), 0.)
    
def warmup_poly(x, warmup=0.002, degree=0.5):
    if x < warmup:
        return x/warmup
    return (1.0 - x)**degree


SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
    'warmup_poly':warmup_poly,
}


class BertLAMB(Optimizer):
    """Implements BERT version of LAMB algorithm.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: LAMBs b1. Default: 0.9
        b2: LAMBs b2. Default: 0.999
        e: LAMBs epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum global norm for the gradients. Default: 1.0
    """
    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_poly',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertLAMB, self).__init__(params, defaults)
        self.step_count = 0
        self.b1 = b1
        self.b2 = b2
        self.epsilon = e
        self.max_global_grad_norm = max_grad_norm
        self.learning_rate = lr
        self.schedule = schedule
        self.warmup = warmup
        self.max_steps = t_total
        self.updates_created=False

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def apply_gradients(self, dummy_overflow_buf, lr_scheduled, per_param_decay, grad_list, param_list, momentum, velocity, update):
        # Compute global gradient norm
        global_grad_norm = multi_tensor_applier(
                        multi_tensor_l2norm,
                        dummy_overflow_buf,
                        [grad_list],
                        False)[0].item()

        # Compute per parameter norm
        param_norms = multi_tensor_applier(
                        multi_tensor_l2norm,
                        dummy_overflow_buf,
                        [param_list],
                        True)[1]

        # Compute LAMB update
        multi_tensor_applier(
                        lamb_compute_update,
                        dummy_overflow_buf,
                        [grad_list, param_list, momentum, velocity, update],
                        torch.cuda.FloatTensor(per_param_decay),
                        self.step_count,
                        self.b1,
                        self.b2,
                        self.epsilon,
                        global_grad_norm,
                        self.max_global_grad_norm,
                        )

        # Computer per parameter update norm
        update_norms = multi_tensor_applier(
                        multi_tensor_l2norm,
                        dummy_overflow_buf,
                        [update],
                        True)[1]

        # Apply LAMB update on parameters
        multi_tensor_applier(
                        lamb_apply_update,
                        dummy_overflow_buf,
                        [param_list, update],
                        param_norms,
                        update_norms,
                        lr_scheduled,
                        )

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        check = 1#torch.norm(all_grads, 2)

        grad_list = []
        param_list = []
        per_param_decay = []
        momentum = []
        velocity = []

        fp16_grad_list = []
        fp16_from_fp32_param_list = []
        fp32_param_list = []
        fp16_per_param_decay = []
        fp16_momentum = []
        fp16_velocity = []
        
        if not self.updates_created:
            self.update = []
            self.fp16_update = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Keep step here for compatibility with earlier resume from checkpoint
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['momentum'] = torch.zeros_like(p.data, dtype=torch.float32)
                    # Exponential moving average of squared gradient values
                    state['velocity'] = torch.zeros_like(p.data, dtype=torch.float32)
                    # fp32 master weights
                if 'master_param' not in state.keys() and p.type() == 'torch.cuda.HalfTensor':
                    state['master_param'] = p.detach().clone().float()

                # ensure these 3 are float tensors
                if state['momentum'].type() != 'torch.cuda.FloatTensor':
                    state['momentum'] = state['momentum'].float()
                if state['velocity'].type() != 'torch.cuda.FloatTensor':
                    state['velocity'] = state['velocity'].float()
                if 'master_param' in state.keys() and state['master_param'].type() != 'torch.cuda.FloatTensor':
                    state['master_param'] = state['master_param'].float()

                # Append all params, gradients, decays, velocity, momentum and updates to a list
                if p.type() == 'torch.cuda.HalfTensor':
                    fp16_grad_list.append(grad)
                    fp32_param_list.append(state['master_param'])
                    fp16_from_fp32_param_list.append(p.data)
                    fp16_per_param_decay.append(group['weight_decay'])
                    fp16_momentum.append(state["momentum"])
                    fp16_velocity.append(state["velocity"])
                    if not self.updates_created:
                        #self.fp16_update.append(torch.empty_like(p.data, dtype=torch.float32))
                        # Use fp16 weights as temporary buffer for update term.
                        # This is safe because fp16 weights are overwritten after apply_gradients
                        self.fp16_update.append(p.data)
                else:
                    grad_list.append(grad)
                    param_list.append(p.data)
                    per_param_decay.append(group['weight_decay'])
                    momentum.append(state["momentum"])
                    velocity.append(state["velocity"])
                    if not self.updates_created:
                        self.update.append(torch.empty_like(p.data))
                state['step'] += 1
        self.updates_created=True
        update = self.update
        fp16_update = self.fp16_update

        self.step_count = state['step']
        # Calculate learning rate from input schedule
        # if self.max_steps != -1:
        schedule_fct = SCHEDULES[self.schedule]
        lr_scheduled = self.learning_rate * schedule_fct(self.step_count / self.max_steps, self.warmup)
        if hvd.rank() == 0 :#torch.distributed.get_rank() == 0:
            print("Step {} LR {}".format(self.step_count, lr_scheduled))
        # else:
        #     lr_scheduled = self.learning_rate

        overflow_buf = torch.cuda.IntTensor([0])

        if len(grad_list) > 0:
            self.apply_gradients(overflow_buf, lr_scheduled, per_param_decay, grad_list, param_list, momentum, velocity, update)
        if len(fp16_grad_list) > 0:
            self.apply_gradients(overflow_buf, lr_scheduled, fp16_per_param_decay, fp16_grad_list, fp32_param_list, fp16_momentum, fp16_velocity, fp16_update)
            multi_tensor_applier(
                    scale,
                    overflow_buf,
                    [fp32_param_list, fp16_from_fp32_param_list],
                    1.)

        return loss

class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr


    def step_one(self, p, group):
        grad = p.grad.data
        if grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['next_m'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['next_v'] = torch.zeros_like(p.data)

        next_m, next_v = state['next_m'], state['next_v']
        beta1, beta2 = group['b1'], group['b2']

        # Add grad clipping
        if group['max_grad_norm'] > 0:
            clip_grad_norm_(p, group['max_grad_norm'])

        # Decay the first and second moment running average coefficient
        # In-place operations to update the averages at the same time
        next_m.mul_(beta1).add_(1 - beta1, grad)
        next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        update = next_m / (next_v.sqrt() + group['e'])

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want to decay the weights in a manner that doesn't interact
        # with the m/v parameters. This is equivalent to adding the square
        # of the weights to the loss with plain (non-momentum) SGD.
        if group['weight_decay'] > 0.0:
            update += group['weight_decay'] * p.data

        if group['t_total'] != -1:
            schedule_fct = SCHEDULES[group['schedule']]
            lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
        else:
            lr_scheduled = group['lr']

        update_with_lr = lr_scheduled * update
        return -update_with_lr
    
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                #update_with_lr = self.step_one(p)
                #p.data.add_(-update_with_lr)
                state = self.state[p]
                # State initialization
                #if len(state) == 0:
                #    state['step'] = 0
                state['step'] += 1
                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss

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
        #p : torch.zeros_like(p, requires_grad=False)
        #    for _, p in named_parameters
        #}

        self._scalers = {}
        #p : DynamicLossScaler()
        #    for _, p in named_parameters
        #}

        self._register_hooks()

    def set_backward_passes_per_step(self, passes):
        self.backward_passes_per_step = passes
        for p in self._allreduce_delay:
            self._allreduce_delay[p] = self.backward_passes_per_step

    def _register_hooks(self):
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                #for p in amp.master_params(self.optimizer):
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _allreduce_delta(self,p):        
        handle = dist.local_reduce_mean_async_(p.grad.data)
        
        if dist.local_rank() == 0:
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
        handle = dist.local_reduce_mean_async_(p.grad.data)
        
        if dist.local_rank() == 0:
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
        handle = dist.local_reduce_mean_async_(tensor_compressed.data)
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

        #local_handles = []
        for p, (handle, ctx) in self._handles.items():
            #if dist.local_rank() == 0:
            #    output = hvd.synchronize(handle)
            #    p.data.add_(self._compression.decompress(output, ctx).data)
            #local_handles.append(dist.local_broadcast_async_(p.data))
            handle.wait()
            self._allreduce_delay[p] = self.backward_passes_per_step

        self._handles.clear()

        #for handle in local_handles:
        #    handle.wait()

    @contextmanager
    def skip_synchronize(self):
        raise AssertionError("Skipping synchronization is not supported when using Adasum optimizer.")

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        #self.synchronize()
        #if dist.local_rank() == 0:
        #    self.optimizer.step()
        #return loss
    
        #if self._is_first:
        #    self._is_first = False

        
        local_broadcast_handles = []
        if self._is_first and dist.local_rank() == 0:
            self._is_first = False
            #for group in self.optimizer.param_groups:
            #    for p in group['params']:
            for p in amp.master_params(self.optimizer):
                #if p.grad is None:
                #    continue
                #self._starting_models[p].data.copy_(p.data)
                self._starting_models[p] = torch.zeros_like(p, requires_grad=False)
                self._starting_models[p].data.copy_(p.data)
                self._scalers[p] = DynamicLossScaler()

        if dist.local_rank() == 0:
            total_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 1.0)
            self.optimizer.step()
            
            handles = []
            #for group in self.optimizer.param_groups:
            #    for p in group['params']:
            for p in amp.master_params(self.optimizer):
                ##if p.grad is None:
                #    continue
                name = self._parameter_names.get(p)
                scaler = self._scalers[p]
                start = self._starting_models[p]
                p.data.sub_(start)
                p.data.mul_(scaler.loss_scale)
                #tensor_compressed, ctx = p, None
                tensor_compressed, ctx = self._compression.compress(p)
                handle = hvd.allreduce_async_(tensor_compressed.data, name=name, op=hvd.Adasum)
                handles.append((handle, p, ctx))

            for handle, p, ctx in handles:
                start = self._starting_models[p]
                scaler = self._scalers[p]
                delta = hvd.synchronize(handle)
                has_overflow = not (torch.isfinite(delta.data).all().item())
                if not has_overflow:                
                    delta = self._compression.decompress(delta, ctx)
                    delta.data.div_(scaler.loss_scale)
                    start.data.add_(delta.data)
                #start.data.add_(delta.data)
                p.data.copy_(start)
                local_broadcast_handles.append(dist.local_broadcast_async_(p.data))                
                scaler.update_scale(has_overflow)

        else:
            #for group in self.param_groups:
            #    for p in group['params']:
            for p in amp.master_params(self.optimizer):
                #if p.grad is None:
                #        continue
                local_broadcast_handles.append(dist.local_broadcast_async_(p.data))

        for handle in local_broadcast_handles:
            handle.wait()

        return loss

    def zero_grad(self):
        if self._handles:
            raise AssertionError("optimizer.zero_grad() was called after loss.backward() "
                                 "but before optimizer.step() or optimizer.synchronize(). "
                                 "This is prohibited as it can cause a race condition.")
        return self.optimizer.zero_grad()
