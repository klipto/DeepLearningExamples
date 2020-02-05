# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

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

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import csv
import os
#os.environ['MASTER_ADDR'] = 'mmmramengw'
os.environ['MASTER_PORT'] = '52578'
os.environ['RANK'] = os.environ['PMI_RANK']
os.environ['WORLD_SIZE'] = os.environ['PMI_SIZE']
import time
import logging
import argparse
import random
import h5py
from tqdm import tqdm, trange
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import math
from apex import amp
import multiprocessing

from tokenization import BertTokenizer
from modeling import BertForPreTraining, BertConfig
from apex.optimizers import FusedLAMB, FusedAdam
from schedulers import PolyWarmUpScheduler

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
#from utils import is_main_process
#from apex.parallel import DistributedDataParallel as DDP
from schedulers import LinearWarmUpScheduler
#from apex.parallel.distributed import flat_dist_call
import amp_C
import apex_C
from apex.amp import _amp_state

from apex.fp16_utils.loss_scaler import DynamicLossScaler    
import horovod.torch as hvd

from concurrent.futures import ProcessPoolExecutor

def is_main_process():
    return hvd.rank() == 0

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def create_pretraining_dataset(input_file, max_pred_length, shared_list, args):

    train_data = pretraining_dataset(input_file=input_file, max_pred_length=max_pred_length)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size * args.n_gpu, num_workers=4,
                                  pin_memory=True)
    return train_dataloader, input_file

class pretraining_dataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, next_sentence_labels]

def parse_arguments():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .hdf5 files  for the task.")

    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")

    parser.add_argument("--bert_model", default="bert-large-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="The initial checkpoint to start training from.")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_predictions_per_seq",
                        default=80,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=1000,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=float, default=1.0,
                        help='frequency of logging loss.')
    parser.add_argument('--checkpoint_activations',
                        default=False,
                        action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to resume training from checkpoint.")
    parser.add_argument('--resume_step',
                        type=int,
                        default=-1,
                        help="Step to resume training from.")
    parser.add_argument('--num_steps_per_checkpoint',
                        type=int,
                        default=100,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--phase2',
                        default=False,
                        action='store_true',
                        help="Whether to train with seq len 512")
    parser.add_argument('--allreduce_post_accumulation',
                        default=False,
                        action='store_true',
                        help="Whether to do allreduces during gradient accumulation steps.")
    parser.add_argument('--allreduce_post_accumulation_fp16',
                        default=False,
                        action='store_true',
                        help="Whether to do fp16 allreduce post accumulation.")
    parser.add_argument('--phase1_end_step',
                        type=int,
                        default=7038,
                        help="Number of training steps in Phase1 - seq len 128")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    args = parser.parse_args()
    return args

def setup_training(args):

    assert (torch.cuda.is_available())
    assert args.local_rank == -1
    args.local_rank = int(os.environ['RANK']) % 4
    if args.local_rank == -1:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        #torch.distributed.init_process_group(backend='nccl', init_method='env://')
        hvd.init()

    print("device {} n_gpu {} distributed training {}".format(device, args.n_gpu, bool(args.local_rank != -1)), flush=True)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
            args.gradient_accumulation_steps, args.train_batch_size))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train:
        raise ValueError(" `do_train`  must be True.")

    if not args.resume_from_checkpoint and os.path.exists(args.output_dir) and (
            os.listdir(args.output_dir) and any([i.startswith('ckpt') for i in os.listdir(args.output_dir)])):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if not args.resume_from_checkpoint:
        os.makedirs(args.output_dir, exist_ok=True)

    return device, args

def prepare_model_and_optimizer(args, device):

    # Prepare model
    config = BertConfig.from_json_file(args.config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model = BertForPreTraining(config)

    checkpoint = None
    if not args.resume_from_checkpoint:
        global_step = 0
    else:
        if args.resume_step == -1 and not args.init_checkpoint:
            model_names = [f for f in os.listdir(args.output_dir) if f.endswith(".pt")]
            args.resume_step = max([int(x.split('.pt')[0].split('_')[1].strip()) for x in model_names])

        global_step = args.resume_step if not args.init_checkpoint else 0

        if not args.init_checkpoint:
            checkpoint = torch.load(os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step)), map_location="cpu")
        else:
            checkpoint = torch.load(args.init_checkpoint, map_location="cpu")

        model.load_state_dict(checkpoint['model'], strict=False)
        if args.phase2:
            global_step -= args.phase1_end_step
        if is_main_process():
            print("resume step from ", args.resume_step, flush=True)

    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = FusedLAMB(optimizer_grouped_parameters,
                          betas=(0.9, 0.999),
                          lr=args.learning_rate)
    lr_scheduler = PolyWarmUpScheduler(optimizer, 
                                       warmup=args.warmup_proportion, 
                                       total_steps=args.max_steps)
    if args.fp16:

        if args.loss_scale == 0:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale="dynamic")
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=args.loss_scale)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    if args.resume_from_checkpoint:
        if args.phase2 or args.init_checkpoint:
            keys = list(checkpoint['optimizer']['state'].keys())
            #Override hyperparameters from previous checkpoint
            for key in keys:
                checkpoint['optimizer']['state'][key]['step'] = global_step
            for iter, item in enumerate(checkpoint['optimizer']['param_groups']):
                checkpoint['optimizer']['param_groups'][iter]['step'] = global_step
                checkpoint['optimizer']['param_groups'][iter]['t_total'] = args.max_steps
                checkpoint['optimizer']['param_groups'][iter]['warmup'] = args.warmup_proportion
                checkpoint['optimizer']['param_groups'][iter]['lr'] = args.learning_rate
        optimizer.load_state_dict(checkpoint['optimizer'])  # , strict=False)

        # Restore AMP master parameters          
        if args.fp16:
            optimizer._lazy_init_maybe_master_weights()
            optimizer._amp_stash.lazy_init_called = True
            optimizer.load_state_dict(checkpoint['optimizer'])
            for param, saved_param in zip(amp.master_params(optimizer), checkpoint['master params']):
                param.data.copy_(saved_param.data)

    if args.local_rank != -1:
        if not args.allreduce_post_accumulation:
            model = DDP(model, message_size=250000000, gradient_predivide_factor=torch.distributed.get_world_size())
        else:
            #flat_dist_call([param.data for param in model.parameters()], torch.distributed.broadcast, (0,) )
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model, optimizer, lr_scheduler, checkpoint, global_step

def take_optimizer_step(args, optimizer, model, overflow_buf, adasum_scalar, global_step):
    
    if args.allreduce_post_accumulation:
        # manually allreduce gradients after all accumulation steps
        # check for Inf/NaN
        # 1. allocate an uninitialized buffer for flattened gradient
        scaler = _amp_state.loss_scalers[0]

        # 2. update loss scale
        hvd.allreduce_(scaler._overflow_buf, op=hvd.Sum)

        had_overflow = scaler._overflow_buf.item() > 0
        scaler._has_overflow = had_overflow
        tmp = scaler.update_scale()
        assert tmp == had_overflow
                
        # 4. call optimizer step function
        if had_overflow == False:

            params = [p for p in amp.master_params(optimizer) if p.grad is not None]
            starts = [p.clone().detach() for p in params]
            
            optimizer.step()

            handles = []
            for index, (start, current) in enumerate(zip(starts, params)):
                current.data.sub_(start)
                norm_sq = current.data.norm(p=2,dtype=torch.float32)**2
                current.data.mul_(adasum_scalar.loss_scale)
                delta = current.data.to(torch.float16)
                adasum_handle = hvd.allreduce_async_(delta, name='all%i'%index, op=hvd.Adasum)
                normsq_handle = hvd.allreduce_async(norm_sq,name='nsq%i'%index, op=hvd.Sum)
                handles.append((adasum_handle, normsq_handle, start, current))

            overflow_buf.zero_()
            deltas = [
                hvd.synchronize(adasum_handle).to(torch.float32)                
                for index, (adasum_handle, normsq_handle, start, current) in enumerate(handles)
            ]
            amp_C.multi_tensor_scale(65536,
                                     overflow_buf,
                                     [deltas, deltas],
                                     1.0 / adasum_scalar.loss_scale)
            
            adasum_had_overflow = overflow_buf.item() == 1
            
            for index, (adasum_handle, normsq_handle, start, current) in enumerate(handles):
                delta = deltas[index]
                a = hvd.synchronize(normsq_handle)
                
                if adasum_had_overflow == False and is_main_process():
                    print("shadow", index, delta.norm(p=2).item() / torch.sqrt(a).item(), flush=True)
                    
                if not adasum_had_overflow:
                    start.add_(delta)
                                    
                current.data.copy_(start)

            adasum_scalar.update_scale(adasum_had_overflow)
            
            if adasum_had_overflow and is_main_process():
                print("Layer {} had overflow: new scale {}".format(
                    index, adasum_scalar.loss_scale))

            if not adasum_had_overflow:
                global_step += 1
            
        else:
            # Overflow detected, print message and clear gradients
            if is_main_process():
                print(("Rank {} :: Gradient overflow.  Skipping step, "  +
                        "reducing loss scale to {}").format(
                            hvd.rank(),
                            scaler.loss_scale()), flush=True)
            if _amp_state.opt_properties.master_weights:
                for param in optimizer._amp_stash.all_fp32_from_fp16_params:
                    param.grad = None
                    
        scaler.clear_overflow_state()
        
        for param in model.parameters():
            param.grad = None
            
    else:
        optimizer.step()
        #optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None
        global_step += 1

    return global_step

def main():

    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device, args = setup_training(args)

    # Prepare optimizer
    model, optimizer, lr_scheduler, checkpoint, global_step = prepare_model_and_optimizer(args, device)

    if is_main_process():
        print("SEED {}".format(args.seed),flush=True)

    if args.do_train:
        if is_main_process():
            print("***** Running training *****", flush=True)
            # logger.info("  Num examples = %d", len(train_data))
            print("  Batch size =", args.train_batch_size, flush=True)
            print("  LR = ", args.learning_rate, flush=True)
            print("Training. . .", flush=True)

        model.train()
        most_recent_ckpts_paths = []
        average_loss = 0.0  # averaged loss every args.log_freq steps
        epoch = 0
        training_steps = 0
        batch_start = time.time()
        
        pool = ProcessPoolExecutor(1)
        adasum_scalar = DynamicLossScaler(init_scale=2**18)
        
        # Note: We loop infinitely over epochs, termination is handled via iteration count
        while True:
            thread = None
            if not args.resume_from_checkpoint or epoch > 0 or (args.phase2 and global_step < 1) or args.init_checkpoint:
                files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                         os.path.isfile(os.path.join(args.input_dir, f)) and 'training' in f]
                files.sort()
                num_files = len(files)
                random.shuffle(files)
                f_start_id = 0
            else:
                f_start_id = checkpoint['files'][0]
                files = checkpoint['files'][1:]
                args.resume_from_checkpoint = False
                num_files = len(files)


            shared_file_list = {}

            if hvd.size() > num_files:
                remainder = hvd.size() % num_files
                data_file = files[(f_start_id*hvd.size()+hvd.rank() + remainder*f_start_id)%num_files]
            else:
                data_file = files[(f_start_id*hvd.size()+hvd.rank())%num_files]

            previous_file = data_file

            train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                          batch_size=args.train_batch_size * args.n_gpu, num_workers=4,
                                          pin_memory=True)
            # shared_file_list["0"] = (train_dataloader, data_file)

            overflow_buf = None
            if args.allreduce_post_accumulation:
                overflow_buf = torch.cuda.IntTensor([0])
            
            if len(files) == 1:
                f_start_id = -1
            for f_id in range(f_start_id + 1 , len(files)):
                
   
                if hvd.size() > num_files:
                    data_file = files[(f_id*hvd.size()+hvd.rank() + remainder*f_id)%num_files]
                else:
                    data_file = files[(f_id*hvd.size()+hvd.rank())%num_files]

                print("file no %s file %s" % (f_id, previous_file))

                previous_file = data_file

                dataset_future = pool.submit(create_pretraining_dataset, data_file, args.max_predictions_per_seq, shared_file_list, args)

                #train_iter = tqdm(train_dataloader, desc="Iteration") if is_main_process() else train_dataloader
                train_iter = train_dataloader
                for step, batch in enumerate(train_iter):

                    training_steps += 1
                    batch = [t.to(device) for t in batch]
                    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
                    loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                    masked_lm_labels=masked_lm_labels, next_sentence_label=next_sentence_labels,
                                    checkpoint_activations=args.checkpoint_activations)
                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.

                    divisor = args.gradient_accumulation_steps
                    if args.gradient_accumulation_steps > 1:
                        if True:#not args.allreduce_post_accumulation:
                            # this division was merged into predivision
                            loss = loss / args.gradient_accumulation_steps
                            divisor = 1.0
                    if args.fp16:
                        with amp.scale_loss(loss, optimizer, delay_overflow_check=args.allreduce_post_accumulation) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    average_loss += loss.item()

                    if training_steps % args.gradient_accumulation_steps == 0:
                        lr_scheduler.step()  # learning rate warmup
                        global_step = take_optimizer_step(args, optimizer, model, overflow_buf, adasum_scalar, global_step)

                    if global_step >= args.max_steps:
                        last_num_steps = int(training_steps / args.gradient_accumulation_steps) % args.log_freq
                        last_num_steps = args.log_freq if last_num_steps == 0 else last_num_steps
                        average_loss = torch.tensor(average_loss, dtype=torch.float32).cuda()
                        average_loss = average_loss / (last_num_steps * divisor)
                        average_loss /= hvd.size()
                        hvd.allreduce_(average_loss)
                        if is_main_process():
                            print("Total Steps:{} Final Loss = {}".format(training_steps / args.gradient_accumulation_steps, average_loss.item()), flush=True)
                    elif training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                        if hvd.rank() == 0:
                            print("Step:{} Average Loss = {} Step Loss = {} LR {} time {}".format(
                                global_step,
                                average_loss / (args.log_freq * divisor),
                                loss.item() * args.gradient_accumulation_steps / divisor,
                                optimizer.param_groups[0]['lr'],
                                time.time() - batch_start), flush=True)
                        batch_start = time.time()
                        average_loss = 0
                        
                    if global_step >= args.max_steps or training_steps % (
                            args.num_steps_per_checkpoint * args.gradient_accumulation_steps) == 0:
                        if is_main_process():
                            # Save a trained model
                            print("** ** * Saving fine - tuned model ** ** * ")
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Only save the model it-self
                            if args.resume_step < 0 or not args.phase2:
                                output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step))
                            else:
                                output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step + args.phase1_end_step))
                            if args.do_train:
                                torch.save({'model': model_to_save.state_dict(),
                                            'optimizer': optimizer.state_dict(),
                                            'master params': list(amp.master_params(optimizer)),
                                            'files': [f_id] + files}, output_save_file)

                                most_recent_ckpts_paths.append(output_save_file)
                                if len(most_recent_ckpts_paths) > 3:
                                    ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                                    os.remove(ckpt_to_be_removed)

                        if global_step >= args.max_steps:
                            del train_dataloader
                            # thread.join()
                            return args

                del train_dataloader
                # thread.join()
                # Make sure pool has finished and switch train_dataloader
                # NOTE: Will block until complete
                train_dataloader, data_file = dataset_future.result(timeout=None)

            epoch += 1


if __name__ == "__main__":
    now = time.time()
    args = main()
    if is_main_process():
        print("Total time taken {}".format(time.time() - now), flush=True)
