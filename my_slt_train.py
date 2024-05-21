import torch
from model.slt_transformer import Transformer, LSTMModel
from transformers import BertTokenizer
from model.slt_dataset_v2 import TranslationDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import math
import os
import time
from torch.autograd import Variable
from model.util import subsequent_mask
import numpy as np
from datasets import load_metric
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel 
from torch.nn.parallel import DistributedDataParallel
import wandb
import random
from fvcore.nn import FlopCountAnalysis

def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("ERROR MAKE DIR")
import argparse, sys



dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
world_size = dist.get_world_size()
dist_url = 'env://'
torch.cuda.set_device(local_rank)
torch.distributed.barrier()
if torch.distributed.get_rank() == 0:
  print("RANK:0")
  #wandb.init(project='paper_verion_korean_slt')
if torch.distributed.get_rank() == 1:
  print("RANK:1")
  
parser = argparse.ArgumentParser()
parser.add_argument('-type', required=True, type=int, help='2010') 
parser.add_argument('-lr', type=float, help='learning rate')
parser.add_argument('-body_part', type=str, default='hands_body_face', help='hands, hands_body, hands_face, hands_body_face')
parser.add_argument('-seed', type=int, default=42, help=' seed')
parser.add_argument('-do_max_len_50', type=bool, default=False)#f prediction(default)', default='2021-11-02')
args = parser.parse_args()
print(args)

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

metric = load_metric("sacrebleu")
rouge = load_metric('rouge')
meteor = load_metric('meteor')
#torch.distributed.init_process_group(backend='nccl',
#                            init_method=dist_url,
#                            world_size=world_size,
#                            rank=local_rank)



class TranslationTrainer():
  def __init__(self,
               dataset,
               tokenizer,
               model,
               max_len,
               device,
               model_name,
               checkpoint_path,
               batch_size,
               ):
    self.dataset = dataset
    self.tokenizer = tokenizer
    self.model = model
    self.max_len = max_len
    self.model_name = model_name
    self.checkpoint_path = checkpoint_path
    self.device = device
    self.ntoken = tokenizer.vocab_size
    self.batch_size = batch_size
  def my_collate_fn(self, samples):
    target_str =[]
    input = [sample['input'] for sample in samples]
    token_type = [sample['token_type'] for sample in samples]
    
    input_mask = [sample['input_mask'] for sample in samples]
    target = [sample['target'] for sample in samples]
    target_mask = [sample['target_mask'] for sample in samples]
    token_num = [sample['token_num'] for sample in samples]
    target_str.append([sample['target_str'] for sample in samples])

    return {
        "input":torch.stack(input).contiguous(),
        "token_type":torch.stack(token_type).contiguous(),

        "input_mask": torch.stack(input_mask).contiguous(),       # input_mask
        "target": torch.stack(target).contiguous(),                                           # target,
        "target_mask": torch.stack(target_mask).contiguous(),   # target_mask
        "token_num": torch.stack(token_num).contiguous(),   # token_num
        "target_str": target_str
    }
  def build_dataloaders(self, train_test_split=0.1, train_shuffle=True, eval_shuffle=True):
    dataset_len = len(self.dataset)
    eval_len = int(dataset_len * train_test_split)
    train_len = dataset_len - eval_len
    train_dataset, eval_dataset = random_split(self.dataset, (train_len, eval_len))
    
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    #batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, self.batch_size, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=16, collate_fn=self.my_collate_fn)
  
  
    eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size*16, shuffle=eval_shuffle , collate_fn=self.my_collate_fn)

    return train_loader, eval_loader

  def train(self, epochs, train_dataset, eval_dataset, optimizer, scheduler):
    self.model.train()
    total_loss = 0.
    global_steps = 0
    start_time = time.time()
    losses = {}
    best_val_loss = float("inf")
    best_model = None
    start_epoch = 0
    start_step = 0
    total_flops = 0  # 전체 FLOPS 누적을 위한 변수
    train_dataset_length = len(train_dataset)

    self.model.to(self.device)
    for epoch in range(start_epoch, epochs):
      train_dataset.sampler.set_epoch(epoch)
      epoch_start_time = time.time()

      pb = tqdm(enumerate(train_dataset),
                desc=f'Epoch-{epoch} Iterator',
                total=train_dataset_length,
                bar_format='{l_bar}{bar:10}{r_bar}'
                )
      pb.update(start_step)
      for i,data in pb:
        if i < start_step:
          continue
        input = data['input'].to(self.device)
        target = data['target'].to(self.device)
        input_mask = data['input_mask'].to(self.device)
        target_mask = data['target_mask'].to(self.device)
        token_type = data['token_type'].to(self.device)
        optimizer.zero_grad()
        generator_logit, loss = self.model.forward(input, token_type, target, input_mask, target_mask, labels=target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        losses[global_steps] = loss.item()
        total_loss += loss.item()
        log_interval = 100
        save_interval = 500
        global_steps += 1
                        
        
                        
        if i % log_interval == 0 and i > 0:
          cur_loss = total_loss / log_interval
          elapsed = time.time() - start_time
          pb.set_postfix_str('| epoch {:3d} | {:5d}/{:5d} batches | '
                             'lr {:02.5f} | ms/batch {:5.2f} | '
                             'loss {:5.2f} | ppl {:8.2f}'.format(
            epoch, i, len(train_dataset), scheduler.get_lr()[0],
            elapsed * 1000 / log_interval,
            cur_loss, math.exp(cur_loss)))
          
          if torch.distributed.get_rank() == 0:
            wandb.log({"loss": cur_loss, 'learning_rate':scheduler.get_lr()[0]}, step=global_steps)
            #wandb.log({"FLOPS": total_flops}, step=global_steps)
          total_loss = 0
          start_time = time.time()
          # self.save(epoch, self.model, optimizer, losses, global_steps)
          if i % save_interval == 0:
            self.save(epoch, self.model, optimizer, losses, global_steps)
      print(len(eval_dataset))
      if (local_rank == 0 and (epoch) % 46 == 0) or epoch == 47 or epoch == 48 or epoch == 49:
        perf = self.evaluate(eval_dataset, global_steps) ## 'bleu, loss'
        #val_loss = perf['loss']
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | bleu {:5.2f} | rougeL {:5.2f} | meteor {:5.2f}'.format(epoch, (time.time() - epoch_start_time), perf['bleu'],perf['rouge'],perf['meteor']))
        print('-' * 89)
        """
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_model = model
        """
      torch.cuda.empty_cache() 
      start_step = 0
      self.model.train()
      scheduler.step()


  
  def compute_metrics(self,decoded_preds, decoded_labels):
    #preds, labels = eval_preds     # 모델이 예측 로짓(logits)외에 다른 것을 리턴하는 경우.
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels) 
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)['rougeL'][1][2]
    meteor_result = meteor.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"], 'meteor':meteor_result['meteor'], 'rouge':rouge_result}

  
  def evaluate(self, dataset, global_steps, max_length=50):
    self.model.eval()
    total_bleu = 0.
    total_rouge = 0.
    total_meteor = 0.
    self.model.to(self.device)

    with torch.no_grad():
        for batch in dataset:
          preds = []
          labels = []
          for i in range(len(batch['input'])):
            encoder_input = batch['input'][i].to(self.device)
            encoder_mask = batch['input_mask'][i].to(self.device)
            
            target = torch.ones(1, 1).fill_(tokenizer.cls_token_id).long().to(self.device)

            encoder_output = self.model.module.encode(encoder_input, encoder_mask)  # Access the underlying model using 'module'

            for _ in range(max_length - 1):
              lm_logits = self.model.module.decode(encoder_output, encoder_mask, target, subsequent_mask(target.size(-1)).type_as(encoder_input.data))  # Access the underlying model using 'module'
              prob = lm_logits[:, -1]
              _, next_word = torch.max(prob, dim=1)


              target = torch.cat((target, next_word.unsqueeze(-1)), dim=1)

            # 결과 문자열 디코딩
            target = target.squeeze()

            
            if tokenizer.pad_token_id in target:
              pad_index = torch.where(target == tokenizer.pad_token_id)[0][0]
              target = target[:pad_index+1]
              
            if tokenizer.sep_token_id in target:
              sep_index = torch.where(target == tokenizer.sep_token_id)[0][0]
              target = target[:sep_index+1]
          
            preds.append(target)
            labels.append(batch['target'][i])
            
          preds = tokenizer.batch_decode(preds,skip_special_tokens=True)
          labels = tokenizer.batch_decode(labels,skip_special_tokens=True)
          result_metric = self.compute_metrics(preds, labels)
          total_bleu += result_metric['bleu']
          total_rouge += result_metric['rouge']
          total_meteor += result_metric['meteor']

    return_item = {
        'meteor': total_meteor / len(dataset), 
        'bleu': total_bleu / len(dataset), 
        'rouge': total_rouge / len(dataset)
    }

    if torch.distributed.get_rank() == 0:
        wandb.log(return_item, step=global_steps)

    torch.cuda.empty_cache()
    return return_item

    
  def save(self, epoch, model, optimizer, losses, train_step):
    torch.save({
      'epoch': epoch,  # 현재 학습 epoch
      'model_state_dict': model.state_dict(),  # 모델 저장
      'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
      'losses': losses,  # Loss 저장
      'train_step': train_step,  # 현재 진행한 학습
    }, f'{self.checkpoint_path}/{self.model_name}.pth')




if __name__ == '__main__':
  other_exp = False
  type_list = [3000,5037,3010,3005,2010,2005,1510,1505,8888,3023,6045, 2015, 1500, 2000, 5035, 1511, 4030, 6045, 8060, 3020, 3010, 4000]
  
  input_dict = {3000:1216, 3010:1216, 3005:1216,3023:1216, 2010:836, 2005:836, 1510:608,1505:608, 8888:248, 6045:2356, 2015:836, 1500:608,4000:1596, 2000:836, 5035:1976, 5037:1976, 1511:608, 4030:1596, 6045:2356, 8060:3116, 3020:1216, 3010:1216, 2000:836, 1500:608}
  max_len_dict = {3000:94, 3009:94, 3023:266, 3005:76, 2010:188, 2005:126, 1510:373, 1505:189, 8888:200, 6045:100, 2015:374, 1500:125, 2000:100, 4000:160, 5035:123, 5037:150, 1511:400, 4030:185, 6045:122, 8060:80, 3020:288, 3010:150, 2000:126, 1500:189, 9999:50}

  if not args.type in type_list:
      print("TYPE ERROR")
      sys.exit()

  #4030: body = 10, face = 36, hands = 30, 
  #8888: body = 12, face = 70, hands = 42
  if args.body_part == 'hands':
    input_dict = {4030:630, 8888:84}
  elif args.body_part == 'hands_body':
    input_dict = {4030:840, 8888:108}
  elif args.body_part == 'hands_face':
    input_dict = {4030:1386, 8888:224}
  
  
  if args.type == 8888 or args.type == 4030 or args.type == 9999:
      print("############### other experiment ###########")
      other_exp = True
      max_len_dict[8888] = 50
      max_len_dict[4030] = 50
      
  dir_path = './'
  vocab_path = f'vocab.txt'
  vocab_num = 22000
  data_path = './slt_stft_data/our_processed_data_{}_{}/'.format(args.type//100, args.type%100)
  max_length = max_len_dict[args.type]
  input_d = input_dict[args.type]
  d_model = 256
  head_num = d_model//64
  dropout = 0.1
  N = 3
  device = 'cuda'
  tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)
  
  # hyper parameter
  epochs = 50
  batch_size = 32
  padding_idx = tokenizer.pad_token_id
  learning_rate = args.lr
  split_dim = None#[288,120,120,80]
  config = {
  "vocab_num": vocab_num,
  "max_length": max_length,
  "input_d": input_d,
  "d_model": d_model,
  "head_num": head_num,
  "dropout": dropout,
  "N": N,
  "epochs": epochs,
  "batch_size_per_gpu": batch_size,
  "learning_rate": learning_rate,
  "split_dim": str(split_dim)
  }
  
  project_name = 'seed_{}_{}_{}_{}_{}_{}'.format(args.type,d_model,str(learning_rate),max_length,args.body_part,seed)
  if torch.distributed.get_rank() == 0:
      wandb.init(project='paper_verion_new_eval_korean_slt_{}'.format(d_model))
      wandb.config.update(config)
      wandb.run.name = project_name#'{}_{}_{}_{}'.format(str(split_dim),str(learning_rate),'our' if max_length != 50 else 'other')
      wandb.run.save()
  checkpoint_path = project_name#'{}_{}_{}'.format(str(split_dim),str(learning_rate),'our' if max_length != 50 else 'other')
  model_name = project_name#'{}_{}_{}'.format(str(split_dim),str(learning_rate),'our' if max_length != 50 else 'other')
  createFolder(checkpoint_path)
  dataset = TranslationDataset(tokenizer=tokenizer, file_path=data_path, max_length=max_length, embedding_length=input_d,other_exp=other_exp, args = args)

  model = Transformer(vocab_num=vocab_num,
                      input_d = input_d,
                      d_model=d_model,
                      max_seq_len=max_length,
                      head_num=head_num,
                      dropout=dropout,
                      split_dim = split_dim,
                      N=N)

  model = DistributedDataParallel(module=model.cuda(), device_ids=[local_rank])
  
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.99)

  trainer = TranslationTrainer(dataset, tokenizer, model, max_length, device, model_name, checkpoint_path, batch_size)
  train_dataloader, eval_dataloader = trainer.build_dataloaders(train_test_split=0.05)
  
  trainer.train(epochs, train_dataloader, eval_dataloader, optimizer, scheduler)

