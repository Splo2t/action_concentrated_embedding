import torch
from torch.utils.data import Dataset, DataLoader, random_split
#from util import load_csv
from model.util import subsequent_mask
from transformers import BertTokenizer
from torch.autograd import Variable
from tqdm import tqdm
import pickle
import numpy as np
import random

class TranslationDataset(Dataset):
  def __init__(self,  tokenizer:BertTokenizer, file_path:str, max_length:int, embedding_length=608, is_save_pickle=False, other_exp = False, args=None):
      self.docs = load_pickle_batch(file_path)
      self.resol_n = int(file_path.split('_')[-2])
      self.pad_token = [0.0 for i in range(embedding_length)]
      print(np.array(self.pad_token).shape)
      self.pad_token_idx = tokenizer.pad_token_id
      self.max_length = max_length
      self.embedding = embedding_length
      self.tokenizer = tokenizer
      self.other_exp = other_exp
      self.args = args
      print(tokenizer(["안녕하세요. 반갑습니다", '오늘의 날씨는 맑음입니다. 내일의 날씨는 흐림입니다.'],padding="max_length"))
    
  @staticmethod
  def make_std_mask(tgt, pad_token_idx):
    'Create a mask to hide padding and future words.'
    target_mask = (tgt != pad_token_idx).unsqueeze(-2)
    target_mask = target_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(target_mask.data))
    return target_mask.squeeze()

  def __len__(self):
    return len(self.docs)
  
  def __getitem__(self, idx):
    file_path = self.docs[idx]
    with open(file_path, 'rb') as csv_file:
      pickle_reader = pickle.load(csv_file)
    line = {'feature':pickle_reader['feature'], 'label':pickle_reader['label']}
    
    if type(line['label']) != type("asdf"):
      self.docs[idx] = self.docs[0]
      with open(self.docs[0], 'rb') as csv_file:
        pickle_reader = pickle.load(csv_file)  
      line = {'feature':pickle_reader['feature'], 'label':pickle_reader['label']}
        
    cut_idx = int(10/(30/self.resol_n))
    
    if self.embedding != 108+9999: #2023년 논문
      input_tuple = []
      if 'hands' in self.args.body_part : 
        input_tuple.append(torch.tensor(line['feature']['left'], dtype=torch.float))
        input_tuple.append(torch.tensor(line['feature']['right'], dtype=torch.float))
      if 'face' in self.args.body_part :
        input_tuple.append(torch.tensor(line['feature']['face'], dtype=torch.float))
      if 'body' in self.args.body_part :
        input_tuple.append(torch.tensor(line['feature']['body'], dtype=torch.float))
      input = torch.hstack(tuple(input_tuple))
    else: #2023년 김영민 논문
      input = torch.tensor(line['feature'], dtype=torch.float) 
    #input = torch.hstack((input_left, input_right, input_face))
    
    if input.shape[1] != self.embedding:
        print(self.embedding, ",", input.shape[1])
        print(file_path, "#ERR")
        with open(self.docs[0], 'rb') as csv_file:
            pickle_reader = pickle.load(csv_file)
        line = {'feature':pickle_reader['feature'], 'label':pickle_reader['label']}
        if self.embedding != 108+9999: 
          input_tuple = []
          if 'hands' in self.args.body_part : 
            input_tuple.append(torch.tensor(line['feature']['left'], dtype=torch.float))
            input_tuple.append(torch.tensor(line['feature']['right'], dtype=torch.float))
          if 'face' in self.args.body_part :
            input_tuple.append(torch.tensor(line['feature']['face'], dtype=torch.float))
          if 'body' in self.args.body_part :
            input_tuple.append(torch.tensor(line['feature']['body'], dtype=torch.float))
          input = torch.hstack(tuple(input_tuple))
        else: #2023년 김영민 논문
          input = torch.tensor(line['feature'], dtype=torch.float)
    
        #input = torch.hstack((input_left, input_right, input_face))
    if self.other_exp == True:
      indices = torch.randperm(input.shape[0])[:self.max_length]
      indices, _ = indices.sort()
      input = input[indices]

    #print(input.shape)
    #pad_tokens = torch.tensor([self.pad_token] * (self.max_length-(input.shape[0])))
    #print(pad_tokens.shape)
    #input = torch.vstack((input,pad_tokens))
    
    if self.max_length -input.shape[0] <= 0:
        input = input[:self.max_length, :]
        input_mask = [True for i in range(input.shape[0])] + [False for i in range(self.max_length-input.shape[0])]
        input_position = [i for i in range(input.shape[0])] + [0 for i in range(self.max_length-input.shape[0])]
    
    else:
        input_mask = [True for i in range(input.shape[0])] + [False for i in range(self.max_length-input.shape[0])]
        pad_tokens = torch.tensor([self.pad_token] * (self.max_length-(input.shape[0])))
        input_position = [i for i in range(input.shape[0])] + [0 for i in range(self.max_length-input.shape[0])]
        input = torch.vstack((input,pad_tokens))
    
    #input_mask = [True for i in range(input.shape[0])] + [False for i in range(self.max_length-input.shape[0])]
    token_type_ids = [0 for i in range(input.shape[0])]
    target = self.tokenizer.encode(line['label'], max_length=self.max_length, truncation=True)
    rest = self.max_length - len(target)
    #if rest > 0:
    target = torch.tensor(target+ [self.pad_token_idx] * rest)

    doc={
      'input':input,# input
      'token_type':torch.tensor(token_type_ids),
      'input_mask': torch.tensor([input_mask]),       # input_mask
      'target_str': self.tokenizer.convert_ids_to_tokens(target),
      'target': target,                                       # target,
      'target_mask': self.make_std_mask(target, self.pad_token_idx),    # target_mask
      'token_num': (target[...,1:] != self.pad_token_idx).data.sum()  # token_num
    }

    return doc

def load_pickle(file_path = '/home/disuper422/yuhw/SLT_korean/data/data.pickle'):
  print(f'Load Data | file path: {file_path}')
  with open(file_path, 'rb') as csv_file:
    pickle_reader = pickle.load(csv_file)
    print(pickle_reader)
    print("LEN:",len(pickle_reader))
    #print(pickle_reader)
    lines = []
    
    for line in pickle_reader:
      new_line = {'feature':line['feature'], 'label':line['label']}
      lines.append(new_line)
      
  print(f'Load Complete | file path: {file_path}')
  #####
  #lines[0]['feature'] = lines[0]['feature'].T
  #####
  #print(lines[0]['feature'].shape)
  #pad_token = [[0.0 for i in range(288)]]
  input = lines[0]['feature']
  print(lines[0]['feature']['face'].shape)
  print(lines[0]['feature']['left'].shape)
  print(lines[0]['feature']['right'].shape)
  print(lines[0]['feature']['body'].shape)
  #pad_tokens = np.array(pad_token * (500-input.shape[0]))
  #print(input.shape)
  #print(pad_tokens.shape)
  
  #lines[0]['feature'] = np.vstack((input, pad_tokens))
  print(lines[0]['feature'].keys())
  print(lines[0]['label'])
  print(lines[0].keys())
  return lines

def load_pickle_batch(file_path = '/home/disuper422/yuhw/slt/preprocessing/our_processed_data/'):
  import glob
  lines = []
  print(f'Load Data | file path: {file_path}')
  file_list = glob.glob(file_path+'*.pickle')
  print(len(file_list))
  return file_list

  print("AVG_OF_TOKENS")
  tokens = 0
  samples = 0
  for file_path in file_list:
    with open(file_path, 'rb') as csv_file:
        pickle_reader = pickle.load(csv_file)
        #print(pickle_reader)
        for line in [pickle_reader]:
            new_line = {'file_path':file_path, 'label':line['label']}
            with open(file_path, 'rb') as csv_file:
              pickle_reader = pickle.load(csv_file)
              line = {'feature':pickle_reader['feature'], 'label':pickle_reader['label']}
         
              input_left = torch.tensor(line['feature']['left'], dtype=torch.float)
              tokens += input_left.shape[0]
              samples += 1
  print(tokens/samples)

  #####
  #pad_token = [[0.0 for i in range(288)]]
  #input = lines[0]['feature']
  
  
  #pad_tokens = np.array(pad_token * (500-input.shape[0]))
  #print(input.shape)
  #print(pad_tokens.shape)
  
  #lines[0]['feature'] = np.vstack((input, pad_tokens))
 
  return lines













