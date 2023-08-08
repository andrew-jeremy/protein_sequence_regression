import random
import numpy
import numpy as np
import pandas as pd
import os, ast

from torch.utils.data import Dataset
from utils import get_protein_encoding, get_smiles_encoding, get_pro_bert_protein_encoding
import torch

GENE_NAME_KEY = 'gene_name'
SCREEN_ID = 'screen_protocol_short_name'
TARGET_SEQUENCE_KEY = 'target_sequence'
CANONICAL_SMILES_KEY = 'canonical_smiles'
ACTIVITY_KEY = 'pIC50'
RESULT_OPERATOR_KEY = 'result_operator'
ENCODING_KEY = 'encoding'

class PharmacologyDataset(Dataset):

    def __init__(self, df, df_tgt, size=1000):

        self.df = df
        self.df_size = len(self.df)
        self.df_tgt = df_tgt

        self.df_grouped = self.df.groupby([GENE_NAME_KEY, SCREEN_ID])

        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        for loop_idx in range(10):
            try:
                i = np.random.randint(self.df_size)
                gene_name = self.df.loc[i, GENE_NAME_KEY]
                screen_name = self.df.loc[i, SCREEN_ID]

                df_cur_group = self.df_grouped.get_group((gene_name, screen_name))
                df_pair = df_cur_group.sample(n=2, replace=True)

                prot_enc = self.df_tgt.loc[self.df_tgt[GENE_NAME_KEY] == gene_name, 'encoding'].values[0]
                smiles1_enc, smiles2_enc = df_pair[ENCODING_KEY].values  

                # smiles1, smiles2 = df_pair[CANONICAL_SMILES_KEY].values
                activity1, activity2 = df_pair[ACTIVITY_KEY].values


                result_operator1, result_operator2 = df_pair[RESULT_OPERATOR_KEY].values
                if (result_operator1 == '>') and (result_operator2 == '>'):
                    if loop_idx < 9:
                        continue
                if (result_operator1 == '<') and (result_operator2 == '<'):
                    if loop_idx < 9:
                        continue

                enc = prot_enc + smiles1_enc + smiles2_enc
                #enc = smiles1_enc + prot_enc + smiles2_enc + prot_enc
                #enc = prot_enc + smiles1_enc + prot_enc + smiles2_enc 
                #enc = prot_enc + smiles1_enc

                # move all padding zeros to end of list
                #result = [n for n in enc if n != 0]
                #result.extend([0] * enc.count(0))
                #enc = result

                x = torch.tensor(enc, dtype=torch.long)
                y1 = torch.tensor([activity1]).unsqueeze(1)
                y2 = torch.tensor([activity2]).unsqueeze(1)
                return x, y1[0], y2[0]  
            except Exception as e:
                print('Error retrieve data ', e)


class PharmacologyTestDataset(Dataset):

    def __init__(self, df_test, df_train, df_tgt, df_file, size=1000):
        self.df_test = df_test
        self.df_train = df_train

        self.df_tgt = df_tgt

        self.df_test_grouped = self.df_test.groupby([SCREEN_ID])
        self.df_train_grouped = self.df_train.groupby([SCREEN_ID])

        test_df_list = []
        train_df_list = []
        for group in self.df_test_grouped:
            try:
                df_test_cur_group = group[1]
                df_train_cur_group = self.df_train_grouped.get_group(group[0])
                if len(df_train_cur_group) > 0:
                    df_train_sampled_subset \
                        = df_train_cur_group.sample(len(df_test_cur_group),
                                                    replace=True, random_state=11)
                    test_df_list.append(df_test_cur_group)
                    train_df_list.append(df_train_sampled_subset)
            except Exception as e:
                pass

        self.df_test2 = pd.concat(test_df_list)
        self.df_test2.reset_index(inplace=True, drop=True)
        self.df_train2 = pd.concat(train_df_list)
        self.df_train2.reset_index(inplace=True, drop=True)

        column_map = {c:c+"_train" for c in self.df_train2.columns}
        df_train2a = self.df_train2.rename(columns=column_map)
        column_map = {c:c+"_test" for c in self.df_test2.columns}
        df_test2a = self.df_test2.rename(columns=column_map)

        df_combined = pd.concat([df_train2a, df_test2a], axis=1)
        df_combined.drop(columns=[ENCODING_KEY + '_train', ENCODING_KEY + '_test'], inplace=True)
        df_combined.to_csv(df_file)

        if size < len(self.df_test2):
            self.size = size
        else:
            self.size = len(self.df_test2)

        print('test set size ', self.size)

        gene_name = self.df_test2.loc[0, GENE_NAME_KEY]
        seq = self.df_tgt.loc[self.df_tgt[GENE_NAME_KEY] == gene_name,
                              TARGET_SEQUENCE_KEY].values[0]
        #self.prot_enc = get_protein_encoding(seq)
        self.prot_enc = get_pro_bert_protein_encoding(seq)
      

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        smiles1_enc = self.df_train2.loc[idx, ENCODING_KEY]
        smiles2_enc = self.df_test2.loc[idx, ENCODING_KEY]

        activity1 = self.df_train2.loc[idx, ACTIVITY_KEY]  
        activity2 = self.df_test2.loc[idx, ACTIVITY_KEY]

        enc = self.prot_enc + smiles1_enc + smiles2_enc 
        #enc = smiles1_enc + self.prot_enc + smiles2_enc + self.prot_enc 
        #enc = self.prot_enc + smiles1_enc

        # move all padding zeros to end of list
        #result = [n for n in enc if n != 0]
        #result.extend([0] * enc.count(0))
        #enc = result

        x = torch.tensor(enc, dtype=torch.long)
        y1 = torch.tensor([activity1]).unsqueeze(1)
        y2 = torch.tensor([activity2]).unsqueeze(1)
        return x, y1[0], y2[0]


class ProteinDataset(Dataset):
 
  def __init__(self,df,transform=None):
    self.df = df
    self.transform = transform
  def __len__(self):
    return len(self.df)
   
  def __getitem__(self,idx):
      self.x = self.df.loc[idx,"embedding"]
      self.y = self.df.loc[idx,"Kd"]
      self.x = torch.tensor(ast.literal_eval(self.x),dtype=torch.long)
      self.x = (self.x).flatten()
      self.y = torch.tensor(self.y,dtype=torch.float32)
      return self.x,self.y
