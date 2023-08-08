import pandas as pd
import torch
from dataset import ProteinDataset, \
    ACTIVITY_KEY, TARGET_SEQUENCE_KEY, ENCODING_KEY, RESULT_OPERATOR_KEY

import shutil
import os, sys
import glob
import argparse

import numpy as np
from transformer import TransformerConfig, TransformerEncoder
from trainer import Trainer, TrainerConfig
from utils import seed_everything, MAX_PROT_LEN, MAX_SMILES_LEN, VOCAB_SIZE
from utils import get_protein_encoding, get_pro_bert_protein_encoding


def get_final_epoch(dirname, target):
    avail_files = glob.glob(dirname + '/' + target + '*pred_output')
    if len(avail_files) == 0:
        return 0

    try:
        epochs = [int((f.split(target + '_0.')[1]).split('-')[0]) for f in avail_files]
        epochs.sort()
        return epochs[-1]
    except Exception as e:
        print(e)
        print('unable to retrieve epoch count', avail_files)
        sys.exit()

def train( output_dir, batch_size, lr, nlayers, seed):
    seed_everything(seed)

    # read in data from EDA and create train, test, and validation sets
    df3 = pd.read_csv('data/20230331_pembro_SSMplus_AlphaSeq_data_unique_embedding.csv')
    #df3 = df3.head(10000)  # debug test poinnt

    # scale target Kd column
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df3[['Kd']] = scaler.fit_transform(df3[['Kd']])

    df_train = df3.sample(frac=0.8,random_state=200)
    df_test = df3.drop(df_train.index)
    df_test = df_test.sample(frac=0.8,random_state=200)
    df_val = df_test.drop(df_test.index)

    # reset index for each dataframe
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)

    # create datasets
    train_dataset = ProteinDataset(df_train)
    test_dataset = ProteinDataset(df_test)
    val_dataset = ProteinDataset(df_val)

    BLOCK_SIZE = 252 #MAX_PROT_LEN + MAX_SMILES_LEN
    mconf = TransformerConfig(vocab_size=BLOCK_SIZE, block_size=BLOCK_SIZE,
                              embd_pdrop=0.2, resid_pdrop=0.2, attn_pdrop=0.2,
                              n_layer=nlayers, n_head=16, n_embd=512)

    # save model path
    ckpt_path = 'data/model_chk_512.pt'

    start_epoch = 0
    model = TransformerEncoder(mconf)  

    '''
    # Andrew Kiruluta... for loading saved checkpoint from a previous run
    if os.path.exists(ckpt_path):
        print('loading saved model...')
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()
        start_epoch = 0
        model = TransformerEncoder(mconf)
    '''

    print(sum(p.numel() for p in model.parameters() if p.requires_grad), 'model parameters')

    tconf = TrainerConfig(start_epoch=start_epoch, max_epochs=60000,
                          batch_size=batch_size, learning_rate=lr,
                          lr_decay=False, warmup_tokens=50*VOCAB_SIZE,
                          final_tokens=2*len(train_dataset)*BLOCK_SIZE,
                          num_workers=0, ckpt_path=ckpt_path)

    trainer = Trainer(model, train_dataset, val_dataset, test_dataset, tconf, output_dir)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Transformer model')
    parser.add_argument('--output_dir', type=str, default='data', help='Outfile directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--nlayers', type=int, default=1, help='Number of layers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    

    parser.add_argument('-v', '--verbose', action='count', default=0)

    args = parser.parse_args()

    output_dir = args.output_dir
    lr = args.lr
    batch_size = args.batch_size
    nlayers = args.nlayers
    seed = args.seed

    print('seed: %d' % seed)
    train(output_dir, batch_size, lr, nlayers, seed)
