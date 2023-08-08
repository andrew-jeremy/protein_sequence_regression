import pandas as pd
import torch
from dataset_selfie import ProteinDataset, \
    ACTIVITY_KEY, TARGET_SEQUENCE_KEY, ENCODING_KEY, RESULT_OPERATOR_KEY

import shutil
import os, sys
import glob
import argparse

import numpy as np
from transformer_selfie import TransformerConfig, TransformerEncoder
from trainer_selfie_mods import Trainer, TrainerConfig
from utils_selfie import seed_everything, MAX_PROT_LEN, MAX_SMILES_LEN, VOCAB_SIZE
from utils_selfie import get_protein_encoding, get_pro_bert_protein_encoding


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
    df_train = df3.sample(frac=0.8,random_state=200)
    df_test = df3.drop(df_train.index)
    df_test = df_test.sample(frac=0.8,random_state=200)
    df_val = df_test.drop(df_test.index)

    train_dataset = ProteinDataset(df_train)
    test_dataset = ProteinDataset(df_test)
    val_dataset = ProteinDataset(df_val)

    BLOCK_SIZE = MAX_PROT_LEN + MAX_SMILES_LEN
    mconf = TransformerConfig(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE,
                              embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,
                              n_layer=nlayers, n_head=16, n_embd=512)

    # save model pat
    ckpt_path = output_dir + '/' +  '.pt'

    start_epoch = 0
    model = TransformerEncoder(mconf)  # ANDREW KIRULUTA

    ''' Andrew Kiruluta... for loading saved checkpoint from a previous run
    if os.path.exists(ckpt_path):
        start_epoch = get_final_epoch(output_dir, test_target)
        if start_epoch > 0:
            start_epoch += 1
        print('start epoch: %d' % start_epoch)

        print('loading saved model...')
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()
    '''

    print(sum(p.numel() for p in model.parameters() if p.requires_grad), 'model parameters')

    tconf = TrainerConfig(start_epoch=start_epoch, max_epochs=60000,
                          batch_size=batch_size, learning_rate=lr,
                          lr_decay=True, warmup_tokens=512*VOCAB_SIZE,
                          final_tokens=2*len(train_dataset)*BLOCK_SIZE,
                          num_workers=0, ckpt_path=ckpt_path)

    trainer = Trainer(model, train_dataset, val_dataset, test_dataset, tconf, output_dir)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Transformer model')
    parser.add_argument('--target_idx', type=int, default=0,
                        help='left-out target')
    parser.add_argument('--output_dir', type=str, default='output', help='Outfile directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--nlayers', type=int, default=2, help='Number of layers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    

    parser.add_argument('-v', '--verbose', action='count', default=0)

    args = parser.parse_args()

    target_idx = args.target_idx
    output_dir = args.output_dir
    lr = args.lr
    batch_size = args.batch_size
    nlayers = args.nlayers
    seed = args.seed

    print('seed: %d' % seed)

    include_target_in_training = args.include_target_in_training

    train(output_dir, batch_size, lr, nlayers, seed)
