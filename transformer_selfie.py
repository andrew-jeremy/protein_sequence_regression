import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils_selfie import MAX_PROT_LEN, MAX_SMILES_LEN


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))



class TransformerConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
#        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # https://github.com/tunz/transformer-pytorch/blob/master/model/transformer.py Andrew Kiruluta
        # https://github.com/huggingface/transformers/issues/356

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.LeakyReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0)])

#multiple input neural network for fusing the two attention heads
'''
Protein+Smiles1 --> FCLayer 
                           \
                         FCLayer    ---> FCLayer ---> Output
                           /
Protein+Smiles2 --> FCLayer                   
'''
class FuseNet(nn.Module):
  def __init__(self,dim_model):
    super(FuseNet, self).__init__() 
    self.fc1 = nn.Linear(131072,dim_model)  #313344
    self.fc2 = nn.Linear(131072,dim_model)
    self.fc3 = nn.Linear(2*dim_model, dim_model)

  def forward(self, input1, input2):
    f1 = self.fc1(input1)
    f2 = self.fc2(input2)
    # now we can reshape `f1` to `f3` to 2D and concat them
    combined = torch.cat((f1.view(f1.size(0), -1),
                          f2.view(f2.size(0), -1)), dim=1)
    out = self.fc3(combined)
    #out  = nn.Sequential(self.fc3(combined))

    return out

class TransformerEncoder(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        #self.tok_emb = nn.Embedding(config.vocab_size, 1) #config.n_embd)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        #self.pos_emb = nn.Parameter(torch.zeros(1, 740, 1))
        #self.positional_encoder = PositionalEncoding(dim_model=740, dropout_p=0.1, max_len=5000)
        self.positional_encoder = PositionalEncoding(dim_model=config.n_embd, dropout_p=0.1, max_len=5000)
        self.FuseNet = FuseNet(config.n_embd)

        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, 1, bias=False)
        self.protein_reduce = nn.Linear(MAX_PROT_LEN, MAX_SMILES_LEN, bias=False)
        self.att = nn.Linear((2*MAX_PROT_LEN + 2*MAX_SMILES_LEN), (MAX_PROT_LEN + MAX_SMILES_LEN), bias=False)

        #self.head = nn.Linear((MAX_PROT_LEN + MAX_SMILES_LEN)*config.n_embd, 1, bias=False)

        #'''
        self.Regression_Output = nn.Sequential(
            nn.Linear(649216, config.n_embd),  # Option #3
            #nn.Linear(671744, config.n_embd),  # SELFIES
            #nn.Linear(262144, config.n_embd),   # Option #2 with Reduced Protein length
            #nn.Linear((2*MAX_PROT_LEN + 2*MAX_SMILES_LEN)*config.n_embd, config.n_embd), # Option #2
            nn.Linear(config.n_embd, 128),    # Option for all: 1, 2, & 3
            nn.LeakyReLU(),
            nn.Linear(128,50),
            nn.LeakyReLU(),
            nn.Linear(50,1)
        )

    
        self.drop2 = nn.Dropout(0.5)
        self.pool = nn.MaxPool1d(3, stride=0)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        self.logcosh_loss = LogCoshLoss()


    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate,
                                      betas=train_config.betas)
        return optimizer

    def forward_base(self, idx):
        b,  t = idx.size()
        # assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the Encoder model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector Andrew Kiruluta
        
        protein_embeddings = token_embeddings[:, :(t-2*MAX_SMILES_LEN), :]
        smile1_embeddings = token_embeddings[:, (t-2*MAX_SMILES_LEN):(t-MAX_SMILES_LEN), :]
        smile2_embeddings = token_embeddings[:, (t-MAX_SMILES_LEN):, :]

        # reduce size of protein to that of MAX_SMILES_LEN - NOT MAKING A DIFFERENCE
        #protein_embeddings = self.protein_reduce(protein_embeddings.permute(0, 2, 1))
        #protein_embeddings = protein_embeddings.permute(0, 2, 1)  # restore order

        # protein-smile1 attention
        x1 = torch.cat([protein_embeddings,smile1_embeddings],dim=1)
        x1 = self.positional_encoder(x1)
        x1 = self.blocks(x1) 
        x1 = self.ln_f(x1) 

        # protein-smile2 attention
        x2 = torch.cat([protein_embeddings,smile2_embeddings],dim=1)
        x2 = self.positional_encoder(x2)
        x2 = self.blocks(x2) 
        x2 = self.ln_f(x2)  

        #-------> OPTION #1: fuse attention embeddings with feedforward network
        #x1 = torch.flatten(x1, start_dim=1)
        #x2 = torch.flatten(x2, start_dim=1)
        #x_max = self.FuseNet(x1,x2)
        #-------->

        #------> OPTION #2: simply concacetante the two attention arms
        x = torch.cat([x1,x2],dim=1)
        x_max = torch.flatten(x, start_dim=1)
        #------->

        #--------> OPTION #3: cross-attention head of the two attention arms
        #x = torch.cat([x1,x2],dim=1)
        #x = x.permute(0, 2, 1)
        #x = self.att(x)
        #x = x.permute(0, 2, 1)
        #x = self.positional_encoder(x)
        #x = self.blocks(x)
        #x_max = torch.flatten(x, start_dim=1)
        #---------->
        return x_max


    def forward(self, idx, y):
        x_max = self.forward_base(idx)

        #outputs = self.head(x_max)   # for Option #1
        outputs = self.Regression_Output(x_max)

        #loss = F.mse_loss(outputs, y)
        loss = self.logcosh_loss(outputs, y)

        return outputs, loss


