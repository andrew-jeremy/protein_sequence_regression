import re
import numpy as np
import torch
import random
#from transformers import BertTokenizer

MAX_PROT_LEN = 484 #485 #810
MAX_SMILES_LEN = 150 #128 #100

MAX_AA = 5

#tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False )

# AA_vocab and pos_vocab were used to train activity oracle
AA_map = {'<PAD>': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4,
            'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
            'M': 11,'N': 12, 'P': 13, 'Q': 14, 'R': 15,
            'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}


smiles_alphabet = {'#', '(', ')', '-', '.', '=',
                   '1', '2', '3', '4', '5', '6', '7', '8',
                   'Br', 'C', 'F', 'I', 'N', 'O', 'P', 'S',
                   '[B-]', '[Br-]', '[H]', '[K]', '[Li]', '[N+]',
                   '[N-]', '[NH+]', '[NH2+]', '[NH3+]', '[Na+]',
                   '[O-]', '[OH-]', '[P-]', '[Pt+2]', '[Pt]', '[S+]',
                   '[SH]', '[Si]', '[n+]', '[nH+]', '[nH]',
                   'c', 'n', 'o', 's'}

# atom-level tokens used for trained the spe vocabulary
# atom_toks
atom_toks = {'[c-]', '[SeH]', '[N]', '[C@@]', '[Te]', '[OH+]', 'n', '[AsH]', '[B]', 'b', 
             '[S@@]', 'o', ')', '[NH+]', '[SH]', 'O', 'I', '[C@]', '-', '[As+]', '[Cl+2]', 
             '[P+]', '[o+]', '[C]', '[C@H]', '[CH2]', '\\', 'P', '[O-]', '[NH-]', '[S@@+]', 
             '[te]', '[s+]', 's', '[B-]', 'B', 'F', '=', '[te+]', '[H]', '[C@@H]', '[Na]', 
             '[Si]', '[CH2-]', '[S@+]', 'C', '[se+]', '[cH-]', '6', 'N', '[IH2]', '[As]', 
             '[Si@]', '[BH3-]', '[Se]', 'Br', '[C+]', '[I+3]', '[b-]', '[P@+]', '[SH2]', '[I+2]', 
             '%11', '[Ag-3]', '[O]', '9', 'c', '[N-]', '[BH-]', '4', '[N@+]', '[SiH]', '[Cl+3]', '#', 
             '(', '[O+]', '[S-]', '[Br+2]', '[nH]', '[N+]', '[n-]', '3', '[Se+]', '[P@@]', '[Zn]', '2', 
             '[NH2+]', '%10', '[SiH2]', '[nH+]', '[Si@@]', '[P@@+]', '/', '1', '[c+]', '[S@]', '[S+]', 
             '[SH+]', '[B@@-]', '8', '[B@-]', '[C-]', '7', '[P@]', '[se]', 'S', '[n+]', '[PH]', '[I+]', '5', 'p', '[BH2-]', '[N@@+]', '[CH]', 'Cl'}


smiles_regex = re.compile(
    r'(\[[^\]]+]|C|Cr|Br|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|'
    r'-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
)

SMILES_map = {'<PAD>': 0}
for idx, char in enumerate(smiles_alphabet, start=len(AA_map)-1):
    SMILES_map[char] = idx + 1
SMILES_map['<Unk>'] = len(AA_map) + len(SMILES_map)

#VOCAB_SIZE = len(AA_map) + len(SMILES_map)
VOCAB_SIZE = 3500  # NEED LARGE VOCAB SIZE FOR 'dose_response_data_processed_smiles.pkl' =>ANDREW KIRULUTA

def get_protein_encoding(seq):
    encoding = [0]*MAX_PROT_LEN
    try:
        if len(seq) > MAX_PROT_LEN:
            seq = seq[:MAX_PROT_LEN]
        for i, aa in enumerate(seq):
            try:
                encoding[i] = AA_map[aa] # maps protein letters to numerics using a dictionary - Andrew Kiruluta
            except Exception as e:
                encoding[i] = 0
    except Exception as e:
        pass
    return encoding

def get_smiles_encoding(smiles):
    encoding = [0]*MAX_SMILES_LEN
    if len(smiles) > MAX_SMILES_LEN:
        smiles = smiles[:MAX_SMILES_LEN]
    print('smiles: %s' %smiles)
    for i, char in enumerate(smiles_regex.split(smiles)[1::2]):
        encoding[i] = SMILES_map.get(char, SMILES_map['<Unk>']) # This maps SMILES to a list of numbers char by char Andrew Kiruluta

    return encoding

def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [0]*(target_len - len(some_list))

# protein tokenization using ProtBert-BFD pretrained-model
def get_pro_bert_protein_encoding(prot):
    #tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False )
    try:
        prot = list(prot)
        prot = [re.sub(r"[UZOB]", "X", sequence) for sequence in prot]
        ids = tokenizer.batch_encode_plus(prot, add_special_tokens=True, padding=True)
        ids = ids[1:-1] # purge start & end tokens
        return pad_or_truncate(ids['input_ids'][0], MAX_PROT_LEN)
    except Exception as e:
        return [0]*MAX_PROT_LEN

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


