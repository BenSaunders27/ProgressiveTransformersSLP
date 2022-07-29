import pandas as pd
import torch
from torchtext.vocab import build_vocab_from_iterator
from transformers import BertTokenizer, GPT2Tokenizer
from torch.nn.utils.rnn import pad_sequence
from utils import load_data


def white_space_tokenizer(string):
    return string.strip().split()


def yield_tokens(source, tokenizer):
    for src in source:
        yield tokenizer(src)


def build_vocab_from_phoenix(fpath, tokenizer, mode = 'text'):
    assert mode in ["gloss", "text"], 'selected between text and gloss.'
    annotation = load_data(fpath)    
    source = [data[mode] for data in annotation]
    
    return build_vocab_from_iterator(
        yield_tokens(source, tokenizer),
        specials = ['<pad>', '<bos>', '<eos>', '<unk>'],
        special_first = True
    )

    
class SimpleTokenizer:
    def __init__(self, tokenizer, vocab):
        vocab_dict = vocab.get_stoi()
        vocab.set_default_index(vocab['<unk>'])
        
        self.vocab_size = len(vocab)
        self.tokenizer = tokenizer
        self.vocab = vocab

        self.pad_token = vocab_dict['<pad>']
        self.start_token = vocab_dict['<bos>']
        self.end_token = vocab_dict['<eos>']

    def set_max_length(self, max_len):
        self.max_length = max_len    
    
    def _encode(self, text, add_special_token = True):
        tokens = self.tokenizer(text)
        if add_special_token:
            tokens.append('<eos>')
            tokens.insert(0, '<bos>')
        ids = [self.vocab[tok] for tok in tokens]
        return ids

    def encode(
        self, 
        text_list, 
        padding = 0, 
        add_special_tokens = True,
        device = 'cpu'
    ):
        id_list, id_len = [], []
        for text in text_list:
            ids = self._encode(text, add_special_tokens)
            id_list.append(torch.tensor(ids))
            id_len.append(torch.ones(len(ids), dtype = int))
        
        if padding == 'max_length':
            assert add_special_tokens == False
            input_ids = pad_sequence(id_list, padding_value = self.pad_token, batch_first = True)
            b, seq_len = input_ids.size()
            input_ids = torch.cat((input_ids, torch.zeros(b, self.max_length - seq_len)), dim = -1)
            attention_mask = pad_sequence(id_len, padding_value = self.pad_token, batch_first = True)
        else:
            input_ids = pad_sequence(id_list, padding_value = self.pad_token, batch_first = True)
            attention_mask = pad_sequence(id_len, padding_value = self.pad_token, batch_first = True)
        
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        return input_ids, attention_mask
    
    def _decode(self, tokens, special_tok = [0, 1, 2]):
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
        tokens = [tok for tok in tokens if tok not in special_tok]
        return self.vocab.lookup_tokens(tokens)
    
    def decode(self, tokens, special_tokens):
        if torch.is_tensor(tokens):
            token_list = tokens.tolist()
        text_list = []
        for tokens in token_list:
            text = ' '.join(self._decode(tokens, special_tokens))
            text_list.append(text)
        return text_list


class HugTokenizer:
    def __init__(self, model = 'bert-base-uncased'):
        tokenizer = BertTokenizer.from_pretrained(model)
        
        self.vocab_size = tokenizer.vocab_size
        
        self.pad_token = tokenizer.pad_token_id
        self.start_token = tokenizer.bos_token_id
        self.end_token = tokenizer.eos_token_id
        
        self.tokenizer = tokenizer

    def set_max_length(self, max_len):
        self.tokenizer.model_max_length = max_len

    def encode(
        self, 
        texts, 
        padding, 
        add_special_tokens = False, 
        device = 'cpu'
    ):
        encoded = self.tokenizer.batch_encode_plus(
            texts, 
            return_tensors = 'pt', 
            padding = padding, 
            truncation = True,
            add_special_tokens = add_special_tokens
        )

        input_ids = encoded.input_ids.to(device)
        pad_mask = encoded.attention_mask.to(device)

        return input_ids, pad_mask
      
    def decode(self, tokens):
        decoded = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        return decoded

    