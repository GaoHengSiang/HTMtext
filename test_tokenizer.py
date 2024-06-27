from datasets import load_dataset
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizer

custom_tokenizer = Tokenizer.from_file("my-new-tokenizer.json")

sequence = "Using a Transformer network is simple" #dummy

tokens = custom_tokenizer.tokenize(sequence) #--> sentence
print(tokens) #display

id_seq = tokenizer.convert_tokens_to_ids(tokens)
print(id_seq)#display