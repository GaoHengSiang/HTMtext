from datasets import load_dataset
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizer

custom_tokenizer = Tokenizer.from_file("my-new-tokenizer.json")

sequence = "Using a Transformer network is simple" #dummy

output = custom_tokenizer.encode(sequence) #--> sentence
print(output.tokens) #display

#id_seq = custom_tokenizer.convert_tokens_to_ids(tokens)
print(output.ids)#display