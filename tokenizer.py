from datasets import load_dataset
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizer
#download pretrained------------------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#encode-------------------------------------------------------------------------------
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
print(tokens) #display

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)#display

#decode-------------------------------------------------------------------------------
decoded_string = tokenizer.decode([7993, 170, 13809, 23763, 2443, 1110, 3014])
print(decoded_string)

# #acquire dataset--------------------------------------------------------------------
# dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
# #display
# print(dataset)
# print(dataset[1])
# print(dataset[4])


# #segment into batch-----------------------------------------------------------------
# def batch_iterator():
#     for i in range(0, len(dataset), batch_size):
#         yield dataset[i : i + batch_size]["text"]

# #train------------------------------------------------------------------------------
# tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]")) 
# tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True) #all lowercase
# tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer() #separate by spaces
# #display
# print(tokenizer.pre_tokenizer.pre_tokenize_str("This is an example!"))

# special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
# trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)#create trainer
# tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)#train

# #post-processing------------------------------------------------------------------
# cls_token_id = tokenizer.token_to_id("[CLS]")
# sep_token_id = tokenizer.token_to_id("[SEP]")
# print(cls_token_id, sep_token_id)

# tokenizer.post_processor = processors.TemplateProcessing(
#     single=f"[CLS]:0 $A:0 [SEP]:0",
#     pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
#     special_tokens=[
#         ("[CLS]", cls_token_id),
#         ("[SEP]", sep_token_id),
#     ],
# )

# #check encoding-------------------------------------------------------------------
# encoding = tokenizer.encode("This is one sentence.", "With this one we have a pair.")
# print(encoding.tokens)
# print(encoding.type_ids)

# #save pre-trained-------------------------------------------------------------------
# tokenizer.save_pretrained("my-new-tokenizer")
