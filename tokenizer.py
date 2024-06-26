from datasets import load_dataset
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizer
# #download pretrained------------------------------------------------------------------
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# #encode-------------------------------------------------------------------------------
# sequence = "Using a Transformer network is simple"
# tokens = tokenizer.tokenize(sequence)
# print(tokens) #display

# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(ids)#display

# #decode-------------------------------------------------------------------------------
# decoded_string = tokenizer.decode([7993, 170, 13809, 23763, 2443, 1110, 3014])
# print(decoded_string)

#settings---------------------------------------------------------------------------
batch_size = 1000
vocab_size = 10000

#acquire dataset--------------------------------------------------------------------
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
#display
print(dataset)
print(dataset[1])
print(dataset[4])


#segment into batch-----------------------------------------------------------------

def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]

#train------------------------------------------------------------------------------
custom_tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]")) 
custom_tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True) #all lowercase
custom_tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer() #separate by spaces
#display
print(custom_tokenizer.pre_tokenizer.pre_tokenize_str("This is an example!"))

special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens)#create trainer
custom_tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)#train

#post-processing------------------------------------------------------------------
cls_token_id = custom_tokenizer.token_to_id("[CLS]")
sep_token_id = custom_tokenizer.token_to_id("[SEP]")
print(cls_token_id, sep_token_id)

custom_tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", cls_token_id),
        ("[SEP]", sep_token_id),
    ],
)

#check encoding-------------------------------------------------------------------
encoding = custom_tokenizer.encode("This is one sentence.", "With this one we have a pair.")
print(encoding.tokens)
print(encoding.type_ids)

#save pre-trained-------------------------------------------------------------------
custom_tokenizer.save_pretrained("my-new-tokenizer")
