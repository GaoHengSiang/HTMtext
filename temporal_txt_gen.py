#import libraries------------------------------------
from datasets import load_dataset
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizer
import numpy as np
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM

#settings--------------------------------------------
tm = TM(columnDimensions = (inputSDR.size,),
        cellsPerColumn=10,       # default: 32
        minThreshold=4,         # default: 10
        activationThreshold=8,  # default: 13
        initialPermanence=0.5,  # default: 0.21
        )
vocab_size = 1000

#acquire data----------------------------------------
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

#display
print(dataset)
print(dataset[1])
print(dataset[4])
#acquire tokenizer-----------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#Should use self-trained, will be much more compact and we can reduce vocab size

#tokenize sentences----------------------------------
sequence = "Using a Transformer network is simple" #dummy
#should use wikitext

tokens = tokenizer.tokenize(sequence)
print(tokens) #display

id_seq = tokenizer.convert_tokens_to_ids(tokens)
print(id_seq)#display


arraySize = vocab_size
inputSDR = SDR( arraySize )
for cycle in range(2):
    for id in id_seq:
        #encode to SDR---------------------------------------
        sensorValueBits = inputSDR.dense
        sensorValueBits = np.zeros(arraySize)
        sensorValueBits[id] = 1
        inputSDR.dense = sensorValueBits


        #pass into TM----------------------------------------
        tm.compute(inputSDR, learn = True)
        #print the active cell ids
    
        tm.activateDendrites(True)
        #print/acquire the predicted cell ids
        
#get user input--------------------------------------


#pass into TM and get prediction---------------------


#pass prediction into TM-----------------------------


#decode and print prediction-------------------------

