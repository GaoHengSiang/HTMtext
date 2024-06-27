#import libraries------------------------------------
from datasets import load_dataset
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizer
import numpy as np
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM

#settings--------------------------------------------
vocab_size = 10000
batch_size = 1000
arraySize = vocab_size
inputSDR = SDR( arraySize )

tm = TM(columnDimensions          = (inputSDR.size,),
        cellsPerColumn            = 10,                 # default: 32
        minThreshold              = 1,                  # default: 10
        activationThreshold       = 1,                  # default: 13
        initialPermanence         = 0.4,                # default: 0.21
        connectedPermanence       = 0.5,                # default: 0.5
        permanenceIncrement       = 0.1,                # default: 0.1
        permanenceDecrement       = 0.1,                # default: 0.1 
        predictedSegmentDecrement = 0.0,                # default: 0.0  --> #set to 0.05?
        maxSegmentsPerCell        = 1,                  # default: 255
        maxSynapsesPerSegment     = 1                   # default: 255
        )

#functions-------------------------------------------
# def formatSdr(sdr):
#   result = ''
#   for i in range(sdr.size):
#     if i > 0 and i % 8 == 0:
#       result += ' '
#     result += str(sdr.dense.flatten()[i])
#   return result

#acquire data----------------------------------------
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

#display
print(dataset)
print(dataset[1])
print(dataset[4])
#acquire tokenizer-----------------------------------
tokenizer = Tokenizer.from_file("my-new-tokenizer") #self trained
#training tokenizers is quick (<30s)


for cycle in range(2):
    for i in range(1): # in dataset:
        #tokenize sentences----------------------------------
        sequence = "Using a Transformer network is simple" #dummy
        #should use wikitext

        tokens = tokenizer.tokenize(sequence) #--> sentence
        #print(tokens) #display

        id_seq = tokenizer.convert_tokens_to_ids(tokens)
        #print(id_seq)#display

        for id in id_seq:
            #encode to SDR---------------------------------------
            sensorValueBits = inputSDR.dense
            sensorValueBits = np.zeros(arraySize)
            sensorValueBits[id] = 1 #this has no semantic meaning
            #ideally words with close relationships should have some overlap or such

            inputSDR.dense = sensorValueBits
            #inputSDR.sparse = id #shorter code

            #pass into TM----------------------------------------
            tm.compute(inputSDR, learn = True)
            #print the active cell ids
            active_cells = np.zeros(arraySize)
            active_cells = tm.getActiveCells()
            decoded_string = tokenizer.decode(np.nonzero(active_cells))
            print('current token: ', decoded_string) #print the current processing token
            
            tm.activateDendrites(True)
            #print/acquire the predicted cell ids
            predicted_cells = np.zeros(arraySize)
            predicted_cells = tm.getPredictiveCells()
            decoded_string = tokenizer.decode(np.nonzero(active_cells))
            print('predicted next token: ', decoded_string) #print the current processing token

#get user input--------------------------------------


#pass into TM and get prediction---------------------


#pass prediction into TM-----------------------------


#decode and print prediction-------------------------

