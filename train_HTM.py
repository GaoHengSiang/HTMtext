#import libraries------------------------------------
from datasets import load_dataset
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizer
import numpy as np
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM
from tqdm import tqdm

#settings--------------------------------------------
vocab_size = 10000
batch_size = 1000
arraySize = vocab_size
inputSDR = SDR( arraySize )

tm = TM(columnDimensions          = (inputSDR.size,),
        cellsPerColumn            = 32,                 # default: 32
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

#slice small portion
#dataset = dataset.select(range(10))

#acquire tokenizer-----------------------------------
custom_tokenizer = Tokenizer.from_file("my-new-tokenizer.json") #self trained
#training tokenizers is quick (<30s)


for cycle in range(3):
    #print('CURRENTLY IN CYCLE = ', cycle+1, "==================================")
    description = f'Processing sentences, cycle = {str(cycle+1)}'
    for sentence in tqdm(dataset, desc=description):
        #tokenize sentences----------------------------------
        #sequence = "Using a Transformer network is simple" #dummy
        #should use wikitext
        #print(sentence['text'])
        sequence = str(sentence)
        
        encodings = custom_tokenizer.encode(sequence) #--> sentence
        #print(encodings.tokens) #display

        id_seq = (encodings.ids)
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
            #print the active cell ids --------------------------
            # active_cell_ids = tm.cellsToColumns(tm.getActiveCells()).sparse
            # print('active cells = ', active_cell_ids)
            # decoded_string = custom_tokenizer.decode(active_cell_ids)
            # print('current token: ', decoded_string) #print the current processing token
            
            # tm.activateDendrites(True) #necessary, call before getPredictiveCells
            # #print/acquire the predicted cell ids
            # predicted_cell_ids = tm.cellsToColumns(tm.getPredictiveCells()).sparse
            # decoded_string = custom_tokenizer.decode(predicted_cell_ids)
            # print('predicted next token: ', decoded_string) #print the current processing token

#save trained model
# File to save the TemporalMemory state
filename = 'trained_HTM'
format = 'BINARY'  # Can be 'BINARY', 'PORTABLE', 'JSON', or 'XML'
#Only use BINARY to avoid load error

# Save the TemporalMemory state to a file
try:
    tm.saveToFile(filename, format)
    print(f"TemporalMemory state saved to {filename} in {format} format.")
except Exception as e:
    print("Error during save:", e)

#get user input--------------------------------------


#pass into TM and get prediction---------------------


#pass prediction into TM-----------------------------


#decode and print prediction-------------------------

