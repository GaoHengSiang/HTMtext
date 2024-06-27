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
filename = 'trained_HTM'
format = 'BINARY'
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

#load HTM model--------------------------------------
# Load the TemporalMemory state from the file
try:
    tm.loadFromFile(filename, fmt=format)
    print("Load successful.")
except Exception as e:
    print("Error during load:", e)

#acquire tokenizer-----------------------------------
custom_tokenizer = Tokenizer.from_file("my-new-tokenizer.json") #self trained
#training tokenizers is quick (<30s)


for cycle in range(10):
    #print the active cell idss
    active_cell_ids = tm.cellsToColumns(tm.getActiveCells()).sparse
    print('active cells = ', active_cell_ids)
    decoded_string = custom_tokenizer.decode(active_cell_ids)
    print('current token: ', decoded_string) #print the current processing token
    
    tm.activateDendrites(True) #necessary, call before getPredictiveCells
    #print/acquire the predicted cell ids
    predicted_cell_ids = tm.cellsToColumns(tm.getPredictiveCells()).sparse
    decoded_string = custom_tokenizer.decode(predicted_cell_ids)
    print('predicted next token: ', decoded_string) #print the current processing token

    #get user input--------------------------------------
    # Asking for user input
    user_input = input("Please enter something: ")

    for i in range(1): # in dataset:
        #tokenize sentences----------------------------------
        sequence = user_input #user input is string type
        #should use wikitext

        encodings = custom_tokenizer.encode(sequence) #--> sentence
        tokens_seq = encodings.tokens[1:-1]
        id_seq = encodings.ids[1: -1]

        print('user input: ', tokens_seq, id_seq) #display

        for id in id_seq:
            #encode to SDR---------------------------------------
            sensorValueBits = inputSDR.dense
            sensorValueBits = np.zeros(arraySize)
            sensorValueBits[id] = 1 #this has no semantic meaning
            #ideally words with close relationships should have some overlap or such

            inputSDR.dense = sensorValueBits
            #inputSDR.sparse = id #shorter code

            #pass into TM----------------------------------------
            tm.compute(inputSDR, learn = False) #learn = False for now for simplicity


#pass prediction into TM-----------------------------


#decode and print prediction-------------------------