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

#load HTM model--------------------------------------
tm = TM.loadFromFile("trained_HTM.json", 'JSON')

#acquire tokenizer-----------------------------------
custom_tokenizer = Tokenizer.from_file("my-new-tokenizer.json") #self trained
#training tokenizers is quick (<30s)


for cycle in range(10):
    #get user input--------------------------------------
    # Asking for user input
    user_input = input("Please enter something: ")

    for i in range(1): # in dataset:
        #tokenize sentences----------------------------------
        sequence = user_input #user input is string type
        #should use wikitext

        encodings = custom_tokenizer.encode(sequence) #--> sentence
        print('user input: ', encodings.tokens, encodings.ids) #display

        id_seq = (encodings.ids)

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


#pass prediction into TM-----------------------------


#decode and print prediction-------------------------