import numpy as np
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM
from time import sleep
from console import fg, bg, utils

def formatCell(cellName, activeState, winnerState, predictedState):
    styleFg = fg.white
    styleBg = bg.black
    style = None

    if(activeState == 1):
        styleFg = fg.green

    if(winnerState == 1):
        styleBg = bg.i22

    if(predictedState == 1):
        styleBg = bg.i241
    
    style = styleFg + styleBg
    if(style != None):
        result = style(format(cellName,'2d'))
    else:
        result = format(cellName,'2d')
    return result

def printHeader(step, sensorValue):
    print('-' * dashMultiplyer)
    print('| Cycle', format(cycle+1,'2d'), '| Step', format(step,'3d'), '| Value', format(sensorValue,'3d'),  '| Anomaly:', format(tm.anomaly, '.1f'), '|')
    print('-' * dashMultiplyer)

    colHeader = '| Column | '
    for colIdx in range(columns):
        colHeader +=  format(colIdx,'2d') + ' | '

    print(colHeader)
    print('-' * dashMultiplyer)

def printConnectionDetails(tm):
    for cell in range(columns * cellsPerColumn):
        segments = tm.connections.segmentsForCell(cell)

        for segment in segments:
            num_synapses = tm.connections.numSynapses(segment)
            for synapse in tm.connections.synapsesForSegment(segment):
                presynCell = tm.connections.presynapticCellForSynapse(synapse)                    
                permanence = tm.connections.permanenceForSynapse(synapse)
                print('cell', format(cell,'2d'), 'segment', format(segment,'2d'), 'has synapse to cell', format(presynCell,'2d'), 'with permanence', format(permanence,'.2f'))
            connected_synapses = tm.connections.numConnectedSynapses(segment)
            print('cell', format(cell,'2d'), 'segment', format(segment,'2d'), 'has', connected_synapses, 'connected synapse(s)')

def process(cycleArray):
    step = 1
    for sensorValue in cycleArray:
        sensorValueBits = inputSDR.dense
        sensorValueBits = np.zeros(columns)
        sensorValueBits[sensorValue] = 1
        inputSDR.dense = sensorValueBits

        tm.compute(inputSDR, learn = True)
        activeCells = tm.getActiveCells()
        tm.activateDendrites(True)

        activeCellsDense = activeCells.dense
        winnerCellsDense = tm.getWinnerCells().dense
        predictedCellsDense = tm.getPredictiveCells().dense

        utils.cls()
        printHeader(step, sensorValue)

        for rowIdx in range(cellsPerColumn):
            rowData = activeCellsDense[:,rowIdx]
            rowStr = '| Cell   | '
            for colI in range(rowData.size):
                cellName = np.ravel_multi_index([colI, rowIdx], (columns, cellsPerColumn))
                stateActive = activeCellsDense[colI,rowIdx]
                stateWinner = winnerCellsDense[colI,rowIdx]
                statePredicted = predictedCellsDense[colI,rowIdx]
                rowStr += formatCell(cellName, stateActive, stateWinner, statePredicted) + ' | ' 
            print(rowStr)
                
        print(tm.connections)
        printConnectionDetails(tm)
        print()
        step = step + 1
        sleep(0.5)

dashMultiplyer = 50

cycleArray = [0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]
cycles = 4

columns = 8
inputSDR = SDR( columns )
cellsPerColumn = 1

tm = TM(columnDimensions          = (inputSDR.size,),
        cellsPerColumn            = cellsPerColumn,     # default: 32
        minThreshold              = 1,                  # default: 10
        activationThreshold       = 1,                  # default: 13
        initialPermanence         = 0.4,                # default: 0.21
        connectedPermanence       = 0.5,                # default: 0.5
        permanenceIncrement       = 0.1,                # default: 0.1
        permanenceDecrement       = 0.1,                # default: 0.1 
        predictedSegmentDecrement = 0.0,                # default: 0.0
        maxSegmentsPerCell        = 1,                  # default: 255
        maxSynapsesPerSegment     = 1                   # default: 255
        )

for cycle in range(cycles):
    process(cycleArray)