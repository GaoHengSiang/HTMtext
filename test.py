import numpy as np
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM

def formatSdr(sdr):
  result = ''
  for i in range(sdr.size):
    if i > 0 and i % 8 == 0:
      result += ' '
    result += str(sdr.dense.flatten()[i])
  return result

arraySize = 80
cycleArray = np.arange(0, 10, 1)
inputSDR = SDR( arraySize )
print("running...")

tm = TM(columnDimensions = (inputSDR.size,),
        cellsPerColumn=1,       # default: 32
        minThreshold=4,         # default: 10
        activationThreshold=8,  # default: 13
        initialPermanence=0.5,  # default: 0.21
        )

for cycle in range(2):
    for sensorValue in cycleArray:
        sensorValueBits = inputSDR.dense
        sensorValueBits = np.zeros(arraySize)
        sensorValueBits[sensorValue * 8:sensorValue * 8 + 8] = 1
        inputSDR.dense = sensorValueBits

        tm.compute(inputSDR, learn = True)
        print(format(sensorValue,'>2') + '/' + format(cycle, '1d')+ ' |', formatSdr(tm.getActiveCells()), 'Active')
    
        tm.activateDendrites(True)
        print(format(tm.anomaly, '.2f') + ' |', formatSdr(tm.getPredictiveCells()), 'Predicted')