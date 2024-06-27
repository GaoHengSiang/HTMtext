from htm.bindings.algorithms import TemporalMemory
import json

# Initialize TemporalMemory with minimal parameters
tm = TemporalMemory(
    columnDimensions=[2048],
    cellsPerColumn=32,
    activationThreshold=13,
    initialPermanence=0.21,
    connectedPermanence=0.50,
    minThreshold=10,
    maxNewSynapseCount=20,
    permanenceIncrement=0.10,
    permanenceDecrement=0.10,
    predictedSegmentDecrement=0.0
)

# File to load the TemporalMemory state from
filename = 'test.json'
format = 'JSON'  # Ensure this matches the content format

# Load the TemporalMemory state from the file
# try:
#     print(f"Loading TemporalMemory state from {filename} in {format} format...")
#     tm.loadFromFile(filename, fmt=format)
#     print("Load successful.")
# except Exception as e:
#     print("Error:", e)
#     print("Filename:", filename)
#     print("Format:", format)
#     print("Type of filename:", type(filename))
#     print("Type of format:", type(format))






# Save the TemporalMemory state to a file
try:
    tm.saveToFile(filename, fmt=format)
    print(f"TemporalMemory state saved to {filename} in {format} format.")
except Exception as e:
    print("Error during save:", e)

# Load the TemporalMemory state from the file
try:
    tm.loadFromFile(filename, fmt=format)
    print("Load successful.")
except Exception as e:
    print("Error during load:", e)

# Open and print the content of the JSON file
# with open(filename, 'r') as file:
#     try:
#         json_content = json.load(file)
#         print("JSON content:", json_content)
#     except json.JSONDecodeError as e:
#         print("Invalid JSON content:", e)