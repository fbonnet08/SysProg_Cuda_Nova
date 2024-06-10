import json

############
# Solution 1
############

# Load the JSON data from the file
file_path = 'E:/Json_DataBase/Metabolite_database_molecule000052_PEPMASS-377.3200.json'

# Open and read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract all "mz rel" values
mz_rel_values = []

for key, value in data.items():
    if key.startswith("mz rel:"):
        mz_rel_values.append(value)

# Print the extracted mz rel values
for mz, rel in mz_rel_values:
    print(f"m/z: {mz}, Intensity: {rel}")

############
# Solution 2
############

# Load the JSON data from the file
file_path = 'E:/Json_DataBase/Metabolite_database_molecule000052_PEPMASS-377.3200.json'

# Open and read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract all "mz rel" values
mz_rel_values = []

for key, value in data.items():
    if key.startswith("mz rel:"):
        mz_rel_values.append({
            "m/z": value[0],
            "Intensity": value[1]
        })

# Print the extracted mz rel values
for entry in mz_rel_values:
    print(f"m/z: {entry['m/z']}, Intensity: {entry['Intensity']}")