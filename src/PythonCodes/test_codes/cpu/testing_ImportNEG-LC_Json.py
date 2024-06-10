import json
import os

all_json_molecule_lst = []
json_file_all_molecule_dict = {}


filename = os.path.join('E:','data','ToBeInserted','NEG_LC.json')
if os.path.isfile(filename):
    print("file: "+filename+" exists")
else:
    print("file: "+filename+" does not exists")

nfile = open(filename, 'r')
jsondata = json.load(nfile)
cnt = 0
for idata in jsondata:
    cnt += 1
    ith_data_key = []
    ith_data_value = []
    for key, value in idata.items():
        ith_data_key.append(key)
        ith_data_value.append(value)
    print(" idata['FILENAME'] --->: ", idata['FILENAME'])
    if cnt == 10: break
# [end-loop] fior idata
#print(ith_data_key[:])
#print(ith_data_value[:])
print("Number of melcules in the Json ---->: ", cnt)

print()

nfile.close()



