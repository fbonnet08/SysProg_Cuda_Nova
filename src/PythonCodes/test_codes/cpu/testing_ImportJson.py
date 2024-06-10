import json
import os
import tqdm
file_lst = []

#filename = os.path.join('E:','data','ToBeInserted','GNPS-LIBRARY.json')
#file_lst.append(filename)

#################################################################################
#'''
filename = os.path.join('E:','data','ToBeInserted','MoNA-export-LC-MS-MS_Positive_Mode.json')
file_lst.append(filename)

kfile = open(filename, encoding='utf-8')
lines = kfile.readlines()
print(lines[1])
print(" filename ---->: "+ filename + " has length ----> ", len(lines))
kfile.close()

filename = os.path.join('E:','data','ToBeInserted','MoNA-export-LC-MS-MS_Negative_Mode.json')
file_lst.append(filename)

lfile = open(filename, encoding='utf-8')
lines = lfile.readlines()
print(lines[1])
print(" filename ---->: "+ filename + " has length ----> ", len(lines))
lfile.close()


#'''
#filename = os.path.join('E:','data','ToBeInserted','MoNA-export-LC-MS-MS_Negative_Mode.json')
#file_lst.append(filename)
#filename = os.path.join('E:','data','ToBeInserted','MoNA-export-LC-MS-MS_Positive_Mode.json')
#file_lst.append(filename)
#filename = os.path.join('E:','data','ToBeInserted','MoNA-export-All_Spectra.json')
#file_lst.append(filename)

#################################################################################

#################################################################################
filename = os.path.join('E:','data','ToBeInserted','NEG_LC.json')
file_lst.append(filename)
filename = os.path.join('E:','data','ToBeInserted','POS_LC.json')
file_lst.append(filename)

print("filename --->: ", filename)
print("filelist[:] --->: ", file_lst[:])

for i in range(len(file_lst)):
    if os.path.isfile(file_lst[i]):
        print("file: "+file_lst[i]+" exists")
    else:
        print("file: "+file_lst[i]+" does not exists")

all_json_molecule_lst = []
json_file_all_molecule_dict = {}
for i in tqdm.tqdm(range(len(file_lst[:])), ncols=90, desc='json file        :'):
    idata = {}
    jfile = open(file_lst[i], encoding='utf-8')
    jsondata = json.load(jfile)
    cnt = 0
    for idata in jsondata:
        cnt += 1
        ith_data_key = []
        ith_data_value = []
        for key, value in idata.items():
            ith_data_key.append(key)
            ith_data_value.append(value)
        # [end-loop]
        #print(" idata["+str(ith_data_key[0])+"] --->: ", idata[ith_data_key[0]])
        #print(" idata filename: "+str(file_lst[i])+"--->: \n", idata)
        all_json_molecule_lst.append(ith_data_key[:])
        if cnt == 1: break
        #print("ith_data_key[:] ---->: ", ith_data_key[:])
    # [end-loop]
    json_file_all_molecule_dict[str(i)] = all_json_molecule_lst
    #print("For json file:"+str(all_json_molecule_lst[i])+"   length    ---->: ", len((all_json_molecule_lst[i][:])))
    # [end-loop] fior idata
    print("\nNumber of melcules in the Json "+str(file_lst[i])+"---->: ", cnt)
    #print("\n")
    jfile.close()
#end [end-loop] for file_lst
