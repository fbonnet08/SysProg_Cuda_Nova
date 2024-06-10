import json
import os

all_json_molecule_lst = []
json_file_all_molecule_dict = {}

filename = os.path.join('MoNA-export-LC-MS_Positive_Mode_PrettyFormat_One.json')
if os.path.isfile(filename):
    print("file: "+filename+" exists")
else:
    print("file: "+filename+" does not exists")

nfile = open(filename)
jsondata = json.load(nfile)
json_length = len(jsondata)

cnt = 0
for idata in jsondata:
    cnt += 1
    ith_data_key = []
    ith_data_value = []

    metaData = []
    #for key, value in idata.items():
    #    ith_data_key.append(key)
    #    ith_data_value.append(value)
    #print(" idata['spectrum'] --->: ", idata['spectrum'])
    metaData = idata['metaData']



    print(" idata['metaData'] --->: ", idata['metaData'])
    print("len(metadata[:]) ---> ", len(metaData[:]))
    print("type(metadata[:]) ---> ", type(metaData[:]))
    print("\n")
    for i in range(len(metaData[:])):
        if metaData[i]['name'] == 'exact mass':
            pepmass = str(metaData[i]['value'])
            print("metaData["+str(i)+"]['name'] --->: "+str(metaData[i]['name'])+" ----> value ----> "+str(pepmass))
        if metaData[i]['name'] == 'precursor type':
            ionmode = str(metaData[i]['value'])
            print("metaData["+str(i)+"]['name'] --->: "+str(metaData[i]['name'])+" ----> value ----> "+str(ionmode))
        if metaData[i]['name'] == 'molecular formula':
            formula = str(metaData[i]['value'])
            print("metaData["+str(i)+"]['name'] --->: "+str(metaData[i]['name'])+" ----> value ----> "+str(formula))



        '''
        for j in range(len(metaData[i][:])):
            print("metaData["+str(i)+"] --->: ", metaData[i][j])

        '''


    #print(" idata['metaData'] --->: ", idata['metaData']['name'])
    if cnt == 10: break
# [end-loop] fior idata
#print(ith_data_key[:])
#print(ith_data_value[:])
print("Number of melcules in the Json ---->: ", cnt)

print()

nfile.close()



