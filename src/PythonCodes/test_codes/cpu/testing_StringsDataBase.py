import os
import sys
import json

smile1 = "C(C(\[O-])"
print("smile1             ---->: ", smile1)

smile2 = "C(C(\\[O-])"
print("smile2             ---->: ", smile2)

insert_msg1 = smile1.replace("\\", '\\\\')
print("insert_msg1        ---->: ", insert_msg1)


file_lst = []
filename = os.path.join('testing_Fields.json',)
file_lst.append(filename)

print("filename           --->: ", filename)
print("filelist[:]        --->: ",file_lst[:])
for i in range(len(file_lst)):
    if os.path.isfile(file_lst[i]):
        print("file: "+file_lst[i]+" exists")
    else:
        print("file: "+file_lst[i]+" does not exists")

for i in range(len(file_lst[:])):

    jfile = open(file_lst[0],'r')
    jsondata = json.load(jfile)
    cnt = 0
    for idata in jsondata:
        cnt += 1
        ith_data_key = []
        ith_data_value = []
        for key, value in idata.items():
            ith_data_key.append(key)
            ith_data_value.append(value)

        #print("ith_data_key[:] ---->: ", ith_data_key[:])
    # [end-loop] fior idata
    print("ith_data_key[:]    --->: ", ith_data_key[:])
    print("ith_data_value[:]  --->: ", ith_data_value[:])
    print("Number of melcules in the Json ---->: ", cnt)
    jfile.close()
#end [end-loop] for file_lst
