import os
import sys

sys.path.append(os.path.join(os.getcwd(), '.'))
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append(os.path.join(os.getcwd(), '..','..'))
sys.path.append(os.path.join(os.getcwd(), '..','..','..'))
sys.path.append(os.path.join(os.getcwd(), '..','..','..','..'))
sys.path.append(os.path.join(os.getcwd(), '..','..','..','..','src','PythonCodes'))
sys.path.append(os.path.join(os.getcwd(), '..','..','..','..','src','PythonCodes','utils'))

import src.PythonCodes.DataManage_common
c = src.PythonCodes.DataManage_common.DataManage_common()

DATA_PATH         = os.path.join('E:', 'data', 'Nuage','FragHub','Input','MGF')

depth = 0
filenames_lst = []
dirpath_lst = []
dirname_lst = []

filestructure_dic = {}

for dirpath, dirname, filenames in os.walk(DATA_PATH):
    depth += 1
    dirpath_lst.append(dirpath)
    dirname_lst.append(dirname)
    filenames_lst.append(filenames)
    print("depth     --->: ", depth)
    filestructure_dic[dirpath] = filenames
# [end-loop]

print("dirpath_lst[:]        --->: ", dirpath_lst[:])
print("dirname_lst[:]        --->: ", dirname_lst[:])
#print("filenames_lst[:]      --->: ", filenames_lst[:])

print("len(dirpath_lst[:])   --->: ", len(dirpath_lst[:]))
print("len(dirname_lst[:])   --->: ", len(dirname_lst[:]))
print("len(filenames_lst[:]) --->: ", len(filenames_lst[:]))

print("len(dirpath_lst[0])   --->: ", len(dirpath_lst[0]))
print("len(dirname_lst[0])   --->: ", len(dirname_lst[0]))
print("len(filenames_lst[0]) --->: ", len(filenames_lst[0]))
print("len(dirpath_lst[1])   --->: ", len(dirpath_lst[1]))
print("len(dirname_lst[1])   --->: ", len(dirname_lst[1]))
print("len(filenames_lst[1]) --->: ", len(filenames_lst[1]))


print(filestructure_dic[dirpath_lst[0]])

for i in range(len(filestructure_dic[dirpath_lst[0]])):
    if ".mgf" in filestructure_dic[dirpath_lst[0]][i]:
        print("filestructure_dic[dirpath_lst[0]]["+str(i)+"] ---->: ", filestructure_dic[dirpath_lst[0]][i])

print("toltal depth          --->: ", depth)
print("depth                 --->: ", "done")




