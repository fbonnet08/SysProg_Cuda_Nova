import os
import sys
from pathlib import Path

#C:\Users\Frederic\OneDrive\UVPD-Perpignan\SourceCodes\CLionProjects\SysProg-Cuda-Nova
current_dir = Path(__file__).parent.absolute()

#file_to_copy = sys.path.append(os.path.join(os.getcwd(), 'c:\\'))

file_to_copy = os.path.join(os.getcwd(),'C:\\','Users','Frederic','Desktop','cuda_12.3.2_546.12_windows.exe')
dir_to = os.path.join(os.getcwd(), 'A:\\','Frederic')

#os.system('ls -al %s'%file_to_copy)
os.system('ls -al %s'%dir_to)
os.system('ls -al %s'%file_to_copy)


print ("current directory is: ", current_dir)
print ("File to copy        : ", file_to_copy)
print ("copy file to        : ", dir_to)
