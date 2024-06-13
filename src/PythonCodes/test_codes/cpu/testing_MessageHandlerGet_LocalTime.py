# system imports
import os
import sys
#appending the utils path
sys.path.append(os.path.join(os.getcwd(), '.'))
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append(os.path.join(os.getcwd(), '..','..'))
sys.path.append(os.path.join(os.getcwd(), '..','..','utils'))
#Applition imports
import src.PythonCodes.DataManage_common
import src.PythonCodes.utils.messageHandler
import src.PythonCodes.DataManage_header
import src.PythonCodes.docopt
#from docopt import docopt

try:
    import torch
    TORCH_AVAILABLE = True
    print("TORCH_AVAILABLE --->: ", TORCH_AVAILABLE)
    import src.PythonCodes.src.MachineLearning.DeepL
except (ImportError, NameError, AttributeError, OSError):
    rc = -1
    print(" Python package torch is not installed on your system, verify or install\n")
    print(" The MachineLearning.DeepL class will not be used and\n")
    TORCH_AVAILABLE = False
    print("TORCH_AVAILABLE --->: ", TORCH_AVAILABLE)

#C:\Program Files\Python312\python.exe
ext_asc = ".asc"
ext_csv = ".csv"
ext_raw = ".raw"
DEBUG = 0   # main debug switch in the entire application

# Main
if __name__ == "__main__":
    __func__= sys._getframe().f_code.co_name
    global version
    version = src.PythonCodes.DataManage_common.DataManage_version()
    c = src.PythonCodes.DataManage_common.DataManage_common()
    rc = c.get_RC_SUCCESS()
    # Getting the log file
    logfile = c.getLogfileName()  #getting the name of the global log file
    m = src.PythonCodes.utils.messageHandler.messageHandler(logfile = logfile)
    # printing the header of Application
    src.PythonCodes.DataManage_header.print_MolRefAnt_DB_header(common=c, messageHandler=m)
    #---------------------------------------------------------------------------
    # system details
    #---------------------------------------------------------------------------
    # TODO: need to insert the system details but ot needed now
    # platform, release  = whichPlatform()
    sysver, platform, system, release, node, processor, cpu_count = src.PythonCodes.DataManage_common.whichPlatform()
    #if (system == 'Linux'):
    m.printMesgStr("System                        : ", c.get_B_Green(), system)
    m.printMesgStr("System time stamp             : ", c.get_B_Yellow(), sysver)
    m.printMesgStr("Release                       : ", c.get_B_Magenta(), release)
    m.printMesgStr("Kernel                        : ", c.get_B_Cyan(), platform)
    m.printMesgStr("Node                          : ", c.get_B_Yellow(), node)
    m.printMesgStr("Processor type                : ", c.get_B_Red(), processor)
    m.printMesgStr("CPU cores count               : ", c.get_B_Green(), cpu_count)
    #---------------------------------------------------------------------------
    # Some Path structure
    #---------------------------------------------------------------------------
    c.setDebug(DEBUG)
    c.setApp_root(os.getcwd())
    m.printMesgStr("Application root path         :", c.getCyan(), c.getApp_root())


    rc, time_stamp = m.get_local_current_Time(c)


    m.printMesg("Query"+c.getGreen()+str(time_stamp)+c.getBlue()+" ---> "+c.getCyan()+__func__ )





