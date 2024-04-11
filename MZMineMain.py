#!/usr/bin/env python
'''!\file
    -- MZMineMain.py usage addon: (Python3 code) Module for analysing spectroscopy data.
    The --rawfile option asks for a raw file but requires a mgf file that has been exported
    from MZMine3. The raw file extension on the input file

Contributors:
    F.D.R. Bonnet 03 April 2024

Usage:
    MZMineMain.py [--rawfile=RAW_FILE]
                  [--scan_number=SCAN_NUMBER]
                  [--RT=RET_TIME]

NOTE: select and option

Arguments: Either choose the --scan_number or the retention tim --RT

Options:
  --rawfile=RAW_FILE                 raw file to be analyzed
  --scan_number=SCAN_NUMBER          scan number in the raw file
  --RT=RET_TIME                      Retention time in the spectral sequence
  --help -h                          Show this help message and exit.
  --version                          Show version.
'''
# system imports
import os
import sys
# path extension
sys.path.append(os.path.join(os.getcwd(), '.'))
sys.path.append(os.path.join(os.getcwd(), '.','src','PythonCodes'))
sys.path.append(os.path.join(os.getcwd(), '.','src','PythonCodes','utils'))
#Applition imports
import src.PythonCodes.DataManage_common
import src.PythonCodes.src.Usage_Network
import src.PythonCodes.utils.messageHandler
import src.PythonCodes.utils.Command_line
import src.PythonCodes.utils.MZmineModel_Analyser
from docopt import docopt

#C:\Program Files\Python312\python.exe
ext_asc = ".asc"
ext_csv = ".csv"
ext_raw = ".raw"

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
    # setting up the argument list and parsing it
    args = docopt(__doc__, version=version)
    #---------------------------------------------------------------------------
    # system details
    #---------------------------------------------------------------------------
    # TODO: need to insert the system details but ot needed now
    #---------------------------------------------------------------------------
    # Setting the variable into the common class
    #---------------------------------------------------------------------------
    l = src.PythonCodes.utils.Command_line.Command_line(args, c, m)
    # building the command line
    l.createScan_number()
    l.createRet_time()

    m.printMesgStr("This is the main program      : ", c.get_B_Magenta(), "MZMineMain.py")
    if args["--rawfile"]:
        l.createRAW_file()
        m.printMesgStr("Raw file that will be analyzed: ", c.getYellow(), c.getRAW_file())
        raw_Analyser = src.PythonCodes.utils.MZmineModel_Analyser.MZmineModel_Analyser(c, m)
        if args["--scan_number"]:
            scan_number = c.getScan_number()   # 2081
            RT = c.getRet_time()               # 13.36
            rc, mgf_len, spectrum = raw_Analyser.read_mgf_file(c.getMGF_file(), scan_number, RT)
            m.printMesgAddStr("spectrum[:]                --->: ", c.getYellow(), spectrum[:])
            rc, mz_I_Rel, mz_I_Rel_sorted = raw_Analyser.extract_sequence_from_spec(spectrum)
            #Graphing output
            rc = raw_Analyser.make_histogram(mz_I_Rel)



    # ---------------------------------------------------------------------------
    # [Final] overall return code
    # ---------------------------------------------------------------------------
    # Final exit statements
    src.PythonCodes.DataManage_common.getFinalExit(c, m, rc)
    # ---------------------------------------------------------------------------
    # End of testing script
    # ---------------------------------------------------------------------------



