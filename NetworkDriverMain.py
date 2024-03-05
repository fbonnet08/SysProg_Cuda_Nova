#!/usr/bin/env python
'''!\file
   -- Network usage addon: (Python3 code)

Contributors:
  F.D.R. Bonnet 01 March 2024

Usage:
  DataManage_GpuConfigure.py [--with_while_loop=WITH_WHILE_LOOP]
                             [--plot_asc=PLOT_ASC]
                             [--csvfile=CSV_FILE]

NOTE: select and option

Arguments:
  --build_cuda or --clean_build_cuda and --cudaver=V10.2.89

Options:
  --with_while_loop=WITH_WHILE_LOOP  Use while loops
  --plot_asc=PLOT_ASC                Plot asc file
  --csvfile=CSV_FILE                 csv file to be analyzed
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
import src.PythonCodes.utils.MathematicalModel_Analyser
from docopt import docopt
#C:\Program Files\Python312\python.exe
ext_asc = ".asc"
ext_csv = ".csv"

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
    l.createWith_while_loop()
    
    m.printMesgStr("This is the main program : ", c.get_B_Magenta(), "NetworkDriverMain.py")
    # The network usage class on the interfaces with the while loop
    Usage_Network = src.PythonCodes.src.Usage_Network.Usage_Network(c, m)
    
    # Checking the command line structure and build the environment
    if args['--plot_asc']:
        l.createPlot_asc()
        m.printMesgStr("The file that will be logged is: ", c.getRed(), c.getPlot_asc())
    if args['--csvfile']:
        l.createCSV_file()
        m.printMesgStr("The csv file that will be analyzed: ", c.getYellow(), c.getCSV_file())
        PLOT_ASC = c.getCSV_file().split(ext_csv)[0]+ext_asc
        c.setPlot_asc(PLOT_ASC)

# Starting the probing
    csvfile = c.getCSV_file()
    ncnt = "while the copy continues"
    if (c.getWith_while_loop() == "yes"):
        m.printCMesgVal("The number of counter in the whle loop: ", c.getGreen() ,str(ncnt))
        rc = Usage_Network.Usage_NetworkInterfaces_whileLoop(csvfile)
    if (c.getWith_while_loop() == "no"):
        rc = Usage_Network.Usage_NetworkInterfaces_onePass(csvfile)
    
    # doing the analysis of the network usage
    if args['--with_while_loop'] == None:
        m.printMesgAddStr("There is no --->: ", c.getRed(), "[--with_while_loop]")
        if args['--csvfile']:
            m.printMesgAddStr("There is    --->: ", c.getYellow(), "[--csvfile]")
            msg = "We will now plot the stuff --->: " + c.getCSV_file()
            Math_Analyser = src.PythonCodes.utils.MathematicalModel_Analyser.MathematicalModel_Analyser(c, m)
            rc = Math_Analyser.printMesg(msg)
            #targetdir = os.path.join(os.getcwd(), '.','src','PythonCodes','Analysis_results')
            targetdir = "./src/PythonCodes/Analysis_results"
            
            rc = Math_Analyser.make_histogram(Math_Analyser.csvfile, targetdir)
            #rc = Math_Analyser.get_statistics()
        
    #---------------------------------------------------------------------------
    # [Final] ovrall return code
    #---------------------------------------------------------------------------
    # Final exit statements
    src.PythonCodes.DataManage_common.getFinalExit(c, m, rc)
    #---------------------------------------------------------------------------
    # End of testing script
    #---------------------------------------------------------------------------



