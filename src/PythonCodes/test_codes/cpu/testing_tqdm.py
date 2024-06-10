'''!\ffile
   -- testing_progressBar: (Python3 code) is a Python3 module to update in a real time
   setting the table count of a given data base using the querry class. This module is part
   of LDMaP-App but indepedent. The output of the JSon can be/is used as inputs for the PreProcessing

Please find the manual at https://sourceforge.net/projects/ldmap/

Author:
  F.D.R. Bonnet

This package is released under the Creative Commons
Attribution-NonCommercial-NoDerivs CC BY-NC-ND License
(http://creativecommons.org/licenses/by-nc-nd/3.0/)

Usage:
  testing_progressBar.py [--sleep_time=SLEEP_TIME]

NOTE: INPUT(s) is/are mandatory

Example:
    [Winodws]
    python3.8 testing_progressBar.py --sleep_time=1
    [Linux]
    python    testing_progressBar.py --sleep_time=1

Arguments:
  INPUTS                          Input For commputation is needed as shown below

Options:
  --sleep_time=SLEEP_TIME         Time to sleep between the querry table counts
  --help -h                       Show this help message and exit.
  --version                       Show version.
'''
# -*- coding: us-ascii -*-
#----------------------------------------------------------------------------
# Function to test the progress bar
# Author: Frederic Bonnet
# Date: 10/07/2021
#----------------------------------------------------------------------------
#System tools
import sys
import os
import time
from tqdm import tqdm
import progressbar
#appending the utils path
sys.path.append(os.path.join(os.getcwd(), '.'))
sys.path.append(os.path.join(os.getcwd(), '..','..'))
sys.path.append(os.path.join(os.getcwd(), '..','..','..','..'))
sys.path.append(os.path.join(os.getcwd(), '..','..','..','..','utils'))
#application imports
#from docopt import docopt
import docopt
import src.PythonCodes.DataManage_common
import src.PythonCodes.utils.messageHandler
import src.PythonCodes.DataManage_header
import src.PythonCodes.utils.StopWatch
#---------------------------------------------------------------------------
# Start of the test script
#---------------------------------------------------------------------------
if __name__ == '__main__':
    __func__ = sys._getframe().f_code.co_name
    #instantiating the common class
    c = src.PythonCodes.DataManage_common.DataManage_common()
    rc = c.get_RC_SUCCESS()
    #instantiating messaging class
    logfile = c.getLogfileName()
    m = src.PythonCodes.utils.messageHandler.messageHandler(logfile = logfile)
    #argument stuff for the command line
    version = src.PythonCodes.DataManage_common.DataManage_version()
    args = docopt.docopt(__doc__, version=version)
    rc = c.get_RC_SUCCESS()
    #printing the header of Application
    src.PythonCodes.DataManage_header.print_header(common=c,messageHandler=m)
    #---------------------------------------------------------------------------
    # Command line handler for now. TODO: need to immplement class
    #---------------------------------------------------------------------------
    #--sleep_time=SLEEP_TIME
    SLEEP_TIME = 5
    if args['--sleep_time']:
        SLEEP_TIME = str(args['--sleep_time'])
    else:
        m.printCMesg("No Sleeping time specified.", c.get_B_Red())
        m.printMesgAddStr(" sleep_time        : ", c.getRed(), args['--sleep_time'])
        m.printMesgAddStr(" Setting sleep_time: ", c.getGreen(), str(SLEEP_TIME) + " seconds")
        m.printMesgInt("Return code: ", c.get_B_White(), c.get_RC_WARNING())
    #---------------------------------------------------------------------------
    # [Start]
    #---------------------------------------------------------------------------
    m.printCMesgCVal("Sleep_taken: ", c.getYellow(), str(SLEEP_TIME),
                     c.getGreen()+" Seconds")
    m.printMesg("StopWatch reports...")
    #creating the timers and starting the stop watch...
    stopwatch = src.PythonCodes.utils.StopWatch.createTimer()
    src.PythonCodes.utils.StopWatch.StartTimer(stopWath=stopwatch)

    m.printMesg("Sleeping time")
    if float(SLEEP_TIME) < 1:
        m.printMesgAddStr(" No Sleep between each scanning: ", c.getGreen(),
                          str(SLEEP_TIME)+" seconds")
    if float(SLEEP_TIME) >= 1:
        for i in tqdm(range(int(SLEEP_TIME)), ncols=90, desc='The text for the loop'): time.sleep(1)

    src.PythonCodes.utils.StopWatch.StopTimer_secs(stopWath=stopwatch)
    time_taken = src.PythonCodes.utils.StopWatch.GetTimerValue_secs(stopWath=stopwatch)
    m.printCMesgCVal("time_taken: ", c.getYellow(),
                     str("{0:>.3f}".format(time_taken)),
                     c.getGreen()+" Seconds")
    #---------------------------------------------------------------------------
    # [Final] ovrall return code
    #---------------------------------------------------------------------------
    src.PythonCodes.DataManage_common.getFinalExit(c,m,rc)
    #---------------------------------------------------------------------------
    # End of testing script
    #---------------------------------------------------------------------------
