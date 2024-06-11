'''!\file
   -- DataManage_ScanAllTableCnts4JSon: (Python3 code) is a Python3 module to update in a real time
   setting the table count of a given database using the querry class. This module is part
   of LDMaP-App but independent. The output of the JSon can be/is used as inputs for the PreProcessing

Please find the manual at https://sourceforge.net/projects/ldmap/

Author:
  F.D.R. Bonnet

This package is released under the Creative Commons
Attribution-NonCommercial-NoDerivs CC BY-NC-ND License
(http://creativecommons.org/licenses/by-nc-nd/3.0/)

Usage:
  DataManage_ScanAllTableCnts4JSon.py [--sleep_time=SLEEP_TIME]

NOTE: INPUT(s) is/are mandatory

Example:
     python3.8 DataManage_ScanAllTableCnts4JSon.py --sleep_time=60

Arguments:
  INPUTS                      Input For commputation is needed as shown below

Options:
  --sleep_time=SLEEP_TIME     Time to sleep between the querry table counts
  --help -h                   Show this help message and exit.
  --version                   Show version.
'''
# System tools
#import json
import sys
import os
import time
#from subprocess import PIPE, run
# appending the utils path
sys.path.append(os.path.join(os.getcwd(), '.'))
sys.path.append(os.path.join(os.getcwd(), '.', 'src', 'PythonCodes'))
sys.path.append(os.path.join(os.getcwd(), '.', 'src', 'PythonCodes', 'utils'))
sys.path.append(os.path.join(os.getcwd(), '.', 'src', 'PythonCodes', 'SQL', 'MsSQL', 'sql_handlers'))
# application imports
import src.PythonCodes.DataManage_common
import src.PythonCodes.utils.messageHandler
import src.PythonCodes.DataManage_header
import src.PythonCodes.utils.JSonCreator
import src.PythonCodes.utils.DjangoLauncher
import src.PythonCodes.SQL.MsSQL.sql_handlers.QueryMsSQL
import src.PythonCodes.docopt
#-------------------------------------------------------------------------------
# [Methods]
#-------------------------------------------------------------------------------
global running
#-------------------------------------------------------------------------------
# [Main]
#-------------------------------------------------------------------------------
# DataManageApp application main
if __name__ == '__main__':

    # instantiating the common class
    running = True
    c = src.PythonCodes.DataManage_common.DataManage_common()
    rc = c.get_RC_SUCCESS()
    # instantiating messaging class
    logfile = c.getLogfileName()  # getting the name of the global log file
    m = src.PythonCodes.utils.messageHandler.messageHandler(logfile=logfile)

    # argument stuff for the command line
    version = src.PythonCodes.DataManage_common.DataManage_version()
    args = src.PythonCodes.docopt.docopt(__doc__, version=version)

    # printing the header of Application
    src.PythonCodes.DataManage_header.print_MolRefAnt_DB_Scan_header(common=c, messageHandler=m)
    # platform, release  = whichPlatform()
    sysver, platform, system, release, node, processor, cpu_count = src.PythonCodes.DataManage_common.whichPlatform()
    # if (system == 'Linux'):
    m.printMesgStr("System            : ", c.get_B_Green(), system)
    m.printMesgStr("System time stamp : ", c.get_B_Yellow(), sysver)
    m.printMesgStr("Release           : ", c.get_B_Magenta(), release)
    m.printMesgStr("Linux Kernel      : ", c.get_B_Cyan(), platform)
    m.printMesgStr("Node              : ", c.get_B_Yellow(), node)
    m.printMesgStr("Processor type    : ", c.get_B_Red(), processor)
    m.printMesgStr("CPU cores count   : ", c.get_B_Green(), cpu_count)
    # ---------------------------------------------------------------------------
    # Command line handler for now. TODO: need to immplement class
    # ---------------------------------------------------------------------------
    # --app_root=APP_ROOT {required}
    SLEEP_TIME = 20
    if args['--sleep_time']:
        SLEEP_TIME = args['--sleep_time']
    else:
        m.printCMesg("No Sleeping time specified.",c.get_B_Red())
        m.printMesgAddStr(" sleep_time        : ",c.getRed(),args['--sleep_time'])
        m.printMesgAddStr(" Setting sleep_time: ",c.getGreen(),str(20)+" seconds")
        m.printMesgInt("Return code       : ",c.get_B_White(), c.get_RC_WARNING())

    m.printMesgStr("Sleeping time     : ", c.get_B_Green(), str(SLEEP_TIME))
    # --------------------------------------------------------------------------
    # [Paths-Setup]
    # --------------------------------------------------------------------------
    APP_ROOT = os.getcwd()
    # DATA_PATH   = os.path.join('E:', 'data', 'LC_MS')
    # DATA_PATH         = os.path.join('C:', os.path.sep, 'Users', 'Frederic', 'OneDrive', 'UVPD-Perpignan', 'SourceCodes', 'SmallData')
    DATA_PATH = os.path.join(os.path.sep, 'data', 'frederic')
    PROJECTNAME = ""
    POOL_COMPONENTDIR = ""
    # TARGETDIR   = os.path.join('E:','DataProcInterCom')
    # TBLECNTS_DIR= os.path.join('E:','DataProcInterCom','TableCounts')
    # TARGETDIR  = os.path.join('C:', os.path.sep, 'Users', 'Frederic', 'OneDrive', 'UVPD-Perpignan', 'SourceCodes', 'SmallData', 'DataProcInterCom')
    TARGETDIR = os.path.join(os.path.sep, 'data', 'frederic', 'DataProcInterCom')
    # TBLECNTS_DIR      = os.path.join('C:', os.path.sep, 'Users', 'Frederic', 'OneDrive', 'UVPD-Perpignan', 'SourceCodes', 'SmallData', 'DataProcInterCom', 'TableCounts')
    TBLECNTS_DIR = os.path.join(os.path.sep, 'data', 'frederic', 'DataProcInterCom', 'TableCounts')

    SOFTWARE    = "N/A"
    c.setApp_root(APP_ROOT)
    c.setData_path(DATA_PATH)
    c.setProjectName(PROJECTNAME)
    c.setPool_componentdir(POOL_COMPONENTDIR)
    c.setTargetdir(TARGETDIR)
    c.setSoftware(SOFTWARE)
    c.setJSon_TableCounts_Dir(TBLECNTS_DIR)
    APP_DATA_PATH = "N/A"        #TARGETDIR+os.path.sep+PROJECTNAME+"_Pool"
    #---------------------------------------------------------------------------
    # Checking if the path structure exists if not create it
    #---------------------------------------------------------------------------
    m.printMesgStr("Check/Create      : ",c.getMagenta(), TARGETDIR.strip() )
    if os.path.exists(TARGETDIR.strip()):
      sys.path.append(TARGETDIR)
      m.printMesgAddStr("  "+TARGETDIR+" : ", c.getGreen(),"Exits nothing to do")
    else:
      m.printMesgAddStr("  "+TARGETDIR+" : ", c.getRed(),"Does not Exits we will create it...")
      os.mkdir(TARGETDIR)
      m.printMesgAddStr("                   : ", c.getGreen(),"Done.")
      sys.path.append(TARGETDIR)

    m.printMesgStr("Check/Create      : ",c.getMagenta(), TBLECNTS_DIR.strip() )
    if os.path.exists(TBLECNTS_DIR.strip()):
      sys.path.append(TBLECNTS_DIR)
      m.printMesgAddStr("  "+TBLECNTS_DIR+" : ", c.getGreen(),"Exits nothing to do")
    else:
      m.printMesgAddStr("  "+TBLECNTS_DIR+" : ", c.getRed(),"Does not Exits we will create it...")
      os.mkdir(TBLECNTS_DIR)
      m.printMesgAddStr("               : ", c.getGreen(),"Done.")
      sys.path.append(TBLECNTS_DIR)
    #---------------------------------------------------------------------------
    # Instantiating the Query class QueryMsSQL to get methods
    #---------------------------------------------------------------------------
    queryMsSQL = src.PythonCodes.SQL.MsSQL.sql_handlers.QueryMsSQL.QueryMsSQL(c, m)
    #---------------------------------------------------------------------------
    # Getting into the DataBase
    #---------------------------------------------------------------------------
    m.printMesg("Connecting to the Database...")
    m.printMesgAddStr(" Database server   : ", c.getGreen(), c.getDatabase_server())
    m.printMesgAddStr(" Database name     : ", c.getYellow(), c.getDatabase_name())
    m.printMesgAddStr(" Database schema   : ", c.getMagenta(), c.getDatabase_schema())
    m.printMesgAddStr(" Database port     : ", c.getCyan(), c.getDatabase_port())
    m.printMesgAddStr(" Connecting        : ", c.getCyan(),"...")
    rc, cnxn = queryMsSQL.open_connection_database_PostGreSQL()
    if rc != c.get_RC_SUCCESS(): exit(c.get_RC_FAIL())
    #---------------------------------------------------------------------------
    # Creating the Cursor
    #---------------------------------------------------------------------------
    m.printMesg("Creating the cursor")
    rc, cursor = queryMsSQL.create_cursor(cnxn)
    if rc != c.get_RC_SUCCESS(): exit(c.get_RC_FAIL())
    #---------------------------------------------------------------------------
    # Getting the full table list from the database
    #---------------------------------------------------------------------------
    while running:
        m.printMesg("Getting the table list from the database")
        m.printMesgAddStr(" Database          : ",c.getYellow(),
                          queryMsSQL.credentialsMsSQL.database_PostGreSQL)
        m.printMesgAddStr(" Schema            : ",c.getYellow(),
                          queryMsSQL.credentialsMsSQL.schema_PostGreSQL)
        rc, full_table_list = queryMsSQL.query_PostGreSQL_GetFullTable_list(cursor)
        if rc != c.get_RC_SUCCESS(): exit(c.get_RC_FAIL())
        #---------------------------------------------------------------------------
        # Creating the Json file for the table counts...
        #---------------------------------------------------------------------------
        m.printMesg("Creating the Json file for the table counts")
        rc, table_list = queryMsSQL.generateTable_PostGreSQL_list(full_table_list)
        if rc != c.get_RC_SUCCESS(): exit(c.get_RC_FAIL())
        #---------------------------------------------------------------------------
        # Table counts
        #---------------------------------------------------------------------------
        m.printMesg("Table counts")
        print("-------------------------------------------------------------------")
        table_count_list = []
        for i in range(len(table_list[:])):
            rc, table_count = queryMsSQL.query_PostGreSQL_GetTableCount(cursor,
                                                                        queryMsSQL.credentialsMsSQL.database_PostGreSQL,
                                                                        queryMsSQL.credentialsMsSQL.schema_PostGreSQL,
                                                                        table_list[i], silent=False)
            table_count_list.append(table_count)
        print("-------------------------------------------------------------------")
        #---------------------------------------------------------------------------
        # [extracting] extracting the Django models
        #---------------------------------------------------------------------------
        ext_json = ".json"
        json_table_file = "current_AllTableCount_rec"+ext_json
        json_write_path = TBLECNTS_DIR

        djangoLauncher = src.PythonCodes.utils.DjangoLauncher.DjangoLauncher(c, m)
        rc = djangoLauncher.generate_djangoModel(table_list=table_list)
        #---------------------------------------------------------------------------
        # Instantiating the JSon file Launcher Class to get methods
        #---------------------------------------------------------------------------
        jsonCreator = src.PythonCodes.utils.JSonCreator.JSonCreator(c, m)
        #---------------------------------------------------------------------------
        # Creating the table count for the json file
        #---------------------------------------------------------------------------
        jsonCreator.setJSon_Table_filename(json_table_file)
        jsonCreator.setJSon_Table_write_path(json_write_path)
        rc = jsonCreator.create_TableCount_JSon_file(table_list=table_list,
                                                     table_count_list=table_count_list)
        #---------------------------------------------------------------------------
        # [Clearing] clearing the list from memory to avoid duplication
        #---------------------------------------------------------------------------
        table_count_list.clear()
        table_list.clear()
        full_table_list.clear()
        #---------------------------------------------------------------------------
        # [Sleeping] sleeping time between scans
        #---------------------------------------------------------------------------
        m.printMesg("Sleeping between each database scanning")
        progressBar = src.PythonCodes.utils.progressBar.ProgressBar()
        for i in range(int(SLEEP_TIME)):
            time.sleep(1)
            progressBar.update(1, int(SLEEP_TIME))
            progressBar.printEv()
        m.printLine()
        progressBar.resetprogressBar()
        #---------------------------------------------------------------------------
        # [Final] overall return code
        #---------------------------------------------------------------------------
        src.PythonCodes.DataManage_common.getFinalExit(c, m, rc)
        exit(0)
#-------------------------------------------------------------------------------
# [End]
#-------------------------------------------------------------------------------
