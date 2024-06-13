#!/usr/bin/env python
"""!\file
    -- MZMineMain.py usage addon: (Python3 code) Module for analysing spectroscopy data.
    The --rawfile option asks for a raw file but requires a mgf file that has been exported
    from MZMine3. The raw file extension on the input file.

Contributors:
    Main author: F.D.R. Bonnet 03 April 2024

Usage:
    MZMineMain.py [--rawfile=RAW_FILE]
                  [--scan_number=SCAN_NUMBER]
                  [--RT=RET_TIME]
                  [--import_db=IMPORT_DB]
                  [--export_db=EXPORT_DB]
                  [--import_external_db=IMPORT_EXTERNAL_DB]
                  [--export_external_db=EXPORT_EXTERNAL_DB]
                  [--multiprocessing_type=MULTIPROCESSING_TYPE]
                  [--import_db_set=IMPORT_DB_SET]
                  [--database_origin=DATABASE_ORIGIN]
                  [--machine_learning=MACHINE_LEARNING]


NOTE: select and option

Arguments: Either choose the --scan_number or the retention tim --RT

    Examples:
        python MZMineMain.py --help

        default is serial:
        python MZMineMain.py --import_db=\'E:\Bacterial metabolites database.txt\'
        python MZMineMain.py --import_db=\'E:\Bacterial metabolites database.txt\' --multiprocessing_type=serial
        python MZMineMain.py --import_db=\'E:\Bacterial metabolites database.txt\' --multiprocessing_type=multiproc

Options:
    --rawfile=RAW_FILE                           raw file to be analysed
    --scan_number=SCAN_NUMBER                    scan number in the raw file
    --RT=RET_TIME                                Retention time in the spectral sequence
    --import_db=IMPORT_DB                        Database imports Internal {path to be imported from}
    --export_db=EXPORT_DB                        DataBase Exports Internal {path to be exported to}
    --import_external_db=IMPORT_EXTERNAL_DB      Database imports External {path to be imported from}
    --export_external_db=EXPORT_EXTERNAL_DB      DataBase Exports External {path to be exported to}
    --multiprocessing_type=MULTIPROCESSING_TYPE  Wether we are using serial or multiprocessing mode [serial, multiproc]
    --import_db_set=IMPORT_DB_SET                Import a set of databases=[single, set_in_dir]
    --database_origin=DATABASE_ORIGIN            Gives the database origin=[FragHub, LC-MS, MoNA, GNPS]
    --machine_learning=MACHINE_LEARNING          Use Machine learning
    --help -h                                    Show this help message and exit.
    --version                                    Show version.
"""
# system imports
import os
import sys
# path extension
sys.path.append(os.path.join(os.getcwd(), '.'))
sys.path.append(os.path.join(os.getcwd(), '.','src','PythonCodes'))
sys.path.append(os.path.join(os.getcwd(), '.','src','PythonCodes','utils'))
sys.path.append(os.path.join(os.getcwd(), '.','src','PythonCodes','SQL','MsSQL','sql_handlers'))
#Applition imports
import src.PythonCodes.DataManage_common
import src.PythonCodes.utils.messageHandler
import src.PythonCodes.utils.Command_line
import src.PythonCodes.DataManage_header
import src.PythonCodes.utils.MZmineModel_Analyser
import src.PythonCodes.utils.MgfTransformer
import src.PythonCodes.utils.DataBaseHandler
import src.PythonCodes.SQL.MsSQL.sql_handlers.MsSQLLauncher
import src.PythonCodes.SQL.MsSQL.sql_handlers.QueryMsSQL
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
    # setting up the argument list and parsing it
    args = src.PythonCodes.docopt.docopt(__doc__, version=version)
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
    #---------------------------------------------------------------------------
    # Setting the variable into the common class
    #---------------------------------------------------------------------------
    l = src.PythonCodes.utils.Command_line.Command_line(args, c, m)
    # building the command line
    l.createScan_number()
    l.createRet_time()
    # --------------------------------------------------------------------------
    # [Paths-Setup]
    # --------------------------------------------------------------------------
    PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/compound/"

    # linux VM machine
    if c.get_system() == "Linux":
        DATA_PATH         = os.path.join(os.path.sep, 'data')    # , 'LC_MS'
        DATAPROCINTERCOM  = os.path.join(os.path.sep, 'data', 'frederic', 'DataProcInterCom')
        TBLECNTS_DIR      = os.path.join(os.path.sep, 'data', 'frederic', 'DataProcInterCom', 'TableCounts')
        SQL_FULLPATH_DIR  = os.path.join(os.path.sep, 'data', 'frederic', 'SQLFiles_sql')
    # Windows Local machine
    if c.get_system() == "Windows":
        DATA_PATH         = os.path.join('E:', 'data')    # , 'LC_MS'
        DATAPROCINTERCOM  = os.path.join('E:', 'DataProcInterCom')
        TBLECNTS_DIR      = os.path.join('E:', 'DataProcInterCom', 'TableCounts')
        SQL_FULLPATH_DIR  = os.path.join('E:', 'SQLFiles_sql')
        #DATA_PATH         = os.path.join('C:', os.path.sep, 'Users', 'Frederic', 'OneDrive', 'UVPD-Perpignan', 'SourceCodes', 'SmallData')
        #DATAPROCINTERCOM  = os.path.join('C:', os.path.sep, 'Users', 'Frederic', 'OneDrive', 'UVPD-Perpignan', 'SourceCodes', 'SmallData','DataProcInterCom')
        #TBLECNTS_DIR      = os.path.join('C:', os.path.sep, 'Users', 'Frederic', 'OneDrive', 'UVPD-Perpignan', 'SourceCodes', 'SmallData','DataProcInterCom', 'TableCounts')
        #SQL_FULLPATH_DIR  = os.path.join('C:', os.path.sep, 'Users', 'Frederic', 'OneDrive', 'UVPD-Perpignan', 'SourceCodes', 'SmallData','SQLFiles_sql')
    # [end-if] statement
    APP_ROOT          = os.getcwd()
    PROJECTNAME       = ""
    POOL_COMPONENTDIR = ""
    SOFTWARE          = "N/A"
    SQL_DIR           = 'SQLFiles_sql'
    APP_DATA_PATH     = "N/A"
    # putting statting
    c.setApp_root(APP_ROOT)
    c.setData_path(DATA_PATH)
    c.setProjectName(PROJECTNAME)
    c.setPool_componentdir(POOL_COMPONENTDIR)
    c.setSoftware(SOFTWARE)
    c.setDataProcInterCom(DATAPROCINTERCOM)
    c.setJSon_TableCounts_Dir(TBLECNTS_DIR)
    c.setSql_dir(SQL_DIR)
    c.setSql_fullPath_dir(SQL_FULLPATH_DIR)
    c.setPubchem_url(PUBCHEM_URL)
    # --------------------------------------------------------------------------
    # [Single-Multiprocessing]
    # --------------------------------------------------------------------------
    l.createMultiprocessing_type()
    m.printMesgStr("Multiprocessing mode          :", c.get_B_Cyan(), c.getMultiprocessing_type())
    # --------------------------------------------------------------------------
    # [Paths-Setup]
    # --------------------------------------------------------------------------
    m.printMesgStr("This is the main program      :", c.get_B_Magenta(), "MZMineMain.py")
    if args["--rawfile"]:
        l.createRAW_file()
        m.printMesgStr("Raw file that will be analyzed:", c.getYellow(), c.getRAW_file())
        raw_Analyser = src.PythonCodes.utils.MZmineModel_Analyser.MZmineModel_Analyser(c, m)
        if args["--scan_number"]:
            scan_number = c.getScan_number()   # 2081
            RT = c.getRet_time()               # 13.36
            rc, mgf_len, spectrum = raw_Analyser.read_mgf_file(c.getMGF_file(), scan_number, RT)
            m.printMesgAddStr("spectrum[:]                --->: ", c.getYellow(), spectrum[:])
            rc, mz_I_Rel, mz_I_Rel_sorted = raw_Analyser.extract_sequence_from_spec(spectrum)
            #Graphing output
            rc = raw_Analyser.make_histogram(mz_I_Rel)
            # now getting the MgfTransformer class to transform the scanned
            mgfTranformer = src.PythonCodes.utils.MgfTransformer.MgfTransformer(c, m)
            rc = mgfTranformer.write_spectrum_to_json(spectrum=spectrum,
                                                      mz_I_Rel=mz_I_Rel,
                                                      mz_I_Rel_sorted=mz_I_Rel_sorted)
    #---------------------------------------------------------------------------
    # [--import_db] Creating the insertors and launcher bats for the SQL insertes
    #---------------------------------------------------------------------------
    if args["--import_db"]:
        l.createImport_db()
        m.printMesgStr("Database to be imported       :", c.getYellow(), c.getImport_db())
        database_handler = src.PythonCodes.utils.DataBaseHandler.DataBaseHandler(c, m, db_action="import_db")
        rc = database_handler.read_database_file(c.getDatabase_file())
        rc, molecule_number_lst = database_handler.create_molecule_dictionary()
        rc = database_handler.create_molecule_jsonfiles_from_molecules_dict()
        #print("molecule_number_lst[:]: ---> ", molecule_number_lst[:])
        #---------------------------------------------------------------------------
        # Create the SQL scripts and populate those in the project folder for
        #---------------------------------------------------------------------------
        mssqlLauncher = src.PythonCodes.SQL.MsSQL.sql_handlers.MsSQLLauncher.MsSQLLauncher(c, m)
        rc, list_len, JsonMoleculeFileList = mssqlLauncher.getListOfJsonFiles_MsSQLLauncher(mssqlLauncher.jsontargetfile)
        rc, jsonfile_content_dict_lst, jsonfile_content_key_lst, jsonfile_content_value_lst = mssqlLauncher.ReadinJson_Molecules_File_MsSQLLauncher(JsonMoleculeFileList)
        #print("jsonfile_content_dict_lst[:] --->: ", jsonfile_content_dict_lst[:])
        #print(jsonfile_content_key_lst[:])
        #print(jsonfile_content_value_lst[:])
        #---------------------------------------------------------------------------
        # TODO: - here insert the the table scanning count,
        #       - move this code to the object constructor of the QueryMsSQL class
        #---------------------------------------------------------------------------
        m.printMesgStr("Check/Create      : ", c.getMagenta(), str(c.getDataProcInterCom()).strip() )
        if os.path.exists(str(c.getDataProcInterCom()).strip()):
            sys.path.append(str(c.getDataProcInterCom()))
            m.printMesgAddStr("  "+c.getDataProcInterCom()+" : ", c.getGreen(),"Exits nothing to do")
        else:
            m.printMesgAddStr("  "+str(c.getDataProcInterCom())+" : ", c.getRed(),"Does not Exits we will create it...")
            os.mkdir(c.getDataProcInterCom())
            m.printMesgAddStr("                   : ", c.getGreen(),"Done.")
            sys.path.append(c.getDataProcInterCom())

        m.printMesgStr("Check/Create      : ",c.getMagenta(), c.getJSon_TableCounts_Dir().strip() )
        if os.path.exists(c.getJSon_TableCounts_Dir().strip()):
            sys.path.append(c.getJSon_TableCounts_Dir())
            m.printMesgAddStr("  "+c.getJSon_TableCounts_Dir()+" : ", c.getGreen(),"Exits nothing to do")
        else:
            m.printMesgAddStr("  "+c.getJSon_TableCounts_Dir()+" : ", c.getRed(),"Does not Exits we will create it...")
            os.mkdir(c.getJSon_TableCounts_Dir())
            m.printMesgAddStr("               : ", c.getGreen(),"Done.")
            sys.path.append(c.getJSon_TableCounts_Dir())
        #---------------------------------------------------------------------------
        # Instantiating the Query class QueryMsSQL to get methods
        #---------------------------------------------------------------------------
        queryMsSQL = src.PythonCodes.SQL.MsSQL.sql_handlers.QueryMsSQL.QueryMsSQL(c, m)
        #---------------------------------------------------------------------------
        # Getting into the DataBase
        #---------------------------------------------------------------------------
        m.printMesg("Connecting to the Database...")

        m.printMesgAddStr(" Database server           --->: ", c.getGreen(), c.getDatabase_server())
        m.printMesgAddStr(" Database name             --->: ", c.getYellow(), c.getDatabase_name())
        m.printMesgAddStr(" Database schema           --->: ", c.getMagenta(), c.getDatabase_schema())
        m.printMesgAddStr(" Database port             --->: ", c.getCyan(), c.getDatabase_port())
        m.printMesgAddStr(" SQL dir                   --->: ", c.getGreen(), c.getSql_dir())
        m.printMesgAddStr(" SQL full path dir         --->: ", c.getMagenta(), c.getSql_fullPath_dir())
        m.printMesgAddStr(" Connecting                --->: ", c.getCyan(),"...")
        rc, cnxn = queryMsSQL.open_connection_database_PostGreSQL()
        if rc != c.get_RC_SUCCESS(): exit(c.get_RC_FAIL())
        #---------------------------------------------------------------------------
        # Creating the Cursor
        #---------------------------------------------------------------------------
        m.printMesg("Creating the cursor")
        rc, cursor = queryMsSQL.create_cursor(cnxn)
        if rc != c.get_RC_SUCCESS(): exit(c.get_RC_FAIL())
        c.setQuery_cursor(cursor=cursor)

        # TODO: continue from here Normally this is in a while loop


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
                                                                        table_list[i], silent=True)
            table_count_list.append(table_count)
        print("-------------------------------------------------------------------")
        #---------------------------------------------------------------------------
        # Instantiating the JSon file Launcher Class to get methods
        #---------------------------------------------------------------------------
        ext_json = ".json"
        json_table_file = "current_AllTableCount_rec"+ext_json
        json_write_path = TBLECNTS_DIR

        jsonCreator = src.PythonCodes.utils.JSonCreator.JSonCreator(c, m)
        #---------------------------------------------------------------------------
        # Creating the table count for the json file
        #---------------------------------------------------------------------------
        jsonCreator.setJSon_Table_filename(json_table_file)
        jsonCreator.setJSon_Table_write_path(json_write_path)
        rc = jsonCreator.create_TableCount_JSon_file(table_list=table_list,
                                                     table_count_list=table_count_list)
        # --------------------------------------------------------------------------
        # Instantiating the JSon file Launcher Class to get methods
        # --------------------------------------------------------------------------
        jsonSanner = src.PythonCodes.utils.JSonScanner.JSonScanner(c, m)
        # --------------------------------------------------------------------------
        # Creating the table count for the json file
        #---------------------------------------------------------------------------
        #json_table_file = TABLE_CNTS_DIR + os.path.sep+c.getCurrent_AllTableCount_json_file()
        json_table_file = c.getJSon_TableCounts_Dir() + os.path.sep + c.getCurrent_AllTableCount_json_file()
        #print(json_table_file)
        #---------------------------------------------------------------------------
        # Reading back the just created JSon file
        #---------------------------------------------------------------------------
        rc, json_data_table_count = jsonSanner.read_TableCount_JSon_file(json_table_file)
        #---------------------------------------------------------------------------
        # [Scanning] sleeping time between scans
        #---------------------------------------------------------------------------
        m.printMesg("Sleeping between each database scanning")
        #table_length = json_data_table_count + 1
        for i in range(len(json_data_table_count[:])):
            for value in json_data_table_count[i]:
                rc, tb_count = jsonSanner.get_TableCount_FromTable(json_data=json_data_table_count, table=value, silent=False)
            # Counted the number of tables
            m.printMesgAddStr(" Counted number of tables  --->: ", c.getMagenta(), len(json_data_table_count[i]))
        #---------------------------------------------------------------------------
        # [Creating] Creating the insertors and launcher bats for the SQL insertes
        #---------------------------------------------------------------------------
        # TODO: here implement the ctfSQLUpdaterInsertors replacement
        #Here we need to insert at the nth+1 entry
        rc, sql_spectral_data_FileList = mssqlLauncher.PostgreSQL_UpdaterInsertors(tb='spectral_data',
                                                                                   jsonSanner=jsonSanner, json_data_table_count=json_data_table_count,
                                                                                   queryMsSQL=queryMsSQL, cursor=cursor,
                                                                                   Json_Mol_FileList=JsonMoleculeFileList[:],
                                                                                   jsonfile_content_dict_lst = jsonfile_content_dict_lst[:],
                                                                                   jsonfile_content_key=jsonfile_content_key_lst,
                                                                                   jsonfile_content_value=jsonfile_content_value_lst)

        bat_file = "LaunchSQL_Insert_into_spectral_data.bat"
        # TODO: call methods here
    #---------------------------------------------------------------------------
    # [--export_db] Creating the insertors and launcher bats for the SQL insertes
    #---------------------------------------------------------------------------
    if args["--export_db"]:
        l.createExport_db()
        m.printMesgStr("Database to be exported       :", c.getYellow(), c.getExport_db())
        database_handler = src.PythonCodes.utils.DataBaseHandler.DataBaseHandler(c, m, db_action="export_db")
        # TODO: call methods here
    #---------------------------------------------------------------------------
    # [--import_external_db] Creating the insertors and launcher bats for the SQL insertes
    #---------------------------------------------------------------------------
    l.createDatabase_origin_db()
    m.printMesgStr("Database origin               :", c.getYellow(), c.getDatabase_origin())

    if args["--import_external_db"]:
        l.createImport_External_db()
        if args["--import_db_set"] == 'single':
            l.createImport_db_set()

            m.printMesgStr("Database to be imported       :", c.getYellow(), c.getImport_External_db())
            database_handler = src.PythonCodes.utils.DataBaseHandler.DataBaseHandler(c, m, db_action="import_external_db")

            if args['--database_origin'] == 'FragHub' or args['--database_origin'] == 'GNPS':
                rc = database_handler.read_external_FragHub_database_files(c.getDatabase_file())
            elif args['--database_origin'] == 'LC-MS':
                rc = database_handler.read_external_Json_LCMS_database_files(c.getDatabase_file(), c.getDatabase_origin())
            elif args['--database_origin'] == 'MoNA':
                rc = database_handler.read_external_Json_MoNA_database_files(c.getDatabase_file())
            else:
                rc = c.get_RC_FAIL()
                m.printMesgStr(   "Database origin not specified :", c.getRed(), c.getDatabase_origin())
                m.printMesgAddStr(" Ex:                           :", c.getYellow(), " --database_origin=LC-MS")
                print("Return code: ", rc)
                exit(rc)
            # [end-if] statement over the args
            rc, molecule_number_lst = database_handler.create_molecule_dictionary()
            rc = database_handler.create_molecule_jsonfiles_from_molecules_dict()
            #---------------------------------------------------------------------------
            # Create the SQL scripts and populate those in the project folder for
            #---------------------------------------------------------------------------
            mssqlLauncher = src.PythonCodes.SQL.MsSQL.sql_handlers.MsSQLLauncher.MsSQLLauncher(c, m)
            rc, list_len, JsonMoleculeFileList = mssqlLauncher.getListOfJsonFiles_MsSQLLauncher(mssqlLauncher.jsontargetfile)
            rc, jsonfile_content_dict_lst, jsonfile_content_key_lst, jsonfile_content_value_lst = mssqlLauncher.ReadinJson_Molecules_File_MsSQLLauncher(JsonMoleculeFileList)
            #print("jsonfile_content_dict_lst[:] --->: ", jsonfile_content_dict_lst[:])
            #print(jsonfile_content_key_lst[:])
            #print(jsonfile_content_value_lst[:])
            #---------------------------------------------------------------------------
            # TODO: - here insert the the table scanning count,
            #       - move this code to the object constructor of the QueryMsSQL class
            #---------------------------------------------------------------------------
            m.printMesgStr("Check/Create      : ", c.getMagenta(), str(c.getDataProcInterCom()).strip() )
            if os.path.exists(str(c.getDataProcInterCom()).strip()):
                sys.path.append(str(c.getDataProcInterCom()))
                m.printMesgAddStr("  "+c.getDataProcInterCom()+" : ", c.getGreen(),"Exits nothing to do")
            else:
                m.printMesgAddStr("  "+str(c.getDataProcInterCom())+" : ", c.getRed(),"Does not Exits we will create it...")
                os.mkdir(c.getDataProcInterCom())
                m.printMesgAddStr("                   : ", c.getGreen(),"Done.")
                sys.path.append(c.getDataProcInterCom())

            m.printMesgStr("Check/Create      : ",c.getMagenta(), c.getJSon_TableCounts_Dir().strip() )
            if os.path.exists(c.getJSon_TableCounts_Dir().strip()):
                sys.path.append(c.getJSon_TableCounts_Dir())
                m.printMesgAddStr("  "+c.getJSon_TableCounts_Dir()+" : ", c.getGreen(),"Exits nothing to do")
            else:
                m.printMesgAddStr("  "+c.getJSon_TableCounts_Dir()+" : ", c.getRed(),"Does not Exits we will create it...")
                os.mkdir(c.getJSon_TableCounts_Dir())
                m.printMesgAddStr("               : ", c.getGreen(),"Done.")
                sys.path.append(c.getJSon_TableCounts_Dir())
            #---------------------------------------------------------------------------
            # Instantiating the Query class QueryMsSQL to get methods
            #---------------------------------------------------------------------------
            queryMsSQL = src.PythonCodes.SQL.MsSQL.sql_handlers.QueryMsSQL.QueryMsSQL(c, m)
            #---------------------------------------------------------------------------
            # Getting into the DataBase
            #---------------------------------------------------------------------------
            m.printMesg("Connecting to the Database...")

            m.printMesgAddStr(" Database server           --->: ", c.getGreen(), c.getDatabase_server())
            m.printMesgAddStr(" Database name             --->: ", c.getYellow(), c.getDatabase_name())
            m.printMesgAddStr(" Database schema           --->: ", c.getMagenta(), c.getDatabase_schema())
            m.printMesgAddStr(" Database port             --->: ", c.getCyan(), c.getDatabase_port())
            m.printMesgAddStr(" SQL dir                   --->: ", c.getGreen(), c.getSql_dir())
            m.printMesgAddStr(" SQL full path dir         --->: ", c.getMagenta(), c.getSql_fullPath_dir())
            m.printMesgAddStr(" Connecting                --->: ", c.getCyan(),"...")
            rc, cnxn = queryMsSQL.open_connection_database_PostGreSQL()
            if rc != c.get_RC_SUCCESS(): exit(c.get_RC_FAIL())
            #---------------------------------------------------------------------------
            # Creating the Cursor
            #---------------------------------------------------------------------------
            m.printMesg("Creating the cursor")
            rc, cursor = queryMsSQL.create_cursor(cnxn)
            if rc != c.get_RC_SUCCESS(): exit(c.get_RC_FAIL())
            c.setQuery_cursor(cursor=cursor)

            # TODO: continue from here Normally this is in a while loop

            m.printMesg("Getting the table list from the database")
            m.printMesgAddStr(" Database          : ",c.getYellow(), queryMsSQL.credentialsMsSQL.database_PostGreSQL)
            m.printMesgAddStr(" Schema            : ",c.getYellow(), queryMsSQL.credentialsMsSQL.schema_PostGreSQL)
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
            # Instantiating the JSon file Launcher Class to get methods
            #---------------------------------------------------------------------------
            ext_json = ".json"
            json_table_file = "current_AllTableCount_rec"+ext_json
            json_write_path = TBLECNTS_DIR

            jsonCreator = src.PythonCodes.utils.JSonCreator.JSonCreator(c, m)
            #---------------------------------------------------------------------------
            # Creating the table count for the json file
            #---------------------------------------------------------------------------
            jsonCreator.setJSon_Table_filename(json_table_file)
            jsonCreator.setJSon_Table_write_path(json_write_path)
            rc = jsonCreator.create_TableCount_JSon_file(table_list=table_list, table_count_list=table_count_list)
            # --------------------------------------------------------------------------
            # Instantiating the JSon file Launcher Class to get methods
            # --------------------------------------------------------------------------
            jsonSanner = src.PythonCodes.utils.JSonScanner.JSonScanner(c, m)
            # --------------------------------------------------------------------------
            # Creating the table count for the json file
            #---------------------------------------------------------------------------
            #json_table_file = TABLE_CNTS_DIR + os.path.sep+c.getCurrent_AllTableCount_json_file()
            json_table_file = c.getJSon_TableCounts_Dir() + os.path.sep + c.getCurrent_AllTableCount_json_file()
            #print(json_table_file)
            #---------------------------------------------------------------------------
            # Reading back the just created JSon file
            #---------------------------------------------------------------------------
            rc, json_data_table_count = jsonSanner.read_TableCount_JSon_file(json_table_file)
            #---------------------------------------------------------------------------
            # [Scanning] sleeping time between scans
            #---------------------------------------------------------------------------
            m.printMesg("Sleeping between each database scanning")
            #table_length = json_data_table_count + 1
            for i in range(len(json_data_table_count[:])):
                for value in json_data_table_count[i]:
                    rc, tb_count = jsonSanner.get_TableCount_FromTable(json_data=json_data_table_count, table=value, silent=False)
                # Counted the number of tables
                m.printMesgAddStr(" Counted number of tables  --->: ", c.getMagenta(), len(json_data_table_count[i]))
            #---------------------------------------------------------------------------
            # [Creating] Creating the insertors and launcher bats for the SQL insertes
            #---------------------------------------------------------------------------
            # TODO: Finish the all table database insertion
            #Here we need to insert at the nth+1 entry
            m.printMesg("Commencing database insertion ...")
            rc, sql_spectral_data_FileList = mssqlLauncher.PostgreSQL_UpdaterInsertors(tb='spectral_data',
                                                                                       jsonSanner=jsonSanner, json_data_table_count=json_data_table_count,
                                                                                       queryMsSQL=queryMsSQL, cursor=cursor,
                                                                                       Json_Mol_FileList=JsonMoleculeFileList[:],
                                                                                       jsonfile_content_dict_lst = jsonfile_content_dict_lst[:],
                                                                                       jsonfile_content_key=jsonfile_content_key_lst,
                                                                                       jsonfile_content_value=jsonfile_content_value_lst)

            bat_file = "LaunchSQL_Insert_into_spectral_data.bat"
            # TODO: call methods here
        #---------------------------------------------------------------------------
        # [--import_db_set] Creating the insertors and launcher bats for the SQL insertes
        #---------------------------------------------------------------------------
        if args["--import_db_set"] == 'set_in_dir':
            m.printMesgStr("Database to be imported       :", c.getYellow(), c.getImport_db_set())
            database_handler = src.PythonCodes.utils.DataBaseHandler.DataBaseHandler(c, m, db_action="import_external_db")
            # TODO: need to implement the batch import with multiple processing
    #---------------------------------------------------------------------------
    # [--export_external_db] Creating the insertors and launcher bats for the SQL insertes
    #---------------------------------------------------------------------------
    if args["--export_external_db"]:
        l.createExport_External_db()
        m.printMesgStr("Database to be exported       :", c.getYellow(), c.getExport_db())
        database_handler = src.PythonCodes.utils.DataBaseHandler.DataBaseHandler(c, m, db_action="export_external_db")
    #---------------------------------------------------------------------------
    # [--machine_learning] Starting the machien learning data analysis
    #---------------------------------------------------------------------------
    if args['--machine_learning']:
        l.createMachineLearning()
        m.printMesgStr("Machine learnig training      :", c.getYellow(), c.getMachineLearning())
        database_handler = src.PythonCodes.utils.DataBaseHandler.DataBaseHandler(c, m, db_action="train_dataset_from_db")

        # Instantiating the DeepL class for machine learning data analysis
        machine_learning = src.PythonCodes.src.MachineLearning.DeepL.DeepL(c, m, db_action="train_dataset_from_db")
        # first check the torch environment
        rc = machine_learning.test_torch_cuda()
        # initialising the CNN neural network
        # TODO: continue from here
        batch_size = 64
        num_classes = 10
        learning_rate = 0.001
        num_epochs = 20
        rc = machine_learning.convNeurNet.initialise_cnn(num_classes=num_classes)

    # ---------------------------------------------------------------------------
    # [Final] overall return code
    # ---------------------------------------------------------------------------
    # Final exit statements
    src.PythonCodes.DataManage_common.getFinalExit(c, m, rc)
    # ---------------------------------------------------------------------------
    # End of testing script
    # ---------------------------------------------------------------------------
