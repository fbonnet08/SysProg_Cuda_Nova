#!/usr/bin/env python3
'''!\file
   -- DataManage addon: (Python3 code) class for transforming the mgf files
                                      to database files
      \author Frederic Bonnet
      \date 19th of April 2024

      Universite de Perpignan March 2024, OBS.

Name:
---
Command_line: class MgfTransformer for transforming the mgf files to database files

Description of classes:
---
This class generates files

Requirements (system):
---
* sys
* datetime
* os
* csv
* scipy
* pandas
* seaborn
* matplotlib.pyplot
* matplotlib.dates
'''
# System imports
import sys
import datetime
import os
import operator
import numpy
import json

#application imports
import src.PythonCodes.DataManage_common

# Path extension
sys.path.append(os.path.join(os.getcwd(), '.'))
sys.path.append(os.path.join(os.getcwd(), '..'))
#Application imports
import src.PythonCodes.utils.JSonLauncher
import src.PythonCodes.utils.JSonScanner
import src.PythonCodes.utils.JSonCreator
# Definiiton of the constructor
class DataBaseHandler:
    #---------------------------------------------------------------------------
    # [Constructor] for the
    #---------------------------------------------------------------------------
    # Constructor
    def __init__(self, c, m, db_action="import_db"):
        __func__= sys._getframe().f_code.co_name
        self.rc = 0
        self.c = c
        self.m = m
        self.app_root = self.c.getApp_root()
        self.m.printMesgStr("Instantiating the class       :", self.c.getGreen(), "DataBaseHandler")

        #gettign the csv file
        self.ext_asc = ".asc"
        self.ext_csv = ".csv"
        self.ext_mgf = ".mgf"
        self.ext_json = ".json"
        self.undr_scr = "_"
        self.dot = "."
        self.csv_len = 0
        self.csv_col = 0
        self.zerolead = 0
        self.rows = []
        # data base directory structure
        self.database_json_targetdir = "Json_DataBase"
        #initialising the lists
        self.indnx = []
        self.columns_lst = []
        self.date = []
        self.time = []
        self.upload = []
        self.download = []
        self.newnumber = []
        self.out_data = []
        self.m_on_Z = []
        self.relative = []
        self.intensity = []
        self.molecules_lst = []
        self.ith_molecules_lst = []
        self.molecule_number_lst = []
        self.molecules_dict = {}
        self.ith_molecules_dict = {}
        self.mz_intensity_relative_lst = []
        self.mz_intensity_relative_sorted_lst = []
        # spectrum details
        self.scan_number = 0
        self.Ret_time = 0.0

        self.pepmass = 355.070007324219
        self.charge = 1
        self.MSLevel = 2

        #Statistics variables
        self.sample_min      = 0.0
        self.sample_max      = 0.0
        self.sample_mean     = 0.0
        self.sample_variance = 0.0
        self.running_mean = []
        # some path iniitialisation
        self.targetdir = "./"
        # Getting the file structure in place

        self.import_db = self.c.getImport_db()
        self.export_db = self.c.getExport_db()

        self.file_asc = self.c.getPlot_asc()

        if db_action == "import_db": self.database_full_path = self.c.getImport_db()
        if db_action == "export_db": self.database_full_path = self.c.getExport_db()

        self.database_path = os.path.dirname(self.database_full_path)
        self.c.setDatabase_path(self.database_path)

        self.database_full_path_no_ext = os.path.splitext(self.database_full_path)[0]
        self.c.setDatabase_full_path_no_ext(self.database_full_path_no_ext)
        self.database_file = os.path.basename(self.database_full_path)
        self.c.setDatabase_file(self.database_file)
        self.basename = self.database_file.split('.')[0]
        self.c.setFile_basename(self.basename)
        ext = ""
        if len(os.path.splitext(self.database_full_path)) > 1: ext = os.path.splitext(self.database_full_path)[1]

        # print("basename: ", basename)
        self.m.printMesgAddStr("Database directory name    --->: ", self.c.getMagenta(), self.database_path)
        self.m.printMesgAddStr("database_full_path_no_ext  --->: ", self.c.getMagenta(), self.database_full_path_no_ext)
        self.m.printMesgAddStr("Database file              --->: ", self.c.getMagenta(), self.database_file)
        self.m.printMesgAddStr("File basename              --->: ", self.c.getMagenta(), self.basename)
        self.m.printMesgAddStr("ext                        --->: ", self.c.getMagenta(), ext)
        self.file_csv = self.basename+self.ext_csv
        self.c.setCSV_file(self.file_csv)
        self.file_asc = self.basename+self.ext_asc
        self.c.setPlot_asc(self.file_asc)
        self.m.printMesgAddStr("the csv file is then       --->: ", self.c.getMagenta(), self.file_csv)
        # printing the files:
        self.printFileNames()
        #-----------------------------------------------------------------------
        # [Constructor-end] end of the constructor
        #-----------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [Driver]
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [Plotters]
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [Handlers]
    #---------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # [Extractor] extract the zerolead from a given list or length
    # --------------------------------------------------------------------------
    def extract_ZeroLead(self, nrows):
        rc = self.c.get_RC_SUCCESS()
        __func__ = sys._getframe().f_code.co_name
        # ----------------------------------------------------------------------
        # constructing the leading zeros according to nboot
        # ----------------------------------------------------------------------
        zerolead = 0
        # counting the number of digits in self.nboot
        import math
        if nrows > 0:
            digits = int(math.log10(nrows)) + 1
        elif nrows == 0:
            digits = 1
        else:
            digits = int(math.log10(-nrows)) + 2
        zerolead = numpy.power(10, digits)
        # ----------------------------------------------------------------------
        # End of method return statement
        # ----------------------------------------------------------------------
        return rc, zerolead
    #---------------------------------------------------------------------------
    # [Creator]
    #---------------------------------------------------------------------------
    def create_molecule_dictionary(self):
        rc = self.c.get_RC_SUCCESS()
        __func__= sys._getframe().f_code.co_name
        self.m.printMesgStr("Extracting database content (database_file): ", self.c.getGreen(), __func__)

        # --------------------------------------------------------------
        # TODO: [start] insert the loop of the molecule i loop
        # start of loop over the molecule dictionary
        ith_dict_key = 'molecule '+str(0)
        rc, zerolead = self.extract_ZeroLead(len(self.molecules_lst[0][:]))
        self.ith_molecules_dict[self.molecules_lst[0][0]] = self.molecules_lst[0][0]
        #self.m.printMesgAddStr("[molecules_lst]: mol("+str(0).zfill(len(str(zerolead)))+"), length("+str(len(self.molecules_lst[0][:]))+") --->: ",
        #                       self.c.getYellow(), self.molecules_lst[0][0])
        cnt = 0
        for j in range(1, len(self.molecules_lst[0][:]) - 1):
            #self.m.printMesgAddStr("[molecules_lst]: mol("+str(j).zfill(len(str(zerolead)))+"), length("+str(len(self.molecules_lst[0][:]))+") --->: ",
            #                       self.c.getYellow(), self.molecules_lst[0][j])
            if '=' in self.molecules_lst[0][j]:
                key = str(self.molecules_lst[0][j]).split('=')[0]
                value = str(self.molecules_lst[0][j]).split('=')[1]
                self.ith_molecules_dict[key] = value
            elif ' ' in self.molecules_lst[0][j]:
                key = 'mz rel: '+str(cnt)
                value = str(self.molecules_lst[0][j]).strip()
                self.ith_molecules_dict[key] = value
                cnt += 1
            #if self.molecules_lst[0][j].isnumeric() == False: self.ith_molecules_dict[self.molecules_lst[0][j]] = self.molecules_lst[0][j]
        # [end-loop]
        self.m.printMesgAddStr("[molecules_lst]: mol("+str(len(self.molecules_lst[0][:]) - 1).zfill(len(str(zerolead)))+"), length("+str(len(self.molecules_lst[0][:]))+") --->: ",
                               self.c.getYellow(), self.molecules_lst[0][len(self.molecules_lst[0][:]) - 1])
        self.ith_molecules_dict[self.molecules_lst[0][len(self.molecules_lst[0][:]) - 1]] = self.molecules_lst[0][len(self.molecules_lst[0][:]) - 1]

        print('printing the ith molecule from the loop ... ')
        self.m.printMesgAddStr("self.ith_molecules_dict["+str(0)+"] ---> ", self.c.getGreen(), self.ith_molecules_dict)

        #creating the molecule dictionary
        self.molecules_dict[ith_dict_key] = self.ith_molecules_dict
        # [end-loop] for the molecule loop
        # TODO: [end] insert the loop of the molecule i loop
        # --------------------------------------------------------------
        print('printing the molecule dictionary ... ')
        self.m.printMesgAddStr("self.molecules_dict     ---> ", self.c.getMagenta(), self.molecules_dict)

        print('testing I can access the correct molecule in the dictionary ... ')
        self.m.printMesgAddStr("molecule0 in the self.molecules_dict ---> ", self.c.getYellow(), self.molecules_dict['molecule 0'])
        self.m.printMesgAddStr("molecule0 in the self.molecules_dict ---> ", self.c.getYellow(), self.molecules_dict['molecule 0']['PEPMASS'])

        print("printing the length of the dictionary ---> ", len(self.molecules_dict))
        cnt = 0
        for key, value in self.molecules_dict.items():
            self.molecule_number_lst.append(key)
            print("The keys of the dictionary self.molecules_dict["+str(cnt)+"] ---> ", key)
            cnt += 1

        self.m.printMesgAddStr("[molecule_number_lst]: molecules_dict(keys) --->: ", self.c.getCyan(), self.molecule_number_lst[:])

        return rc, self.molecule_number_lst[:]
    def create_molecule_jsonfiles_from_molecules_dict(self):
        rc = self.c.get_RC_SUCCESS()
        __func__= sys._getframe().f_code.co_name
        database_file_len = 0
        self.m.printMesgStr("Cretaing molecule jsonfiles from molecules_dict (database_file): ", self.c.getGreen(), __func__)

        rc, zerolead = self.extract_ZeroLead(len(self.molecules_lst[:]))

        print(" Database_path            ---> ", self.c.getDatabase_path())
        print(" the name of the database ---> ", self.c.getDatabase_file())
        print(" database file basename   ---> ", self.c.getFile_basename())
        mol_num = str(len(self.molecules_lst[:])).zfill(len(str(zerolead)))
        print(" Number of molecules      ---> ", mol_num)
        print(" database_json_targetdir  ---> ", self.database_json_targetdir)

        self.dir_json = os.path.join(self.c.getDatabase_path(), self.database_json_targetdir)
        print(" Targetdir                ---> ", self.dir_json)

        self.c.setTargetdir(self.dir_json)

        if os.path.exists(self.dir_json):
            msg = self.c.get_B_Green()+" Already exists, nothing to do, continuing... "
            self.m.printMesgAddStr(" Directory                 --->: ", self.c.getMagenta(), self.dir_json + msg)
        else:
            self.m.printMesgAddStr("Creating                   --->: ", self.c.getMagenta(), self.dir_json)
            os.mkdir(self.dir_json)

        filename_json = self.c.getTargetdir() + os.sep + \
                        str(self.c.getFile_basename()).replace(' ', self.undr_scr) + \
                        self.undr_scr +"PEPMASS-" + str(self.molecules_dict['molecule 0']['PEPMASS']) + \
                        self.undr_scr+"molecule" + mol_num + self.ext_json
        print(" The future file name --->: ", filename_json)


        return rc
    #---------------------------------------------------------------------------
    # [Writters]
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [Reader]
    #---------------------------------------------------------------------------
    def read_database_file(self, database_file):
        rc = self.c.get_RC_SUCCESS()
        __func__= sys._getframe().f_code.co_name
        database_file_len = 0
        self.m.printMesgStr("Extracting database content (database_file): ", self.c.getGreen(), __func__)
        self.m.printMesgAddStr("[file_DB ]:(database file) --->: ", self.c.getYellow(), database_file)
        self.m.printMesgAddStr("[file_DB ]:(Obj DB   file) --->: ", self.c.getYellow(), self.c.getDatabase_file())
        # ----------------------------------------------------------------------
        # Checking the existence and the coherence of the inpout file with object
        # ----------------------------------------------------------------------
        msg = "[file_DB]:      (checking) --->: "+self.c.getMagenta()+database_file+self.c.getBlue()+" <--> "+self.c.getMagenta()+self.database_file+", "
        if database_file == self.c.getDatabase_file():
            self.m.printMesgAddStr(msg, self.c.getGreen(), "SUCCESS -->: are the same")
        else:
            rc = self.c.get_RC_WARNING()
            self.m.printMesgAddStr(msg, self.c.getRed(), "WARNING -->: are not the same")

        file_path = self.database_path+os.sep+self.c.getDatabase_file()
        if os.path.isfile(file_path):
            msg = file_path + self.c.getGreen() + " ---> Exists"
            self.m.printMesgAddStr("[file_DB ]:(database file) --->: ", self.c.getMagenta(), msg)
        else:
            msg = file_path + self.c.getRed() + " ---> Does not exists"
            rc = self.c.get_RC_FAIL()
            self.m.printMesgAddStr("[file_DB ]:(database file) --->: ", self.c.getMagenta(), msg)
            src.PythonCodes.DataManage_common.getFinalExit(self.c, self.m, rc)
        # ----------------------------------------------------------------------
        # Starting the timers for the constructor
        # ----------------------------------------------------------------------
        start_key = "BEGIN IONS"
        end_key = "END IONS"
        trigger_key = "Scan#: "   # "PEPMASS"
        try:
            #print("file_path -=-- > ", file_path)
            file = open(file_path)
            lines = file.readlines()
            database_file_len = len(lines)
            n_molecules = 0
            for i in range(database_file_len):
                if start_key in lines[i].split('\n')[0]:
                    n_molecules += 1
                #print("lines["+str(i)+"]", lines[i].split('\n')[0])
                # [end-loop]
            self.m.printMesgAddStr("[file_DB ]:(database file) --->: ", self.c.getMagenta(), database_file_len)
            self.m.printMesgAddStr("[file_DB ]:  (n molecules) --->: ", self.c.getYellow(), n_molecules)
            appending = False
            cnt = 0
            ith_molecule = 0
            insert_msg = ""
            for i in range(database_file_len):
                if start_key in lines[i].split('\n')[0]:
                    ith_molecule += 1
                    appending = True
                if end_key in lines[i].split('\n')[0]:
                    appending = False
                    self.ith_molecules_lst.append(lines[i].split('\n')[0])
                    self.molecules_lst.append(self.ith_molecules_lst[:])
                    #self.m.printMesgAddStr("[molecules_lst]: mol("+str(ith_molecule)+"), line("+str(i)+") --->: ",
                    #                       self.c.getYellow(), self.ith_molecules_lst[:])
                    self.ith_molecules_lst.clear()
                elif appending:
                    insert_msg = (lines[i].split('\n')[0]).replace('\t', ' ')
                    self.ith_molecules_lst.append(insert_msg)
                # [end-if]
                cnt += 1
            # [end-loop]
            #self.m.printMesgAddStr("[molecules_lst]: last(ith) --->: ", self.c.getYellow(), self.ith_molecules_lst[:])
            #self.m.printMesgAddStr("[molecules_lst]:       (0) --->: ", self.c.getYellow(), self.molecules_lst[0])
            #self.m.printMesgAddStr("[molecules_lst]:      ("+str(len(self.molecules_lst)-1)+") --->: ",
            #                       self.c.getYellow(), self.molecules_lst[len(self.molecules_lst)-1])
            self.m.printMesgAddStr("[molecules_lst]:     (len) --->: ", self.c.getYellow(), len(self.molecules_lst[:]))

        except IOError:
            rc = self.c.get_RC_WARNING()
            print("[mgf_len]: Could not open file:", file_path, database_file_len)
            csv_len = 0
            print("Return code: ", self.c.get_RC_FAIL())
            exit(self.c.get_RC_FAIL())

        return rc
    #---------------------------------------------------------------------------
    # [Getters]
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [Printers]
    #---------------------------------------------------------------------------
    def printFileNames(self):
        rc = self.c.get_RC_SUCCESS()
        self.m.printMesgAddStr("[file_DB ]:(database file) --->: ", self.c.getYellow(), self.c.getDatabase_file())
        self.m.printMesgAddStr("[basename]:(file basename) --->: ", self.c.getCyan(), self.c.getFile_basename())
        self.m.printMesgAddStr("[file_asc]:     (asc file) --->: ", self.c.getRed(), self.c.getPlot_asc())
        self.m.printMesgAddStr("[file_csv]:     (csv file) --->: ", self.c.getCyan(), self.c.getCSV_file())
        return rc
#---------------------------------------------------------------------------
# end of DataBaseHandler
#---------------------------------------------------------------------------
