#!/usr/bin/env python3
"""!\file
   -- DataManage addon: (Python3 code) class for transforming the mgf files
                                      to database files
      \author Frederic Bonnet
      \date 4th of June 2024

      Universite de Perpignan March 2024, OBS.

Name:
---
Command_line: class MachineLearning for machine learning database files

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
"""
# System imports
import sys
import datetime
import os
import operator
import numpy
import json
import tqdm
import math
import re
import subprocess
import locale
# Deeplearning library to import
try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    rc = -1
    print(" torch is not installed on your system, verify or install")
    TORCH_AVAILABLE = False
    exit(rc)


if TORCH_AVAILABLE:
    import torch.nn
    import torchvision
    import torchvision.transforms

#application imports
import src.PythonCodes.DataManage_common
# Path extension
sys.path.append(os.path.join(os.getcwd(), '.'))
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append(os.path.join(os.getcwd(), '..', '..', '..', '..'))
sys.path.append(os.path.join(os.getcwd(), '..', '..', '..', '..','src','PythonCodes'))
sys.path.append(os.path.join(os.getcwd(), '..', '..', '..', '..','src','PythonCodes','utils'))
#Application imports
import src.PythonCodes.utils.JSonLauncher
import src.PythonCodes.utils.JSonScanner
import src.PythonCodes.utils.JSonCreator
import src.PythonCodes.utils.StopWatch
import src.PythonCodes.utils.progressBar
# Definiiton of the constructor
class CNN:
    #---------------------------------------------------------------------------
    # [Constructor] for the
    #---------------------------------------------------------------------------
    # Constructor
    def __init__(self, c, m, db_action="train_dataset_from_db"):
        __func__= sys._getframe().f_code.co_name
        # making
        super(CNN, self).__init__()
        # initialising the variables in the constructor
        self.rc = 0
        self.c = c
        self.m = m
        self.app_root = self.c.getApp_root()
        # Greeting message
        self.m.printMesgStr("Instantiating the class       :", self.c.getGreen(), "CNN")
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
        self.mz_intensity_relative_lst = []
        self.mz_intensity_relative_sorted_lst = []

        self.molecules_lst = []
        self.ith_molecules_lst = []
        self.molecules_dict = {}
        self.ith_molecules_dict = {}

        self.database_hreader_lst = []
        self.database_hreader_dict = {}

        self.molecule_number_lst = []

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
        if db_action == "import_external_db": self.database_full_path = self.c.getImport_External_db()
        if db_action == "export_external_db": self.database_full_path = self.c.getExport_External_db()

        if db_action == "train_dataset_from_db": self.database_full_path = self.c.getData_path()

        self.database_path = self.c.getDatabase_path()

        self.database_full_path_no_ext = self.c.getDatabase_full_path_no_ext()
        self.database_file = self.c.getDatabase_file()
        self.basename = self.c.getFile_basename()
        ext = ""
        if len(os.path.splitext(self.database_full_path)) > 1: ext = os.path.splitext(self.database_full_path)[1]

        self.file_csv = self.c.getCSV_file()
        self.file_asc = self.c.getPlot_asc()

        # print("basename: ", basename)
        if self.c.getDebug() == 1:
            self.m.printMesgAddStr("Database directory name    --->: ", self.c.getMagenta(), self.database_path)
            self.m.printMesgAddStr("database_full_path_no_ext  --->: ", self.c.getMagenta(), self.database_full_path_no_ext)
            self.m.printMesgAddStr("Database file              --->: ", self.c.getMagenta(), self.database_file)
            self.m.printMesgAddStr("Database name              --->: ", self.c.getMagenta(), self.c.getDatabase_name())
            self.m.printMesgAddStr("File basename              --->: ", self.c.getMagenta(), self.basename)
            self.m.printMesgAddStr("ext                        --->: ", self.c.getMagenta(), ext)
            self.m.printMesgAddStr("the csv file is then       --->: ", self.c.getMagenta(), self.file_csv)
            # printing the files:
            self.printFileNames()
        #-----------------------------------------------------------------------
        # [Initializor]
        #-----------------------------------------------------------------------
        # check if the gpu  has been set to tru
        self.m.printMesgAddStr("Use GPU true or false      --->: ", self.c.getMagenta(), self.c.getUse_gpu())
        if self.c.getUse_gpu():
            self.m.printMesgAddStr("Which device               --->: ", self.c.getCyan(), self.c.getSet_gpu())
        #-----------------------------------------------------------------------
        # [Constructor-end] end of the constructor
        #-----------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [Initialisor]
    #---------------------------------------------------------------------------
    def initialise_cnn(self, num_classes):
        rc = self.c.get_RC_SUCCESS()
        __func__ = sys._getframe().f_code.co_name
        self.m.printMesgStr("Initialising the CNN layers   :", self.c.getGreen(), __func__)

        self.conv_layer1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(1600, 128)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, num_classes)

        return rc
    #---------------------------------------------------------------------------
    # [Progressors]
    #---------------------------------------------------------------------------
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

    #---------------------------------------------------------------------------
    # [Checkers]
    #---------------------------------------------------------------------------
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
    #---------------------------------------------------------------------------
    # [Writters]
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [Reader]
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [Getters]
    #---------------------------------------------------------------------------

    def get_and_check_filename_path(self, database_file):
        rc = self.c.get_RC_SUCCESS()
        __func__= sys._getframe().f_code.co_name
        database_file_len = 0
        self.m.printMesgStr("get_and_check_filename_path (database_file): ", self.c.getGreen(), __func__)
        # ----------------------------------------------------------------------
        # Preamble of the readers
        # ----------------------------------------------------------------------
        self.m.printMesgAddStr("[file_DB ]:(database file) --->: ", self.c.getYellow(), database_file)
        self.m.printMesgAddStr("[file_DB ]:(Obj DB   file) --->: ", self.c.getYellow(), self.c.getDatabase_file())
        # ----------------------------------------------------------------------
        # Checking the existence and the coherence of the inpout file with object
        # ----------------------------------------------------------------------
        msg = "[file_DB]:      (checking) --->: " + self.c.getMagenta() + database_file + self.c.getBlue() + " <--> " + self.c.getMagenta() + self.database_file + ", "
        if database_file == self.c.getDatabase_file():
            self.m.printMesgAddStr(msg, self.c.getGreen(), "SUCCESS -->: are the same")
        else:
            rc = self.c.get_RC_WARNING()
            self.m.printMesgAddStr(msg, self.c.getRed(), "WARNING -->: are not the same")
        file_path = self.database_path + os.sep + self.c.getDatabase_file()
        if os.path.isfile(file_path):
            msg = file_path + self.c.getGreen() + " ---> Exists"
            self.m.printMesgAddStr("[file_DB ]:(database file) --->: ", self.c.getMagenta(), msg)
        else:
            msg = file_path + self.c.getRed() + " ---> Does not exists"
            rc = self.c.get_RC_FAIL()
            self.m.printMesgAddStr("[file_DB ]:(database file) --->: ", self.c.getMagenta(), msg)
            src.PythonCodes.DataManage_common.getFinalExit(self.c, self.m, rc)
        return file_path, rc
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
