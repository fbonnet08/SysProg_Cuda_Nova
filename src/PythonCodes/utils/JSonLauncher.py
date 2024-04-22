'''!\file
   -- LDMaP-APP addon: (Python3 code) to handle the sql scripting and
      generation for incoming data from the microscopes into a pool of
      file for a given project
      (code under construction and subject to constant changes).
      \author Frederic Bonnet
      \date 27th of July 2020

      Leiden University July 2020

Name:
---
JSonLauncher: module for launching scanning the file from the projects folder and
                    create a pool directory for a given project.

Description of classes:
---
This class generates and handles data coming from the microscopes for given 
project folder into a pool.

It uses class MRC_Data: Read, process, and write MRC volumes.

Requirements (system):
---
* sys
* os

Requirements (application):
---
* DataManage_common
* messageHandler
* StopWatch
* progressBar

'''
# System tools
import json
import sys
import os
from subprocess import PIPE, run
# appending the utils path
import utils.StopWatch

sys.path.append(os.path.join(os.getcwd(), '.'))
sys.path.append(os.path.join(os.getcwd(), '..'))
# application imports
from DataManage_common import *
#from messageHandler import *
#from StopWatch import *
#from progressBar import *
# ------------------------------------------------------------------------------
# Class to read star file from Relion
# file_type can be 'mrc' or 'ccp4' or 'imod'.
# ------------------------------------------------------------------------------
# ******************************************************************************
##\brief Python3 method.
# Class to read star file from Relion
# file_type can be 'mrc' or 'ccp4' or 'imod'.
class JSonLauncher:
    #***************************************************************************
    ##\brief Python3 method.
    #The constructor and Initialisation of the class for the PoolCreator.
    #Strip the header and generate a raw asc file to bootstarp on.
    #\param self       The object
    #\param c          Common class
    #\param m          Messenger class
    def __init__(self, c, m):
        # data_path, projectName, targetdir ,gainmrc, software):
        import os.path
        __func__= sys._getframe().f_code.co_name
        # first mapping the input to the object self
        self.c = c
        self.m = m
        self.app_root = self.c.getApp_root()        # app_root
        self.data_path = self.c.getData_path()      # data_path
        self.projectName = self.c.getProjectName()  # projectName
        self.targetdir = self.c.getTargetdir()      # targetdir
        self.software = self.c.getSoftware()        # software
        # Instantiating logfile mechanism
        logfile = self.c.getLogfileName()
        self.pool_targetdir = os.path.join(self.targetdir,
                                           self.projectName+"_Pool")
        self.c.setPool_Targetdir(self.pool_targetdir)

        m.printMesg("Instantiating the JSonLauncher class...")
        # ----------------------------------------------------------------------
        # Starting the timers for the constructor
        # ----------------------------------------------------------------------
        stopwatch = utils.StopWatch.createTimer()
        if c.getBench_cpu():
            # creating the timers and starting the stop watch...
            utils.StopWatch.StartTimer(stopwatch)

        self.m.printMesgAddStr(" app_root          : ",
                               self.c.getYellow(), self.app_root)
        self.m.printMesgAddStr(" data_path         : ",
                               self.c.getGreen(), str(self.data_path))
        self.m.printMesgAddStr(" projectName       : ",
                               self.c.getGreen(), str(self.projectName))
        self.m.printMesgAddStr(" targetdir         : ",
                               self.c.getMagenta(), str(self.targetdir))
        self.m.printMesgAddStr(" Software          : ",
                               self.c.getRed(), str(self.software))
        self.m.printMesgAddStr(" Pool path         : ",
                               self.c.getBlue(), str(self.pool_targetdir))
        # ----------------------------------------------------------------------
        # Setting up the file environment
        # ----------------------------------------------------------------------
        self.json_ext = ".json"
        self.jsonfilenamelist = "target_jsonfile.txt"
        self.jsondir = "JSonFiles_json"
        self.jsontargetfile = self.pool_targetdir+os.path.sep+"target_jsonfile.txt"
        # ----------------------------------------------------------------------
        # Genrating the json file target list
        # ----------------------------------------------------------------------
        rc, self.jsontargetfile = self.generate_JsonFileList(self.jsondir, self.json_ext)
        # ----------------------------------------------------------------------
        # Reporting time taken to instantiate and strip innitial star file
        # ----------------------------------------------------------------------
        if c.getBench_cpu():
            utils.StopWatch.StopTimer_secs(stopwatch)
            info = self.c.GetFrameInfo()
            self.m.printBenchMap("data_path",self.c.getRed(), self.data_path, info, __func__)
            self.m.printBenchTime_cpu("Read data_path file", self.c.getBlue(), stopwatch, info, __func__)
        # ----------------------------------------------------------------------
        # end of construtor __init__(self, path, file_type)
        # ----------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # [Methods] for the class
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # [Generators]
    # --------------------------------------------------------------------------
    def generate_JsonFileList(self, jsondir, ext):
        __func__= sys._getframe().f_code.co_name
        rc = self.c.get_RC_SUCCESS()

        fileout = self.jsontargetfile

        ls_cmd = [jsondir+os.path.sep+"*_JsonFile"+ext, ">", fileout]
        CMD = ' '.join(ls_cmd)
        self.m.printMesgStr("CMD: ls -1 ", self.c.get_B_Magenta(), CMD)
        os.system('ls -1 %s'%CMD)

        return rc, fileout
    #---------------------------------------------------------------------------
    # [Creator]
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [Reader]
    #---------------------------------------------------------------------------
    def getListOfJsonFiles(self, target_file):
        __func__= sys._getframe().f_code.co_name
        rc = self.c.get_RC_SUCCESS()
        FileList = []
        list_len = 0
        try:
            file = open(target_file, 'r')
            FileList = file.read().splitlines()
            for i in range(len(FileList)):
                list_len += 1
        except IOError:
            rc = self.c.get_RC_WARNING()
            self.m.printMesgStr("[target_file]: Could not open file:",
                           self.c.get_B_Red(), target_file)
            list_len = 0
            #exit(-1) #self.c.get_RC_FAIL())

        self.m.printCMesgValMesgVal(self.c.getBlue(),"Read from file: ",
                                    self.c.getCyan(),target_file,
                                    " ---> length list_len: ",
                                    self.c.getMagenta(),list_len)
        return rc, list_len, FileList
    #---------------------------------------------------------------------------
    # [JSon-Interpretors]
    #---------------------------------------------------------------------------
    def ReadingCTFJsonFile(self, filelist):
        __func__= sys._getframe().f_code.co_name
        rc = self.c.get_RC_SUCCESS()

        jsonfile_content = []
        msg = ""
        cnt = 0
        try:
            for i in range(len(filelist)):
                jfile = open(filelist[i],'r')
                jsondata = json.load(jfile)
                for idata in jsondata:
                    msg = "["+str(i)+"]"                              +" <---> "\
                          +self.c.getYellow() + idata['EntryID']      +" <---> "\
                          +self.c.getCyan() + str(idata['CTF_values'])+" <...> "\
                          +self.c.getYellow() + idata['CTF_SumCor2DiagCTFLive_png']
                    #self.m.printMesgAddStr(" Json file entry: ",
                    #                       self.c.getGreen(), msg)
                    jsonfile_content.append([idata['EntryID']
                                             , idata['CTF_values']
                                             , idata['Frame']
                                             , idata['DataMRC']
                                             , idata['SumCor2Live_mrc']
                                             , idata['SumCor2Live_txt']
                                             , idata['SumCor2FitCoeffLive_log']
                                             , idata['SumCor2FrameLive_log']
                                             , idata['SumCor2FullLive_log']
                                             , idata['SumCor2PatchLive_log']
                                             , idata['CTF_SumCor2Live_mrc']
                                             , idata['CTF_SumCor2Live_txt']
                                             , idata['CTF_SumCor2DiagAvrLive_png']
                                             , idata['CTF_SumCor2DiagAvrLive_txt']
                                             , idata['CTF_SumCor2DiagCTFLive_png']])
                cnt += 1
                jfile.close()
        except:
            rc = self.c.get_RC_WARNING()
            self.m.printMesgStr("[JSonFile("+str(cnt)+\
                                ")]: Could not open file:",
                                self.c.get_B_Red(), jfile)
            cnt = 0
        #End of the try catch    
        return rc, jsonfile_content
    #---------------------------------------------------------------------------
    # [Getters]
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [Setters]
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # Helper methods for the class
    #---------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# end of JSonLauncher module
#-------------------------------------------------------------------------------
