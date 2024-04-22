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
MsSQLLauncher: module for launching scanning the file from the projects folder and
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
import numpy as np
from subprocess import PIPE, run
# appending the utils path
import utils.StopWatch
import utils.JSonLauncher
sys.path.append(os.path.join(os.getcwd(), '.'))
sys.path.append(os.path.join(os.getcwd(), '..', '..', '..'))
sys.path.append(os.path.join(os.getcwd(), '..', '..', '..', 'utils'))
# application imports
from DataManage_common import *
#import utils.messageHandler
#from messageHandler import *
#from StopWatch import *
import utils.StopWatch
#from progressBar import *
import utils.progressBar
# ------------------------------------------------------------------------------
# Class to read star file from Relion
# file_type can be 'mrc' or 'ccp4' or 'imod'.
# ------------------------------------------------------------------------------
# ******************************************************************************
##\brief Python3 method.
# Class to read star file from Relion
# file_type can be 'mrc' or 'ccp4' or 'imod'.
class MsSQLLauncher:
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

        self.m.printMesg("Instantiating the MsSQLLauncher class...")
        # ----------------------------------------------------------------------
        # Starting the timers for the constructor
        # ----------------------------------------------------------------------
        if c.getBench_cpu():
            # creating the timers and starting the stop watch...
            stopwatch = utils.StopWatch.createTimer()
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
        self.ProjectsNameID = self.projectName.split('_')[0]
        self.ProjectNumber = self.projectName.split('_')[1]
        # ----------------------------------------------------------------------
        # Setting up the file environment
        # ----------------------------------------------------------------------
        self.json_ext = ".json"
        self.jsonfilenamelist = "target_jsonfile.txt"
        self.jsondir = "JSonFiles_json"
        self.jsontargetfile = self.pool_targetdir+os.path.sep + \
                              "target_jsonfile.txt"
        # ----------------------------------------------------------------------
        # Genrating the json file target list
        # ----------------------------------------------------------------------
        self.jsonLauncher = utils.JSonLauncher.JSonLauncher(c,m)
        # ----------------------------------------------------------------------
        # Genrating the json file target list
        # ----------------------------------------------------------------------
        rc, self.jsontargetfile = self.jsonLauncher.generate_JsonFileList(self.jsondir, self.json_ext)
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

    # --------------------------------------------------------------------------
    # [SQL-Updateor-Insertors]
    # --------------------------------------------------------------------------
    def ProjectsNameSQLUpdaterInsertors(self, projects_info_lst, sql_dir):
        __func__= sys._getframe().f_code.co_name
        rc = self.c.get_RC_SUCCESS()

        sql_fullPath_dir = self.pool_targetdir+os.path.sep+sql_dir
        self.m.printMesg("Project Details in question...")
        self.m.printMesgAddStr(" ProjectName      : ",
                               self.c.getCyan(), str(self.projectName))
        self.m.printMesgAddStr(" ProjectsNameID   : ",
                               self.c.getYellow(), str(self.ProjectsNameID))
        self.m.printMesgAddStr(" ProjectNumber    : ",
                               self.c.getGreen(), str(self.ProjectNumber))
        self.m.printMesgAddStr(" SQL full path dir: ",
                               self.c.getGreen(), sql_fullPath_dir)
        #-----------------------------------------------------------------------
        # Generating the zerolead for the sql file constriction
        #-----------------------------------------------------------------------
        rc, zerolead = self.generate_Zerolead(len(projects_info_lst))
        # -----------------------------------------------------------------------
        # constructing the leading zeros according to nboot
        # -----------------------------------------------------------------------
        database = "NeCENDatabase-Prod"
        table = "ProjectsNameTable"
        sqlProjectsNameFileList = []
        ext_json = ".json"
        ext_sql = ".sql"
        progressBar = utils.progressBar.ProgressBar() #ProgressBar()
        for iproj in range(len(projects_info_lst)):
            #print(user_info_lst[iuser])
            file_name = sql_fullPath_dir + os.path.sep + "ProjectsNameUpdatorInsertor" + \
                                  str(iproj).zfill(len(str(zerolead))) + ext_sql

            sqlProjectsNameFileList.append(file_name)
            file_sql = open(file_name, 'w')

            rc = self.projectsNameSSQLWritter(file_sql, iproj, projects_info_lst[iproj], database, table)

            progressBar.update(1, len(projects_info_lst))
            progressBar.printEv()
            file_sql.close()

        self.m.printLine()
        progressBar.resetprogressBar()
        # ----------------------------------------------------------------------
        # End of the method
        # ----------------------------------------------------------------------
        return rc, sqlProjectsNameFileList

    def usersSQLUpdaterInsertors(self, users_info_lst, sql_dir):
        __func__= sys._getframe().f_code.co_name
        rc = self.c.get_RC_SUCCESS()

        sql_fullPath_dir = self.pool_targetdir+os.path.sep+sql_dir
        self.m.printMesg("Project Details in question...")
        self.m.printMesgAddStr(" ProjectName      : ",
                               self.c.getCyan(), str(self.projectName))
        self.m.printMesgAddStr(" ProjectsNameID   : ",
                               self.c.getYellow(), str(self.ProjectsNameID))
        self.m.printMesgAddStr(" ProjectNumber    : ",
                               self.c.getGreen(), str(self.ProjectNumber))
        self.m.printMesgAddStr(" SQL full path dir: ",
                               self.c.getGreen(), sql_fullPath_dir)
        #-----------------------------------------------------------------------
        # Generating the zerolead for the sql file constriction
        #-----------------------------------------------------------------------
        rc, zerolead = self.generate_Zerolead(len(users_info_lst))
        # -----------------------------------------------------------------------
        # constructing the leading zeros according to nboot
        # -----------------------------------------------------------------------
        database = "NeCENDatabase-Prod"
        table = "UsersTable"
        sqlUsersFileList = []
        ext_json = ".json"
        ext_sql = ".sql"
        progressBar = utils.progressBar.ProgressBar()#ProgressBar()
        for iuser in range(len(users_info_lst)):
            #print(user_info_lst[iuser])
            file_name = sql_fullPath_dir + os.path.sep + "UserUpdatorInsertor" + \
                                  str(iuser).zfill(len(str(zerolead))) + ext_sql

            sqlUsersFileList.append(file_name)
            file_sql = open(file_name, 'w')
            rc = self.usersSSQLWritter(file_sql, iuser, users_info_lst[iuser], database, table)

            progressBar.update(1, len(users_info_lst))
            progressBar.printEv()
            file_sql.close()

        self.m.printLine()
        progressBar.resetprogressBar()
        # ----------------------------------------------------------------------
        # End of the method
        # ----------------------------------------------------------------------
        return rc, sqlUsersFileList

    def ctfSQLUpdaterInsertors(self, JsonCTFFileList, jsonfile_content, sql_dir, table_length):
        __func__= sys._getframe().f_code.co_name
        rc = self.c.get_RC_SUCCESS()

        sql_fullPath_dir = self.pool_targetdir+os.path.sep+sql_dir
        self.m.printMesg("Project Details in question...")
        self.m.printMesgAddStr(" ProjectName      : ",
                               self.c.getCyan(), str(self.projectName))
        self.m.printMesgAddStr(" ProjectsNameID   : ",
                               self.c.getYellow(), str(self.ProjectsNameID))
        self.m.printMesgAddStr(" ProjectNumber    : ",
                               self.c.getGreen(), str(self.ProjectNumber))
        self.m.printMesgAddStr(" SQL full path dir: ",
                               self.c.getGreen(), sql_fullPath_dir)
        #-----------------------------------------------------------------------
        # Generating the zerolead for the sql file constriction
        #-----------------------------------------------------------------------
        rc, zerolead = self.generate_Zerolead(len(JsonCTFFileList))
        # -----------------------------------------------------------------------
        # constructing the leading zeros according to nboot
        # -----------------------------------------------------------------------
        database = "NeCENDatabase-Prod"
        table = "CTFDetailsKrios1Table"
        sqlCTFFileList = []
        ext_json = ".json"
        ext_sql = ".sql"
        progressBar = utils.progressBar.ProgressBar()#ProgressBar()
        for ifile in range(len(JsonCTFFileList)):
            head, tail = os.path.split(JsonCTFFileList[ifile])
            sqlCTFFileList.append(sql_dir + os.path.sep + tail.split(ext_json)[0] + \
                                  str(ifile).zfill(len(str(zerolead))) + ext_sql)

            file_name = self.pool_targetdir + os.path.sep + sqlCTFFileList[ifile]
            file_sql = open(file_name, 'w')

            rc = self.ctfSSQLWritter(file_sql, jsonfile_content[ifile],
                                     database, table, table_length)

            progressBar.update(1, len(JsonCTFFileList))
            progressBar.printEv()

            file_sql.close()

        self.m.printLine()
        progressBar.resetprogressBar()

        return rc, sqlCTFFileList
    #---------------------------------------------------------------------------
    # [SQL-Writters]
    #---------------------------------------------------------------------------
    def projectsNameSSQLWritter(self, file_hdl, iproj, project_info, database, table):
        __func__= sys._getframe().f_code.co_name
        rc = self.c.get_RC_SUCCESS()
        #-----------------------------------------------------------------------
        # First create an empty row
        #-----------------------------------------------------------------------
        sqlUsersFileList_entries = []
        start_UProjectsNameINDX = 8
        file_hdl.write("USE ["+database+"]\n")
        file_hdl.write("GO\n")
        file_hdl.write("\n")
        file_hdl.write("INSERT INTO [dbo].["+table+"]\n")
        file_hdl.write("(ProjectsNameINDX, UsersID, ProjectsNameID, ProjectNumber, ScopeNameID, Operator,\n")
        file_hdl.write("ComplementInfo, DateMinStamp, DateMaxStamp, DataSize, AnaSize, Software, ProjType\n")
        file_hdl.write(")\n")
        file_hdl.write("VALUES\n")
        file_hdl.write("('"+str(start_UProjectsNameINDX+iproj)+"', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)\n")
        file_hdl.write(";\n")
        #-----------------------------------------------------------------------
        # Updating the row thta has just been created
        #-----------------------------------------------------------------------

        return rc

    def usersSSQLWritter(self, file_hdl, iuser, users_info, database, table):
        __func__= sys._getframe().f_code.co_name
        rc = self.c.get_RC_SUCCESS()
        # ----------------------------------------------------------------------
        # First create an empty row
        # ----------------------------------------------------------------------
        sqlUsersFileList_entries = []
        start_UsersID = 132
        file_hdl.write("USE ["+database+"]\n")
        file_hdl.write("GO\n")
        file_hdl.write("\n")
        file_hdl.write("INSERT INTO [dbo].["+table+"]\n")
        file_hdl.write("([UsersID], [Firstname], [LastName], [EmailAddress], [StartUserActivityDate], [StopUserActivityDate],\n")
        file_hdl.write(" [UserGroup], [Affiliation], [AffiliationSubsidy], [ActivityStatus], [ScreeningID], [DataPreProcID], [DataPostProcID]\n")
        file_hdl.write(")\n")

        file_hdl.write("VALUES\n")
        file_hdl.write("('"+str(start_UsersID+iuser)+"', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)\n")
        file_hdl.write(";\n")
        # ----------------------------------------------------------------------
        # Updating the row thta has just been created
        # ----------------------------------------------------------------------
        #login,lname,fname,email,phone,bcode,affiliation,unitlogin,mustchpwd,mustchbcode,active
        sqlUsersFileList_entries = users_info.split(',')
        #print("sqlUsersFileList_entries: ", sqlUsersFileList_entries)
        file_hdl.write("\n")
        file_hdl.write("\n")
        file_hdl.write("USE ["+database+"]\n")
        file_hdl.write("GO\n")
        file_hdl.write("\n")
        file_hdl.write("UPDATE [dbo].["+table+"]\n")
        file_hdl.write("SET\n")
        file_hdl.write("[UsersID] = '" + str(start_UsersID+iuser)+"',\n")
        file_hdl.write("[Firstname] = N'" +sqlUsersFileList_entries[2]+"',\n")
        file_hdl.write("[LastName] = N'" +sqlUsersFileList_entries[1]+"',\n")
        file_hdl.write("[EmailAddress] = N'" +sqlUsersFileList_entries[3]+"',\n")
        file_hdl.write("[StartUserActivityDate] = "+'NULL'+",\n")   #N'"+''+"',\n")
        file_hdl.write("[StopUserActivityDate] = "+'NULL'+",\n")    #N'"+''+"',\n")

        group_afiliat = sqlUsersFileList_entries[7].split('-')
        if len(group_afiliat[:]) == 0:
            file_hdl.write("[UserGroup] = N'',\n")
            file_hdl.write("[Affiliation] = N'',\n")
        if len(group_afiliat[:]) == 1:
            file_hdl.write("[UserGroup] = N'"+ str(group_afiliat[0]).strip()+"',\n")
            if str(group_afiliat[0]).strip() == "NeCEN":
                file_hdl.write("[Affiliation] = N'"+str(group_afiliat[0]).strip()+"',\n")
            else:
                file_hdl.write("[Affiliation] = N'',\n")
        elif len(group_afiliat[:]) == 2:
            file_hdl.write("[UserGroup] = N'" + sqlUsersFileList_entries[7].strip() + "',\n")
            if str(group_afiliat[1]).strip() == "LU":
                file_hdl.write("[Affiliation] = N'" + "Leiden University" + "',\n")
            elif str(group_afiliat[1]).strip() == "UU":
                file_hdl.write("[Affiliation] = N'" + "Utrecht University" + "',\n")
            elif str(group_afiliat[1]).strip() == "MU":
                file_hdl.write("[Affiliation] = N'" + "Maastricht University" + "',\n")
            elif str(group_afiliat[1]).strip() == "RUG":
                file_hdl.write("[Affiliation] = N'" + "University of Groningen" + "',\n")
            elif str(group_afiliat[1]).strip() == "NKI":
                file_hdl.write("[Affiliation] = N'" + "Netherlands Cancer Institute" + "',\n")
            elif str(group_afiliat[1]).strip() == "LUMC":
                file_hdl.write("[Affiliation] = N'" + "Leiden University Medical Center" + "',\n")
            elif str(group_afiliat[1]).strip() == "TUD":
                file_hdl.write("[Affiliation] = N'" + "Delft University of Technology" + "',\n")
            else:
                file_hdl.write("[Affiliation] = N'" + str(group_afiliat[1]).strip() + "',\n")

        file_hdl.write("[AffiliationSubsidy] = N'"+sqlUsersFileList_entries[6]+"',\n")
        if sqlUsersFileList_entries[len(sqlUsersFileList_entries)-1] == 'false':
            file_hdl.write("[ActivityStatus] = N'" +'Non Active' +"',\n")
        if sqlUsersFileList_entries[len(sqlUsersFileList_entries)-1] == 'true':
            file_hdl.write("[ActivityStatus] = N'" +'Active' +"',\n")
        file_hdl.write("[ScreeningID] = N'"+str(start_UsersID+iuser)+'001'+"',\n")
        file_hdl.write("[DataPreProcID] = N'"+str(start_UsersID+iuser)+'001'+"',\n")
        file_hdl.write("[DataPostProcID] = N'"+str(start_UsersID+iuser)+'001'+"'\n")
        file_hdl.write("WHERE\n")
        file_hdl.write("UsersID = "+str(start_UsersID+iuser)+"\n")  #3
        file_hdl.write("GO\n")

        return rc

    def usersSQLBatPusher(self, sqlUsersFileList, sql_dir):
        __func__= sys._getframe().f_code.co_name
        rc = self.c.get_RC_SUCCESS()
        # ----------------------------------------------------------------------
        # First create an empty row
        # ----------------------------------------------------------------------

        return rc

    def ctfSSQLWritter(self, file_hdl, i_jsonfile_content, database, table, table_length):
        __func__= sys._getframe().f_code.co_name
        rc = self.c.get_RC_SUCCESS()
        # ----------------------------------------------------------------------
        # First create an empty row
        # ----------------------------------------------------------------------
        #table_legnth = 59
        start_CTFDetailsINDX = table_length - 1 # this make sure of c++ format
        insertion_point = start_CTFDetailsINDX + int(i_jsonfile_content[0])
        file_hdl.write("USE ["+database+"]\n")
        file_hdl.write("GO\n")
        file_hdl.write("\n")
        file_hdl.write("INSERT INTO [dbo].["+table+"]\n")
        file_hdl.write("(CTFDetailsINDX, CTFDetailsID, ProjectsNameID, ProjectNumber, MicrographNumber, Defocus1, Defocus2, Astigmatism, PhaseShift,\n")
        file_hdl.write("CrossCorrelation, Spacing, Frame, DataMRC, DataJPG, DataXML, CTF_SumCor2Live_mrc, CTF_SumCor2Live_txt, CTF_SumCor2DiagAvrLive_png,\n")
        file_hdl.write("CTF_SumCor2DiagAvrLive_txt, CTF_SumCor2DiagCTFLive_png\n")
        file_hdl.write(")\n")
        file_hdl.write("VALUES\n")
        file_hdl.write("("+str(insertion_point)+", NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)\n")
        file_hdl.write(";\n")
        # ----------------------------------------------------------------------
        # Updating the row thta has just been created
        # ----------------------------------------------------------------------
        file_hdl.write("\n")
        file_hdl.write("\n")
        file_hdl.write("USE ["+database+"]\n")
        file_hdl.write("GO\n")
        file_hdl.write("\n")
        file_hdl.write("UPDATE [dbo].["+table+"]\n")
        file_hdl.write("SET\n")
        file_hdl.write("[CTFDetailsINDX] = "+str(insertion_point)+",\n")#3,
        file_hdl.write("[CTFDetailsID] = "+str(i_jsonfile_content[0])+",\n")#3,
        file_hdl.write("[ProjectsNameID] = N'"+self.ProjectsNameID+"',\n")#20210401
        file_hdl.write("[ProjectNumber] = N'"+self.ProjectNumber+"',\n")#2101-278
        # ----------------------------------------------------------------------
        # CTF VAlues
        # ----------------------------------------------------------------------
        # Output from CTFFind version 4.1.8, run on 2021-05-28 22:21:19
        # Input file: ././MotionCorr2_mrc/FoilHole_352185_Data_351988_351990_20210401_161307_fractions_SumCor2.mrc ; Number of micrographs: 1
        # Pixel size: 1.097 Angstroms ; acceleration voltage: 300.0 keV ; spherical aberration: 2.70 mm ; amplitude contrast: 0.07
        # Box size: 512 pixels ; min. res.: 18.0 Angstroms ; max. res.: 4.0 Angstroms ; min. def.: 3100.0 um; max. def. 6900.0 um
        # Columns:
        # #1 - micrograph number;
        # #2 - defocus 1 [Angstroms];
        # #3 - defocus 2;
        # #4 - azimuth of astigmatism;
        # #5 - additional phase shift [radians];
        # #6 - cross correlation;
        # #7 - spacing (in Angstroms) up to which CTF rings were fit successfully
        #1.000000 5779.293457 5638.637207 27.500006 0.000000 -0.115308 6.642923
        ctf_values_lst = i_jsonfile_content[1].split(',')
        file_hdl.write("[MicrographNumber] = "+str(ctf_values_lst[0])+",\n")
        file_hdl.write("[Defocus1] = "+str(ctf_values_lst[1])+",\n")
        file_hdl.write("[Defocus2] = "   +str(ctf_values_lst[2])+",\n")
        file_hdl.write("[Astigmatism] = "    +str(ctf_values_lst[3])+",\n")
        file_hdl.write("[PhaseShift] = " + str(ctf_values_lst[4].strip()) + ",\n")
        file_hdl.write("[CrossCorrelation] = " + str(ctf_values_lst[5].strip()) + ",\n")
        if (str(ctf_values_lst[6].strip()) == "inf"):
            file_hdl.write("[Spacing] = " + "NULL" + ",\n")
        else:
            file_hdl.write("[Spacing] = " + str(ctf_values_lst[6]) + ",\n")
        # ----------------------------------------------------------------------
        # Files Entries
        # ----------------------------------------------------------------------
        file_hdl.write("[Frame] = N'"                     +str(i_jsonfile_content[2 ])+"',\n")#FoilHole_o_Data_ppp_fractions.mrc',
        file_hdl.write("[DataMRC] = N'"                   +str(i_jsonfile_content[3 ])+"',\n")#FoilHole_o_Data_jjj.mrc',
        file_hdl.write("[DataJPG] = N'"                   +str("")                    +"',\n")#FoilHole_o_Data_jjj.jpg',
        file_hdl.write("[DataXML] = N'"                   +str("")                    +"',\n")#FoilHole_o_Data_jjj.xml',
        file_hdl.write("[CTF_SumCor2Live_mrc] = N'"       +str(i_jsonfile_content[10])+"',\n")#CTFFind_mrc/FoilHole_o_Data_ppp_fractions_SumCor2_diag.mrc',
        file_hdl.write("[CTF_SumCor2Live_txt] = N'"       +str(i_jsonfile_content[11])+"',\n")#CTFFind_txt/FoilHole_o_Data_ppp_fractions_SumCor2_diag.txt',
        file_hdl.write("[CTF_SumCor2DiagAvrLive_png] = N'"+str(i_jsonfile_content[12])+"',\n")#CTFFind_png/FoilHole_o_Data_ppp_fractions_SumCor2_diag_avrot.png',
        file_hdl.write("[CTF_SumCor2DiagAvrLive_txt] = N'"+str(i_jsonfile_content[13])+"',\n")#CTFFind_avr/FoilHole_o_Data_ppp_fractions_SumCor2_diag_avrot.txt',
        file_hdl.write("[CTF_SumCor2DiagCTFLive_png] = N'"+str(i_jsonfile_content[14])+"' \n")#CTFFind_fit/FoilHole_o_Data_ppp_fractions_SumCor2_diag_CTF_fit_Graph.png'
        file_hdl.write("WHERE\n")
        file_hdl.write("CTFDetailsINDX = "+str(insertion_point)+"\n")  #3
        file_hdl.write("GO\n")
        # ----------------------------------------------------------------------
        # Debugging and checking code
        # ----------------------------------------------------------------------
        # file_hdl.write("\n")
        # file_hdl.write(str(i_jsonfile_content)+"\n")
        return rc
    #---------------------------------------------------------------------------
    # [Bat-Writters]
    #---------------------------------------------------------------------------
    def SQLBatPusher(self, server, sqlFileList, sql_dir, bat_file):
        __func__ = sys._getframe().f_code.co_name
        rc = self.c.get_RC_SUCCESS()

        sql_fullPath_dir = self.pool_targetdir + os.path.sep + sql_dir
        self.m.printMesg("Project Details in question...")
        self.m.printMesgAddStr(" ProjectName      : ",
                               self.c.getCyan(), str(self.projectName))
        self.m.printMesgAddStr(" ProjectsNameID   : ",
                               self.c.getYellow(), str(self.ProjectsNameID))
        self.m.printMesgAddStr(" ProjectNumber    : ",
                               self.c.getGreen(), str(self.ProjectNumber))
        self.m.printMesgAddStr(" SQL full path dir: ",
                               self.c.getMagenta(), sql_fullPath_dir)
        self.m.printMesgAddStr(" Server name      : ",
                               self.c.getYellow(), server)
        #-----------------------------------------------------------------------
        # Generating the zerolead for the sql file constriction
        #-----------------------------------------------------------------------
        rc, zerolead = self.generate_Zerolead(len(sqlFileList))
        #-----------------------------------------------------------------------
        # constructing the leading zeros according to nboot
        #-----------------------------------------------------------------------
        database = "NeCENDatabase-Prod"
        file_name = self.pool_targetdir + os.path.sep + sql_dir+os.path.sep+bat_file
        #os.chmod(file_name,777)
        file_bat = open(file_name, 'w')
        rc = self.ctfSQLBatWritter_header(file_bat, database, server)
        progressBar = utils.progressBar.ProgressBar()#ProgressBar()
        for ifile in range(len(sqlFileList)):
            head, tail = os.path.split(sqlFileList[ifile])

            rc = self.ctfSQLBatWritter(file_bat, tail)

            progressBar.update(1, len(sqlFileList))
            progressBar.printEv()

        self.m.printLine()
        progressBar.resetprogressBar()
        file_bat.close()

        os.system('chmod a+x %s'%file_name)
        return rc

    def ctfSQLBatWritter(self, file_bat, i_sqlCTFFileList):
        __func__= sys._getframe().f_code.co_name
        rc = self.c.get_RC_SUCCESS()

        file_bat.write("sqlcmd -S %SERVER% -i "+i_sqlCTFFileList+"\n")

        return rc

    def ctfSQLBatWritter_header(self, file_bat, database, server):
        __func__= sys._getframe().f_code.co_name
        rc = self.c.get_RC_SUCCESS()

        file_bat.write("@echo off\n")
        file_bat.write("\n")
        file_bat.write("echo Launching the SQL pushing script...\n")
        file_bat.write("echo Moving the executable directory\n")
        file_bat.write("\n")
        file_bat.write("REM set DIR=\"C:\cygwin64\home\Frederic\Leiden\SourceCode\DataManage_project\SQL\MsSQL\DatabaseCode\Production-Scripts\"\n")
        file_bat.write("REM echo The current directory is %CD%\n")
        file_bat.write("\n")
        file_bat.write("set OUTPUT_FILE=\"sql_InsertUpdate_out.log\"\n")
        file_bat.write("set SERVER=\""+server+"\"\n")
        file_bat.write("set DB=\""+database+"\"\n")
        file_bat.write("set USER=\"quarky\"\n")
        file_bat.write("set PASSWORD=\"*********\"\n")
        file_bat.write("\n")

        return rc
    #---------------------------------------------------------------------------
    # [Generators]
    #---------------------------------------------------------------------------
    def generate_Zerolead(self, lst_len):
        __func__= sys._getframe().f_code.co_name
        rc = self.c.get_RC_SUCCESS()
        #-----------------------------------------------------------------------
        # constructing the leading zeros according to nboot
        #-----------------------------------------------------------------------
        #counting the number of digits in lst_len
        import math
        if lst_len > 0:
            digits = int(math.log10(lst_len))+1
        elif lst_len == 0:
            digits = 1
        else:
            digits = int(math.log10(-lst_len))+2

        zerolead = np.power(10,digits)
        
        return rc, zerolead
    #---------------------------------------------------------------------------
    # [Reader]
    #---------------------------------------------------------------------------
    def getListOfJsonFiles_MsSQLLauncher(self, target_file):
        __func__= sys._getframe().f_code.co_name
        rc = self.c.get_RC_SUCCESS()
        FileList = []

        rc, list_len, FileList = self.jsonLauncher.getListOfJsonFiles(target_file)
        
        return rc, list_len, FileList
    #---------------------------------------------------------------------------
    # [JSon-Interpretors]
    #---------------------------------------------------------------------------
    def ReadinJsonFile_MsSQLLauncher(self, filelist):
        __func__= sys._getframe().f_code.co_name
        rc = self.c.get_RC_SUCCESS()

        jsonfile_content = []
        rc, jsonfile_content = self.jsonLauncher.ReadingCTFJsonFile(filelist)
        
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
# end of PoolCreator_Fileio module
#-------------------------------------------------------------------------------
