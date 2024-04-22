'''!\file
   -- LDMaP-APP addon: (Python3 code) to query the database
      \author Frederic Bonnet
      \date 20th of June 2021

      Leiden University June 2021

Name:
---
QueryMsSQL: module for quering a data base

Description of classes:
---
This class queries the data base.

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
import pyodbc
# appending the utils path
sys.path.append(os.path.join(os.getcwd(), '.'))
sys.path.append(os.path.join(os.getcwd(), '..','..','..'))
sys.path.append(os.path.join(os.getcwd(), '..','..','..','utils'))
# application imports
from DataManage_common import *
import utils.StopWatch
import SQL.MsSQL.sql_scripts.CredentialsMsSQL
# ------------------------------------------------------------------------------
# Class to read star file from Relion
# file_type can be 'mrc' or 'ccp4' or 'imod'.
# ------------------------------------------------------------------------------
# ******************************************************************************
##\brief Python3 method.
# Class to read star file from Relion
# file_type can be 'mrc' or 'ccp4' or 'imod'.
class QueryMsSQL:
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

        m.printMesg("Instantiating the QueryMsSQL class...")
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
        #-----------------------------------------------------------------------
        # Setting up the file environment
        #-----------------------------------------------------------------------
        self.table_list = []
        self.full_table_list = []
        self.columNames_table_list = []
        self.rowth_lst = []
        #-----------------------------------------------------------------------
        # Instantiating the Credentials class CredentialsMsSQL to get methods
        #-----------------------------------------------------------------------
        self.credentialsMsSQL = SQL.MsSQL.sql_scripts.CredentialsMsSQL.CredentialsMsSQL(c, m)
        self.server = self.credentialsMsSQL.server
        self.database = self.credentialsMsSQL.database
        self.username = self.credentialsMsSQL.username
        self.password = self.credentialsMsSQL.password
        #-----------------------------------------------------------------------
        # Reporting time taken to instantiate and strip innitial star file
        #-----------------------------------------------------------------------
        if c.getBench_cpu():
            utils.StopWatch.StopTimer_secs(stopwatch)
            info = self.c.GetFrameInfo()
            self.m.printBenchMap("data_path",self.c.getRed(), self.data_path, info, __func__)
            self.m.printBenchTime_cpu("Read data_path file", self.c.getBlue(), stopwatch, info, __func__)
        #-----------------------------------------------------------------------
        # end of construtor __init__(self, path, file_type)
        #-----------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [Methods] for the class
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [Connector]
    #---------------------------------------------------------------------------
    def open_connection_database(self):
        rc = self.c.get_RC_SUCCESS()
        self.cnxn = pyodbc.connect('DRIVER={SQL Server};'
                                   'SERVER='+self.server+';'
                                   'DATABASE='+self.database+';'
                                   'UID='+self.username+';'
                                   'PWD='+ self.password+';'
                                   'Trusted_Connection=yes;')
        return rc, self.cnxn
    #---------------------------------------------------------------------------
    # [Query-SQL]
    #---------------------------------------------------------------------------
    def query_GetFullTable_list(self, cursor):
        rc = self.c.get_RC_SUCCESS()
        self.m.printMesg("Query ---> GetFullTable_list...")
        cursor.execute('USE [NeCENDatabase-Prod]')
        cursor.execute('SELECT * FROM INFORMATION_SCHEMA.TABLES')
        cnt = 0
        for row in cursor:
            if cnt > 1: self.full_table_list.append(row)
            cnt += 1
        return rc, self.full_table_list

    def query_GetTableCount(self, cursor, database, table):
        rc = self.c.get_RC_SUCCESS()
        cursor.execute('SELECT count(*) FROM ['+database+'].[dbo].['+table+']')
        table_count = 0
        for row in cursor:
            self.m.printMesgAddStr("["+database+"].[dbo].[{0:<22}".format(table) + "] ---> count: ",
                                   self.c.getCyan(), row)
        table_count = row[0]
        return rc, table_count

    def query_AllFromTable(self, cursor, database, table):
        rc = self.c.get_RC_SUCCESS()
        self.m.printMesg("Query ---> AllFromTable...")
        cursor.execute('SELECT * FROM ['+database+'].[dbo].['+table+']')
        for row in cursor:
            #print(row)
            self.m.printMesgAddStr("["+database+"].[dbo].[{}".format(table) + "] ---> row: ",
                                   self.c.getCyan(), row)
        return rc

    def query_SelectFromTable_ColList(self, cursor, database, list, table):
        rc = self.c.get_RC_SUCCESS()
        self.m.printMesg("Query ---> SelectFromTable_ColList...")
        cursor.execute('SELECT '+list+' FROM ['+database+'].[dbo].['+table+']')
        for row in cursor: #print(row)
            self.m.printMesgAddStr("["+database+"].[dbo].[{}".format(table) + "] ---> row: ",
                                   self.c.getCyan(), row)
        return rc

    def query_SelectFromTable_Where_Value(self, cursor, database, table, column, value ):
        rc = self.c.get_RC_SUCCESS()
        self.m.printMesg("Query ---> SelectFromTable_Where_Value...")
        cursor.execute('SELECT * FROM ['+database+'].[dbo].['+table+'] where ['+column+']=\''+str(value)+'\'')
        for row in cursor:# print(row)
            self.m.printMesgAddStr("["+database+"].[dbo].[{}".format(table) + "] ---> row: ",
                                   self.c.getCyan(), row)
        return rc

    def query_SelectFromTable_ColList_Where_Value(self, cursor, database, list, table, column, value):
        rc = self.c.get_RC_SUCCESS()
        self.m.printMesg("Query ---> SelectFromTable_ColList_Where_Value...")
        cursor.execute('SELECT '+list+' FROM ['+database+'].[dbo].['+table+'] where ['+column+']=\''+str(value)+'\'')
        self.rowth_lst.clear()
        for row in cursor:
            self.m.printMesgAddStr("["+database+"].[dbo].[{}".format(table) + "] ---> row: ",
                                   self.c.getCyan(), row)
            self.rowth_lst.append(row)
        return rc, self.rowth_lst

    def query_GetColumNameFromTable(self, cursor, database, table):
        rc = self.c.get_RC_SUCCESS()
        self.columNames_table_list.clear()
        self.m.printMesg("Query ---> GetColumNameFromTable...")
        cursor.execute('SELECT * FROM INFORMATION_SCHEMA.COLUMNS where TABLE_NAME = \''+table+'\'')
        for row in cursor:
            self.m.printMesgAddStr("["+database+"].[dbo].[{}".format(table)+"] ---> column: ",
                                   self.c.getCyan(), row[3])
            self.columNames_table_list.append(row[3])
        return rc, self.columNames_table_list

    def query_GetRowFromTable(self, cursor, database, table, column, row_indx):
        rc = self.c.get_RC_SUCCESS()
        #cursor.execute('SELECT * FROM INFORMATION_SCHEMA.COLUMNS where TABLE_NAME = \''+table+'\'')
        cursor.execute('SELECT * FROM ['+database+'].[dbo].['+table+'] where ['+column+']=\''+str(row_indx)+'\'')
        rowth = ""
        for row in cursor: rowth = row #print(row)
        return rc, rowth
    #---------------------------------------------------------------------------
    # [Creator]
    #---------------------------------------------------------------------------
    def create_cursor(self, cnxn):
        rc = self.c.get_RC_SUCCESS()
        self.cursor = cnxn.cursor()
        return rc, self.cursor
    #---------------------------------------------------------------------------
    # [Generators]
    #---------------------------------------------------------------------------
    def generateTable_list(self, full_table_list):
        rc = self.c.get_RC_SUCCESS()
        for i in range(len(full_table_list[:])):
            self.table_list.append(full_table_list[i][2])
        return rc, self.table_list
    #---------------------------------------------------------------------------
    # [Reader]
    #---------------------------------------------------------------------------
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
# end of QueryMsSQL module
#-------------------------------------------------------------------------------
