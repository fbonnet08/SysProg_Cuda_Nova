'''!\file
   -- Gatan_PC_K3_GUI: (Python3 code) is a Python (NumPy/SciPy) application with a Tkinter GUI.
It is a software package for handling the data from EM-microscope and pre processing studied in structural biology, primarily electron cryo-microscopy (cryo-EM).

Please find the manual at https://sourceforge.net/projects/ldmap/


Contributors:
  F.D.R. Bonnet

This package is released under the Creative Commons
Attribution-NonCommercial-NoDerivs CC BY-NC-ND License
(http://creativecommons.org/licenses/by-nc-nd/3.0/)

Usage:
  Gatan_PC_K3_GUI.py [(--noguiSplit INPUT)]
                     [--maskVol=MASKVOL]
                     [--launchChimera=CHIMERABIN]
                     [--use_gpu=USE_GPU]
                     [--set_gpu=SET_GPU]
                     [--lib_krnl_gpu=LIB_GPU]
                     [--doBenchMarking]

NOTE: INPUT(s) is/are mandatory

Arguments:
  INPUTS                      Input volumes in MRC format

Options:
  --noguiSplit                Run Gatan_PC_K3_GUI for Split Volumes in command-line mode.
  --maskVol=MASKVOL           Mask volume.
  --launchChimera=CHIMERABIN  Launch Chimera after execution with bin Path [default:~/UCSF-Chimera64-1.13.1/bin/chimera].
  --use_gpu=USE_GPU           Uses GPU {yes|no} [default=no].
  --set_gpu=SET_GPU           Sets GPU {(0,1,...) [default=0].
  --lib_krnl_gpu=LIB_GPU      Specifies the library path for the GPU kernels
  --doBenchMarking            Run Gatan_PC_K3_GUI in BenchMarking mode, if combined with   
  --help -h                   Show this help message and exit.
  --version                   Show version.
'''
#System tools
import sys
import tkinter as tk
import multiprocessing
import subprocess
from queue import Empty, Full
#from multiprocessing import Pool
#GUI imports
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askopenfilenames
from tkinter.filedialog import askdirectory
from tkinter.messagebox import showerror
from tkinter.messagebox import showinfo
from tkinter import ttk
import time
import threading
#from tkinter import *
import os
from sys import exit
#TODO: If statement will be removed after checking
sys.path.append(os.path.join(os.getcwd(), '.'))
sys.path.append(os.path.join(os.getcwd(), 'utils'))
sys.path.append(os.path.join(os.getcwd(), 'utils','StateMachine'))
sys.path.append(os.path.join(os.getcwd(), 'Scripts'))
sys.path.append(os.path.join(os.getcwd(), 'Scripts','Gatan_PC_K3'))
#appending the utils path
from DataManage_common import *
#platform, release  = whichPlatform()
sysver, platform, system, release, node, processor, cpu_count = whichPlatform()
#application methods
from docopt import docopt
from DataManage_fileIO import *
#from DataManage_header import print_Gatan_PC_K3_header
import DataManage_header
#from messageHandler import *
import utils.messageHandler
from DataManage_descriptors import *  #bring in the documentation
import DataManage_common
#from simple_device import SimpleDevice
import utils.StateMachine.simple_device
#import gui_ioChannel
################################################################################
#                                                                              
#                                                                              
#           Gatan_PC_K3_GUI application to manage and real process data on CPU-GPU   
#                          for given data sets from EM-microscope
#                                                                              
################################################################################
running = True  # Global flag
#-------------------------------------------------------------------------------
# [class] GainRefGuiApp
#-------------------------------------------------------------------------------
##\brief Python3 method.
# GainRefGuiApp application helper class
class StopAllGuiApp(object):
    def __init__(self,q):
        self.root = tk.Tk()
        self.root.geometry('750x250')
        self.root.title("Stop All Running Jobs (Project Handler)")
        self.text_wid = tk.Text(self.root,height=200,width=200)
        self.text_wid.pack(expand=1,fill=tk.BOTH)
        self.root.after(200,self.CheckQueuePoll,q)
    #---------------------------------------------------------------------------
    # [CheckQueuePoll] help method
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # CheckQueuePoll class method to check the inputs and then run the application
    # \param self     The self Object
    # \param c_queue  multiprocessing Queue object
    def CheckQueuePoll(self,c_queue):
        try:
            str = c_queue.get(0)
            self.text_wid.insert('end',str)
        except Empty:
            pass
        finally:
            self.root.after(100, self.CheckQueuePoll, c_queue)
#-------------------------------------------------------------------------------
# [class] GainRefGuiApp
#-------------------------------------------------------------------------------
##\brief Python3 method.
# GainRefGuiApp application helper class
class GainRefGuiApp(object):
    def __init__(self,q):
        self.root = tk.Tk()
        self.root.geometry('750x250')
        self.root.title("Gain Reference copy Handler (Project Handler)")
        self.text_wid = tk.Text(self.root,height=200,width=200)
        self.text_wid.pack(expand=1,fill=tk.BOTH)
        self.root.after(200,self.CheckQueuePoll,q)
    #---------------------------------------------------------------------------
    # [CheckQueuePoll] help method
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # CheckQueuePoll class method to check the inputs and then run the application
    # \param self     The self Object
    # \param c_queue  multiprocessing Queue object
    def CheckQueuePoll(self,c_queue):
        try:
            str = c_queue.get(0)
            self.text_wid.insert('end',str)
        except Empty:
            pass
        finally:
            self.root.after(100, self.CheckQueuePoll, c_queue)
#-------------------------------------------------------------------------------
# [class] FramesWarpGuiApp
#-------------------------------------------------------------------------------
##\brief Python3 method.
# FramesWarpGuiApp application helper class
class FramesWarpGuiApp(object):
    def __init__(self,q):
        self.root = tk.Tk()
        self.root.geometry('600x250')
        self.root.title("Frames copy/move Handler (Project Handler)")
        self.text_wid = tk.Text(self.root,height=200,width=200)
        self.text_wid.pack(expand=1,fill=tk.BOTH)
        self.root.after(200,self.CheckQueuePoll,q)
    #---------------------------------------------------------------------------
    # [CheckQueuePoll] help method
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # CheckQueuePoll class method to check the inputs and then run the application
    # \param self     The self Object
    # \param c_queue  multiprocessing Queue object
    def CheckQueuePoll(self,c_queue):
        try:
            str = c_queue.get(0)
            self.text_wid.insert('end',str)
        except Empty:
            pass
        finally:
            self.root.after(100, self.CheckQueuePoll, c_queue)
#-------------------------------------------------------------------------------
# [class] ResearchDriveGuiApp
#-------------------------------------------------------------------------------
##\brief Python3 method.
# ResearchDriveGuiApp application helper class
class ResearchDriveGuiApp(object):
    def __init__(self,q):
        self.root = tk.Tk()
        self.root.geometry('1000x500')
        self.root.title("ResearchDrive Downloading (Project Handler)")
        self.text_wid = tk.Text(self.root,height=200,width=200)
        self.text_wid.pack(expand=1,fill=tk.BOTH)
        self.root.after(200,self.CheckQueuePoll,q)
    #---------------------------------------------------------------------------
    # [CheckQueuePoll] help method
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # CheckQueuePoll class method to check the inputs and then run the application
    # \param self     The self Object
    # \param c_queue  multiprocessing Queue object
    def CheckQueuePoll(self,c_queue):
        try:
            str = c_queue.get(0)
            self.text_wid.insert('end',str)
        except Empty:
            pass
        finally:
            self.root.after(100, self.CheckQueuePoll, c_queue)
#-------------------------------------------------------------------------------
# [class] ResearchDriveGuiApp
#-------------------------------------------------------------------------------
##\brief Python3 method.
# RCloneSetupGuiApp application helper class
class RCloneSetupGuiApp(object):
    def __init__(self,q):
        self.root = tk.Tk()
        self.root.geometry('1000x300')
        self.root.title("RClone config setup (Project Handler)")
        self.text_wid = tk.Text(self.root,height=200,width=200)
        self.text_wid.pack(expand=1,fill=tk.BOTH)
        self.root.after(200,self.CheckQueuePoll,q)
    #---------------------------------------------------------------------------
    # [CheckQueuePoll] help method
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # CheckQueuePoll class method to check the inputs and then run the application
    # \param self     The self Object
    # \param c_queue  multiprocessing Queue object
    def CheckQueuePoll(self,c_queue):
        try:
            str = c_queue.get(0)
            self.text_wid.insert('end',str)
        except Empty:
            pass
        finally:
            self.root.after(100, self.CheckQueuePoll, c_queue)
#-------------------------------------------------------------------------------
# [class] DataManagerGuiApp
#-------------------------------------------------------------------------------
##\brief Python3 method.
# DataManagerGuiApp application helper class
class DataManagerGuiApp(object):
    def __init__(self,q):
        self.root = tk.Tk()
        self.root.geometry('500x250')
        self.root.title("DataManager Application (Project Handler)")
        self.text_wid = tk.Text(self.root,height=200,width=200)
        self.text_wid.pack(expand=1,fill=tk.BOTH)
        self.root.after(200,self.CheckQueuePoll,q)
    #---------------------------------------------------------------------------
    # [CheckQueuePoll] help method
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # CheckQueuePoll class method to check the inputs and then run the application
    # \param self     The self Object
    # \param c_queue  multiprocessing Queue object
    def CheckQueuePoll(self,c_queue):
        try:
            str = c_queue.get(0)
            self.text_wid.insert('end',str)
        except Empty:
            pass
        finally:
            self.root.after(100, self.CheckQueuePoll, c_queue)
#-------------------------------------------------------------------------------
# GatanFramesGainApp application
#-------------------------------------------------------------------------------
##\brief Python3 method.
# GatanFramesGainApp application class
class GatanFramesGainApp(object):
    """
    GUI Tkinter class for DataManage
    """
    ##\brief Python3 method.
    #GatanFramesGainApp application class constructor
    def __init__(self, parent, c, m):

        #function stamp
        __func__= sys._getframe().f_code.co_name
        #instantiating the common class
        self.c = c #DataManage_common()
        #instantiating messaging class
        logfile = self.c.getLogGatan_PC_K3_GUI()  #getting the name of the global log file
        self.m = m #utils.messageHandler.messageHandler(logfile=logfile)#messageHandler(logfile = logfile)

        #Starting the StateMachine
        self.m.printMesg("State Machine SimpleDevice class instantiated, log: "+ c.getLogfileName())
        self.device = utils.StateMachine.simple_device.SimpleDevice(self.c, self.m)#SimpleDevice(c,m)

        self.m.printMesgStr("State Machine: ",c.get_B_Green()," device_locked")
        self.device.on_event('device_locked')
        self.device.state

        # General Settings
        self.parent = parent
        self.parent.title("LDMaP-APP (Live Data Management and Processing Application for EM-Microscope data) Data Handler v" + version)
        self.parent.option_add('*tearOff', False)

        self.myStyle = ttk.Style()
        self.myStyle.configure('GatanFramesGainApp.TButton', foreground='blue4',
                                font='Helvetica 9 bold')
        self.myStyle.configure('DataManageAppRed.TButton', foreground='red4',
                                font='Helvetica 9 bold')
        self.myStyle.configure('DataManageAppStd.TButton',
                               font='Helvetica 9')

        self.font_size_all_text = "Helvetica 10"
        self.font_size_headers =  "Helvetica 10 bold"
        
        ## MENUBAR
        # Create menubar frame
        self.menubar = ttk.Frame(parent)
        self.menubar.pack(side=tk.TOP, fill=tk.X)

        # Create Help menubutton"path":
        self.mb_help = ttk.Menubutton(self.menubar, text="Help")
        self.mb_help.pack(side=tk.RIGHT)

        # Create Help menu
        self.helpMenu = tk.Menu(self.mb_help)
        self.helpMenu.add_command(label="Documentation",
                                  command=self.showDocumentation)
        self.helpMenu.add_command(label="About LDMaP-APP Data Handler",
                                  command=self.showAbout)

        # Attach the Help menu to the Help menubutton
        self.mb_help.config(menu = self.helpMenu)

        ## MASTER FRAME
        # Create master frame that holds everything
        self.masterframe = ttk.Frame(parent)
        self.masterframe.pack()

        # Create notebook
        self.nb = ttk.Notebook(self.masterframe, style='DataManage.TNotebook')
        self.nb.enable_traversal() # allow for keyboard bindings
        self.nb.pack()

        #Micrograph frame
        #[Home]
        #[NeCEN]
        self.app_root         = tk.StringVar(value="/home/frederic/Leiden/SourceCode/DataManage_project")
        self.projectName      = tk.StringVar(value="")
        #/data/local/frederic/SourceDir_KriosData
        self.data_path        = tk.StringVar(value= "/data/local/frederic/ResearchDriveDownload")    #"/data/krios1buffer") #"../Micrographs/*.mrc")
        #self.targetDirectory  = tk.StringVar(value="/data/local/frederic/TargetDir")
        #Warp target directory
        self.WarpTargetDir    = tk.StringVar(value="/scratch/frederic/TargetDir/GatanScripts/WarpFolder")
        #Gain reference 
        self.GainSourceDir    = tk.StringVar(value="/data/local/frederic/GainRef_krios1")
        self.GainTargetDir    = tk.StringVar(value="/scratch/frederic/TargetDir/GatanScripts/GainrefFolder")
        #Research Drive
        self.cdiName          = tk.StringVar(value="cdi-Frederic")
        self.resrchDrvSrcDir  = tk.StringVar(value="krios1/20210401_2001-997_EPU-SP_Fred") # 20210401_2001-997_EPU-SP_Fred supervisor_20210401_153757")
        self.resrchDrvTrgDir  = tk.StringVar(value="/data/local/frederic/ResearchDriveDownload")

        self.doBenchMarking   = tk.BooleanVar(value=False)
        self.use_gpu          = tk.BooleanVar(value=False)
        #Software to be used ComboBox
        self.software_SP      = tk.StringVar(value="EPU-2.6")
        self.software_TOMO    = tk.StringVar(value="TOMO4")

        #Movie Extension #--movies_ext=tiff--->radioButton
        self.movies_ext       = tk.StringVar(value="mrc")
        #Gain type --gain_type=[x0m0,x0m1,x1m0,x1m1]--->radioButton
        self.gain_type        = tk.StringVar(value="x0.m0")

        #--file_frame_copy_move_handler=copy---> combo box
        self.file_frame_copy_move_handler = "copy" #tk.StringVar(value="copy")
        #--multiThreads=no---> combo box
        self.multiThreads_pst_handler = "no" #tk.StringVar(value="no")
        
        # Create split volume input frame
        self.splitframe = ttk.Frame(self.nb)

        self.splitframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.splitframe.columnconfigure(0, weight=1)
        self.splitframe.rowconfigure(   0, weight=1)

        #-----------------------------------------------------------------------
        # [Project and Micrographs Input]
        #-----------------------------------------------------------------------
        #adding the frames to the main frame
        self.nb.add(self.splitframe, text='LDMaP Project and Micrographs Input', underline=0, padding=10)
        #-----------------------------------------------------------------------
        # self.splitframe Frame
        #-----------------------------------------------------------------------
        #-----------------------------------------------------------------------
        # [Required Inputs]
        #-----------------------------------------------------------------------
        # ROW 0
        row = 0
        ttk.Label(self.splitframe, text="Inputs", font = self.font_size_headers).grid(column=1, row=0, sticky=tk.W)

        # ROW 1
        row = 1
        ttk.Label(self.splitframe, text="Application Root:", foreground="blue", font=self.font_size_all_text).grid(column=1, row=1, sticky=tk.E)
        ttk.Entry(self.splitframe, width=30, textvariable=self.app_root, foreground="green", font=self.font_size_all_text).grid(column=2, columnspan=10, row=1, sticky=(tk.W, tk.E))
        ttk.Button(self.splitframe, text="Set app path   ", style='DataManageAppStd.TButton', command=(lambda: self.load_directory(self.app_root))).grid(column=12, row=1, sticky=(tk.W,tk.E))

        # ROW 2
        row = 2
        ttk.Label(self.splitframe, text="Project Name (optional):", foreground="blue", font=self.font_size_all_text).grid(column=1, row=2, sticky=tk.E)
        ttk.Entry(self.splitframe, width=30, textvariable=self.projectName, foreground="green", font=self.font_size_all_text).grid(column=2, columnspan=10, row=2, sticky=(tk.W, tk.E))
        ttk.Button(self.splitframe, text="Set Project Name", style='DataManageAppStd.TButton', command=(lambda: self.projectNameEntryActivator(self.projectName))) .grid(column=12, row=row, sticky=(tk.W,tk.E))#tk.W)
        self.c.setProjectName(self.projectName.get())

        # ROW 3
        row = 3
        ttk.Label(self.splitframe, text="Source Directory:", foreground="blue", font=self.font_size_all_text).grid(column=1, row=3, sticky=tk.E)
        ttk.Entry(self.splitframe, width=30, textvariable=self.data_path, foreground="green", font=self.font_size_all_text).grid(column=2, columnspan=10, row=3, sticky=(tk.W, tk.E))
        ttk.Button(self.splitframe, text="Set Source Directory", style='DataManageAppStd.TButton', command=(lambda: self.load_directory(self.data_path))).grid(column=12, row=3, sticky=(tk.W,tk.E))

        # ROW 4
        #ttk.Label(self.splitframe, text="Gain Refs  :", foreground="blue").grid(column=1, row=4, sticky=tk.E)
        #ttk.Entry(self.splitframe, width=30, textvariable=self.gainRef, foreground="green").grid(column=2, columnspan=10, row=4, sticky=(tk.W, tk.E))
        #ttk.Button(self.splitframe, text="Load Gain References", command=(lambda: self.load_file(self.gainRef))).grid(column=12, row=4, sticky=tk.W)

        # ROW 5
        #ttk.Label(self.splitframe, text="Target Directory:", foreground="blue").grid(column=1, row=5, sticky=tk.E)
        #ttk.Entry(self.splitframe, width=30, textvariable=self.targetDirectory, foreground="green").grid(column=2, columnspan=10, row=5, sticky=(tk.W, tk.E))
        #ttk.Button(self.splitframe, text="Set Target Directory", command=(lambda: self.load_directory(self.targetDirectory))).grid(column=12, row=5, sticky=tk.W)
        
        # ROW 6
        row = 6
        ttk.Label(self.splitframe, text="Single Particle:", font=self.font_size_all_text).grid(column=1, row=6, sticky=tk.E)
        #SP software
        #softwareSP=["EPU-2.6", "EPU-2.7", "SerialEM"]
        softwareSP=["EPU-2.10.5", "SerialEM"]
        # Radiobutton callback function
        def radCall_SP():
            radSel_SP=radVar_SP.get()
            if   radSel_SP == 0:
                self.software_SP = "EPU-2.10.5"
                self.m.printMesgAddStr(" software_SP = ",
                                       self.c.getGreen(),self.software_SP)
            #elif radSel_SP == 1:
            #    self.software_SP = "EPU-2.7"
            #    self.m.printMesgAddStr(" software_SP = ",
            #                           self.c.getYellow(),self.software_SP)
            elif radSel_SP == 1:
                self.software_SP = "SerialEM"
                self.m.printMesgAddStr(" software_SP = ",
                                       self.c.getCyan(),self.software_SP)

        radVar_SP = tk.IntVar()
        # Next we are selecting a non-existing index value for radVar_SP.
        radVar_SP.set(99)

        # Now we are creating all three Radiobutton widgets within one loop.
        for col in range(len(softwareSP)):
            set_col = col+2
            curRad = 'rad' + str(col)
            curRad = tk.Radiobutton(self.splitframe, text=softwareSP[col], variable=radVar_SP, value=col, command=radCall_SP)
            curRad.grid(column=set_col, row=6, columnspan=3, sticky=tk.W)

        # ROW 7
        row = 7
        ttk.Label(self.splitframe, text="Tomography:", font=self.font_size_all_text).grid(column=1, row=7, sticky=tk.E)
        #TOMO software
        softwareTOMO=["TOMO4", "TOMO5", "SerialEM"]
        # Radiobutton callback function
        def radCall_TOMO():
            radSel_TOMO=radVar_TOMO.get()
            if   radSel_TOMO == 0:
                self.software_TOMO = "TOMO4"
                #print("software_TOMO = ",self.software_TOMO)
                self.m.printMesgAddStr(" software_TOMO = ",
                                       self.c.getRed(),self.software_TOMO)

            elif radSel_TOMO == 1:
                self.software_TOMO = "TOMO5"
                self.m.printMesgAddStr(" software_TOMO = ",
                                       self.c.getRed(),self.software_TOMO)
            elif radSel_TOMO == 2:
                self.software_TOMO = "SerialEM"
                self.m.printMesgAddStr(" software_TOMO = ",
                                       self.c.getRed(),self.software_TOMO)

        radVar_TOMO = tk.IntVar()
        # Next we are selecting a non-existing index value for radVar_TOMO.
        radVar_TOMO.set(98)

        # Now we are creating all three Radiobutton widgets within one loop.
        for col in range(len(softwareTOMO)):
            set_col = col+2
            curRad = 'rad' + str(col)
            curRad = tk.Radiobutton(self.splitframe, text=softwareTOMO[col], variable=radVar_TOMO, value=col, command=radCall_TOMO)
            curRad.grid(column=set_col, row=7, columnspan=3, sticky=tk.W)

        # ROW 8
        row = 8
        ttk.Label(self.splitframe, text="Movie Extension:", font=self.font_size_all_text).grid(column=1, row=8, sticky=tk.E)
        #Movie Extension
        moviesExt=["mrc", "tiff"]
        # Radiobutton callback function
        def radCall_MovieExt():
            radSel_MovieEXT=radVar_MovieEXT.get()
            if   radSel_MovieEXT == 0:
                self.movies_ext = "mrc"
                self.m.printMesgAddStr(" Movies ext = ",
                                       self.c.getYellow(),self.movies_ext)
            elif radSel_MovieEXT == 1:
                self.movies_ext = "tiff"
                self.m.printMesgAddStr(" Movies ext = ",
                                       self.c.getMagenta(),self.movies_ext)

        radVar_MovieEXT = tk.IntVar()
        # Next we are selecting a non-existing index value for radVar_MovieEXT.
        radVar_MovieEXT.set(97)

        # Now we are creating all three Radiobutton widgets within one loop.
        for col in range(len(moviesExt)):
            set_col = col+2
            curRad = 'rad' + str(col)
            curRad = tk.Radiobutton(self.splitframe, text=moviesExt[col], variable=radVar_MovieEXT, value=col, command=radCall_MovieExt)
            curRad.grid(column=set_col, row=8, columnspan=3, sticky=tk.W)
        #-----------------------------------------------------------------------
        # [Fractions movement Inputs]
        #-----------------------------------------------------------------------
        # ROW 9
        row = 9
        ttk.Label(self.splitframe, text="Fractions movement", font = self.font_size_headers).grid(column=1, row=9, sticky=tk.W)

        # ROW 11
        row = 11
        ttk.Label(self.splitframe, text="Warp Target Directory:", foreground="blue", font=self.font_size_all_text).grid(column=1, row=11, sticky=tk.E)
        ttk.Entry(self.splitframe, width=30, textvariable=self.WarpTargetDir, foreground="green", font=self.font_size_all_text).grid(column=2, columnspan=10, row=11, sticky=(tk.W, tk.E))
        ttk.Button(self.splitframe, text="Set Warp Target Directory", style='DataManageAppStd.TButton', command=(lambda: self.load_directory(self.WarpTargetDir))).grid(column=12, row=11, sticky=(tk.W,tk.E))

        # ROW 12
        row = 12
        ttk.Label(self.splitframe, text="How to handle files:", font=self.font_size_all_text).grid(column=1, row=12, sticky=tk.E)
        #Movie Extension
        filePstHandler=["copy", "move", "rsync_copy","rsync_move"]
        # Radiobutton callback function
        def radCall_file_frames_handler():
            radSel_FPSTHdl=radVar_FilePstHdl.get()
            if   radSel_FPSTHdl == 0:
                self.file_frame_copy_move_handler = "copy"
                self.m.printMesgAddStr(" File handler = ",
                                       self.c.getGreen(),
                                       "Frames will be: "+self.c.get_B_Yellow()+\
                                       self.file_frame_copy_move_handler)
            elif radSel_FPSTHdl == 1:
                self.file_frame_copy_move_handler = "move"
                self.m.printMesgAddStr(" File handler = ",
                                       self.c.getGreen(),
                                       "Frames will be: "+self.c.get_B_Cyan()+\
                                       self.file_frame_copy_move_handler)
            elif radSel_FPSTHdl == 2:
                self.file_frame_copy_move_handler = "rsync_copy"
                self.m.printMesgAddStr(" File handler = ",
                                       self.c.getGreen(),
                                       "Frames will be: "+self.c.get_B_Blue()+\
                                       self.file_frame_copy_move_handler)
            elif radSel_FPSTHdl == 3:
                self.file_frame_copy_move_handler = "rsync_move"
                self.m.printMesgAddStr(" File handler = ",
                                       self.c.getGreen(),
                                       "Frames will be: "+self.c.get_B_Magenta()+\
                                       self.file_frame_copy_move_handler)

        radVar_FilePstHdl = tk.IntVar()
        # Next we are selecting a non-existing index value for radVar_FilePstHdl.
        radVar_FilePstHdl.set(95)

        # Now we are creating all three Radiobutton widgets within one loop.
        for col in range(len(filePstHandler)):
            set_col = col+2
            curRad = 'rad' + str(col)
            curRad = tk.Radiobutton(self.splitframe, text=filePstHandler[col], variable=radVar_FilePstHdl, value=col, command=radCall_file_frames_handler)
            curRad.grid(column=set_col, row=12, columnspan=6, sticky=tk.W)

        # ROW 13
        row = 13
        ttk.Label(self.splitframe, text="Multithreads copying (Default no):", font=self.font_size_all_text).grid(column=1, row=13, sticky=tk.E)
        #Movie Extension
        multiThreadsHandler=["yes", "no"]
        # Radiobutton callback function
        def radCall_multiThreads_handler():
            radSel_multiThreadsHdl=radVar_multiThreadsHdl.get()
            if   radSel_multiThreadsHdl == 0:
                self.multiThreads_pst_handler = "yes"
                self.m.printMesgAddStr(" multiThreads = ",
                                       self.c.getGreen(),
                                       "Frames will be copied/moved in"+self.c.get_B_White()+" parallel")
            elif radSel_multiThreadsHdl == 1:
                self.multiThreads_pst_handler = "no"
                self.m.printMesgAddStr(" multiThreads = ",
                                       self.c.getGreen(),
                                       "Frames will be copied/moved in"+self.c.get_B_Magenta()+" serial")

        radVar_multiThreadsHdl = tk.IntVar()
        # Next we are selecting a non-existing index value for radVar_multiThreadsHdl.
        radVar_multiThreadsHdl.set(95)

        # Now we are creating all three Radiobutton widgets within one loop.
        for col in range(len(multiThreadsHandler)):
            set_col = col+2
            curRad = 'rad' + str(col)
            curRad = tk.Radiobutton(self.splitframe, text=multiThreadsHandler[col], variable=radVar_multiThreadsHdl, value=col, command=radCall_multiThreads_handler)
            curRad.grid(column=set_col, row=13, columnspan=6, sticky=tk.W)

        #-----------------------------------------------------------------------
        # [Gain Reference movement Inputs]
        #-----------------------------------------------------------------------
        # ROW 17
        row = 17
        ttk.Label(self.splitframe, text="Gain Reference movement", font = self.font_size_headers).grid(column=1, row=17, sticky=tk.W)

        # ROW 18
        row = 18
        ttk.Label(self.splitframe, text="Gain Source Directory:", foreground="blue", font=self.font_size_all_text).grid(column=1, row=18, sticky=tk.E)
        ttk.Entry(self.splitframe, width=30, textvariable=self.GainSourceDir, foreground="green", font=self.font_size_all_text).grid(column=2, columnspan=10, row=18, sticky=(tk.W, tk.E))
        ttk.Button(self.splitframe, text="Set Gain Source Directory", style='DataManageAppStd.TButton', command=(lambda: self.load_directory(self.GainSourceDir))).grid(column=12, row=18, sticky=(tk.W,tk.E))

        # ROW 19
        row = 19
        ttk.Label(self.splitframe, text="Gain Target Directory:", foreground="blue", font=self.font_size_all_text).grid(column=1, row=19, sticky=tk.E)
        ttk.Entry(self.splitframe, width=30, textvariable=self.GainTargetDir, foreground="green", font=self.font_size_all_text).grid(column=2, columnspan=10, row=19, sticky=(tk.W, tk.E))
        ttk.Button(self.splitframe, text="Set Gain Target Directory", style='DataManageAppStd.TButton', command=(lambda: self.load_directory(self.GainTargetDir))).grid(column=12, row=19, sticky=(tk.W,tk.E))

        #ROW 20
        row = 20
        ttk.Label(self.splitframe, text="Gain Ref Type:", font=self.font_size_all_text).grid(column=1, row=20, sticky=tk.E)
        #Movie Extension
        gainType=["x0.m0", "x0.m1", "x1.m0", "x1.m1"]
        # Radiobutton callback function
        def radCall_GainType():
            radSel_GainType=radVar_GainType.get()
            if   radSel_GainType == 0:
                self.gain_type = "x0.m0"
                self.c.setGain_Type(self.gain_type)
                self.m.printMesgAddStr(" Gain type  = ",
                                       self.c.getYellow(),self.c.getGain_Type())
            elif radSel_GainType == 1:
                self.gain_type = "x0.m1"
                self.c.setGain_Type(self.gain_type)
                self.m.printMesgAddStr(" Gain Type  = ",
                                       self.c.getMagenta(),self.c.getGain_Type())
            elif radSel_GainType == 2:
                self.gain_type = "x1.m0"
                self.c.setGain_Type(self.gain_type)
                self.m.printMesgAddStr(" Gain Type  = ",
                                       self.c.getGreen(),self.c.getGain_Type())
            elif radSel_GainType == 3:
                self.gain_type = "x1.m1"
                self.c.setGain_Type(self.gain_type)
                self.m.printMesgAddStr(" Gain Type  = ",
                                       self.c.getCyan(),self.c.getGain_Type())

        radVar_GainType = tk.IntVar()
        # Next we are selecting a non-existing index value for radVar_GainType.
        radVar_GainType.set(97)

        # Now we are creating all three Radiobutton widgets within one loop.
        for col in range(len(gainType)):
            set_col = col+2
            curRad = 'rad' + str(col)
            curRad = tk.Radiobutton(self.splitframe, text=gainType[col], variable=radVar_GainType, value=col, command=radCall_GainType)
            curRad.grid(column=set_col, row=20, columnspan=3, sticky=tk.W)

        #-----------------------------------------------------------------------
        # [Gain Reference movement Inputs]
        #-----------------------------------------------------------------------
        # ROW 21
        row = 21
        ttk.Label(self.splitframe, text="Download Research Drive", font = self.font_size_headers).grid(column=1, row=21, sticky=tk.W)

        # ROW 22
        row = 22
        #cdi
        ttk.Label(self.splitframe, text="cdi Name:", foreground="blue", font=self.font_size_all_text).grid(column=1, row=22, sticky=tk.E)
        ttk.Entry(self.splitframe, width=30, textvariable=self.cdiName, foreground="green", font=self.font_size_all_text).grid(column=2, columnspan=10, row=22, sticky=(tk.W, tk.E))

        # ROW 23
        row = 23
        #name of directory on research drive
        ttk.Label(self.splitframe, text="Research Drive Source Directory:", foreground="blue", font=self.font_size_all_text).grid(column=1, row=23, sticky=tk.E)
        ttk.Entry(self.splitframe, width=30, textvariable=self.resrchDrvSrcDir, foreground="green", font=self.font_size_all_text).grid(column=2, columnspan=10, row=23, sticky=(tk.W, tk.E))
        # ROW 24
        row = 24
        #Target directory on system
        ttk.Label(self.splitframe, text="Download Target Directory:", foreground="blue", font=self.font_size_all_text).grid(column=1, row=24, sticky=tk.E)
        ttk.Entry(self.splitframe, width=30, textvariable=self.resrchDrvTrgDir, foreground="green", font=self.font_size_all_text).grid(column=2, columnspan=10, row=24, sticky=(tk.W, tk.E))
        ttk.Button(self.splitframe, text="Set Target Directory", style='DataManageAppStd.TButton', command=(lambda: self.load_directory(self.resrchDrvTrgDir))).grid(column=12, row=24, sticky=(tk.W,tk.E))
        #-----------------------------------------------------------------------
        # [Lower-top-buttons]
        #-----------------------------------------------------------------------
        # ROW 25
        row = 25
        # Copy/Move Fractions button
        ttk.Button(self.splitframe, text="Download From RD", style='GatanFramesGainApp.TButton', command=self.DownloadResearchDrive).grid(column=2, row=25, sticky=(tk.W,tk.E))#tk.W, sticky=tk.E
        # Copy/Move Fractions button
        ttk.Button(self.splitframe, text="Copy/Move Fractions", style='GatanFramesGainApp.TButton', command=self.Copy_Move_Fractions).grid(column=3, row=25, sticky=(tk.W,tk.E))#tk.W, sticky=tk.E
        # Copy Gain Reference button
        ttk.Button(self.splitframe, text="Copy Gain Reference", style='GatanFramesGainApp.TButton', command=self.Copy_Gain_Reference).grid(column=4, row=25, sticky=(tk.W,tk.E)) #, columnspan=4, sticky=tk.E
        # Stop All button
        ttk.Button(self.splitframe, text="Stop ALL", style='GatanFramesGainApp.TButton', command=self.stopAll).grid(column=5, row=25, sticky=(tk.W,tk.E)) #, columnspan=4, sticky=tk.E
        # Check Inputs and Launch DataManagerApp button
        ttk.Button(self.splitframe, text="Launch Pre-Proc", style='GatanFramesGainApp.TButton', command=self.checkInputsAndLaunchDataManagerApp).grid(column=6, row=25, sticky=(tk.W,tk.E))#tk.E) #, columnspan=4
        #-----------------------------------------------------------------------
        # [Lower-bottom-buttons]
        #-----------------------------------------------------------------------
        # ROW 26
        row = 26
        # Quit button
        ttk.Button(self.splitframe, text="Quit", style='DataManageAppRed.TButton', command=self.quit).grid(column=1, row=26, sticky=tk.W) # ,
        # Stop Download Button
        ttk.Button(self.splitframe, text="Stop Download", style='GatanFramesGainApp.TButton', command=self.StopDownloading).grid(column=2, row=26, sticky=(tk.W,tk.E)) #, columnspan=4, sticky=tk.E
        # Stop Copy/Move button
        ttk.Button(self.splitframe, text="Stop Copy/Move", style='GatanFramesGainApp.TButton', command=self.StopCopyMoveFractions).grid(column=3, row=26, sticky=(tk.W,tk.E)) #, columnspan=4, sticky=tk.E
        # Stop Copy Gain Reference button
        ttk.Button(self.splitframe, text="Stop Copy Gain Reference", style='GatanFramesGainApp.TButton', command=self.StopCopyGainReference).grid(column=4, row=26, sticky=(tk.W,tk.E)) #, columnspan=4, sticky=tk.E
        # Rclone setup button
        ttk.Button(self.splitframe, text="Setup rclone", style='GatanFramesGainApp.TButton', command=self.setupRcloneConfig).grid(column=6, row=26, sticky=(tk.W,tk.E))#, sticky=tk.E) #, columnspan=4
        #-----------------------------------------------------------------------
        # [Setup-grid] Setup grid with padding
        #-----------------------------------------------------------------------
        for child in self.splitframe.winfo_children(): child.grid_configure(padx=1, pady=2)
        #-----------------------------------------------------------------------
        # Binding upon return
        #-----------------------------------------------------------------------
        self.parent.bind("<Return>",self.checkInputsAndLaunchDataManagerApp)
    #---------------------------------------------------------------------------
    # class methods
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [Setters]
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [browsePath]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # GatanFramesGainApp class method to browse the file and load the chimera binary
    # \param self                  The self Object
    # \param chimeraBin            Chimera binary
    def browsePath(self,chimeraBin):
        __func__= sys._getframe().f_code.co_name
        options =  {}
        options['title'] = "Gatan_PC_K3_GUI - Chimera path to binary"
        fname = askopenfilename(**options)
        if fname:
            try:
                chimeraBin.set(fname)
            except:                     # <- naked except is a bad idea
                showerror("Binary file unusable", "Check path and location and try again\n'%s'" % fname)
            return
    #---------------------------------------------------------------------------
    # [load_file]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # GatanFramesGainApp class method to load file
    # \param self                  The self Object
    # \param fileNameStringVar     Strin variable file name
    def load_file(self, fileNameStringVar):
        __func__= sys._getframe().f_code.co_name
        options =  {}
        # options['filetypes'] = [ ("All files", ".*"), ("MRC map", ".map,.mrc") ]      # THIS SEEMS BUGGY...
        options['title'] = "DataManager - Select data file"
        fname = askopenfilename(**options)
        if fname:
            try:
                fileNameStringVar.set(fname)
            except:                     # <- naked except is a bad idea
                showerror("Open Source File", "Failed to read file\n'%s'" % fname)
            return
    #---------------------------------------------------------------------------
    # [load_files]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # GatanFramesGainApp class method to load the two inpout maps files
    # \param self                  The self Object
    # \param fileNameStringVar1    Input string for Volume 1
    # \param fileNameStringVar2    Input string for Volume 2
    def load_files(self, fileNameStringVar1, fileNameStringVar2):
        __func__= sys._getframe().f_code.co_name
        options =  {}
        options['title'] = "DataManager - Select data files"
        fname = askopenfilenames(**options)
        if isinstance( fname, tuple ):
            try:
                fileNameStringVar1.set(fname[0])
                fileNameStringVar2.set(fname[1])
            except:                     # <- naked except is a bad idea
                showerror("Open Source Files", "Failed to read files\n'%s'" % fname)
            return
        if isinstance( fname, unicode ):
            try:
                fileNameStringVar1.set(fname.partition(' ')[0])
                fileNameStringVar2.set(fname.partition(' ')[2])
            except:                     # <- naked except is a bad idea
                showerror("Open Source Files", "Failed to read files\n'%s'" % fname)
            return
    #---------------------------------------------------------------------------
    # [projectNameEntryActivator]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # DataManageApp class method to load the two inpout maps files
    # \param self                  The self Object
    # \param fileNameStringVar1    Input string for Volume 1
    def projectNameEntryActivator(self, projectName):
        __func__= sys._getframe().f_code.co_name
        options =  {}
        options['title'] = "LDMaP-App - Select a Project Name"
        fname = askdirectory(**options)
        head, tail =os.path.split(fname)
        if fname:
            try:
                projectName.set(tail)
                self.m.printMesgAddStr(" Project Name = ", self.c.getGreen(), projectName.get())
            except:                     # <- naked except is a bad idea
                showerror("Open Source Files", "Failed to load Directory\n'%s'" % fname)
            return
    #---------------------------------------------------------------------------
    # [load_directory]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # GatanFramesGainApp class method to load the two inpout maps files
    # \param self                  The self Object
    # \param fileNameStringVar1    Input string for Volume 1
    # \param fileNameStringVar2    Input string for Volume 2
    def load_directory(self, fileNameStringVar1):
        __func__= sys._getframe().f_code.co_name
        options =  {}
        options['title'] = "DataManager - Select data files"
        fname = askdirectory(**options)
        if fname:
            try:
                fileNameStringVar1.set(fname)
            except:                     # <- naked except is a bad idea
                showerror("Open Source Files", "Failed to load Directory\n'%s'" % fname)
            return

    #---------------------------------------------------------------------------
    # [quit] method for quiting the application
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # GatanFramesGainApp class method to quit the application
    # \param self     The self Object
    def quit(self):
        self.m.printCMesg("DataManage exits without computing.", self.c.get_B_Red())
        self.m.printMesgInt("Return code: ", self.c.get_B_Red(),self.c.get_RC_STOP())
        exit(self.c.get_RC_STOP())
    #---------------------------------------------------------------------------
    # [cat_outputfile]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # cat_outputfile class method to cat output file via an Xterminal
    # \param self     The self Object
    def cat_outputfile(self,c,m,q,resrchDrvSrcDir,resrchDrvTrgDir):
        __func__= sys._getframe().f_code.co_name
        q.put("-------------------------------------------------------------------------\n")
        q.put("Function                       : "+__func__+"\n")
        q.put("Changing dir for cat_outputfile: "+self.app_root.get()+os.path.sep+"Scripts"+os.path.sep+"Research_drive_upload"+"\n")

        os.chdir(self.app_root.get()+os.path.sep+"Scripts"+os.path.sep+"Research_drive_upload")
        #q.put(subprocess.check_output(['ls'])  )
        q.put(subprocess.check_output(['pwd'])  )
        time.sleep(10)

        file_to_log="ResearchDriveDownload_rclone-Log"+ os.path.sep + \
                     "download_rclone.log"
        logfile=resrchDrvTrgDir+os.path.sep+file_to_log
        q.put("The logfile for this Download  : "+logfile+"\n")
        #print("logfile: ", logfile)
        q.put("-------------------------------------------------------------------------\n")
        q.put("Launching download progress window...\n")
        q.put("-------------------------------------------------------------------------\n")
        os.system('xterm -bg black -fg white -geometry 180x40+50+50 -e \"tail -f %s\"&'%logfile)
        os.chdir(self.app_root.get()+os.path.sep+"Scripts"+os.path.sep+"Gatan_PC_K3")
        #os.system('pwd')
    #---------------------------------------------------------------------------
    # [GenerateData]
    #---------------------------------------------------------------------------
    def GenerateData(self,c,m,q,cdiName,resrchDrvSrcDir,resrchDrvTrgDir):
        __func__= sys._getframe().f_code.co_name
        q.put("-------------------------------------------------------------------------\n")
        q.put("Research Drive download input parameters \n")
        q.put("-------------------------------------------------------------------------\n")
        q.put("Function                       : "+__func__+"\n")
        q.put("The cdi used for this Download : "+cdiName+"\n")
        q.put("Research Drive Source Directory: "+resrchDrvSrcDir+"\n")
        q.put("Download Target directory      : "+resrchDrvTrgDir+"\n")
        q.put("Changing directory to          : "+self.app_root.get()+\
              os.path.sep+"Scripts"+os.path.sep+"Research_drive_upload\n")
        os.chdir(self.app_root.get()+os.path.sep+\
                 "Scripts"+os.path.sep+"Research_drive_upload")
        #q.put(subprocess.check_output(['ls'])  )
        projectName=os.path.basename(os.path.normpath(resrchDrvSrcDir))
        q.put("The Project name downloaded    : "+projectName+"\n")
        targetDir = resrchDrvTrgDir+os.path.sep+projectName
        q.put("Target directory               : "+targetDir+"\n")
        if os.path.exists(targetDir):
            q.put("Target directory               : Exist \n")
        else:
            q.put("Target directory               : Does not exist, we will create it...\n")
            os.mkdir(targetDir)
            q.put("Target directory               : Created ---> Done.\n")

        q.put("The project name               : ["+projectName+"] content "+\
              "from Research Drive will be downloaded in "+\
              "Directory: \n")
        q.put("Target directory               : "+targetDir+"\n")

        #Launching the downloading process from reserach drive
        q.put("-------------------------------------------------------------------------\n")
        q.put("Launching download process from Research Drive...\n")
        q.put("-------------------------------------------------------------------------\n")

        time.sleep(0.5)

        q.put(subprocess.check_output(['sh',
                                       'get_dataResearchDrive_WithLogFile.sh',
                                       cdiName,
                                       resrchDrvSrcDir,
                                       projectName,
                                       resrchDrvTrgDir])  )

        os.chdir(self.app_root.get()+os.path.sep+"Scripts"+os.path.sep+"Gatan_PC_K3")
        #os.system('pwd')
        #os.system('ls')
        return
    #---------------------------------------------------------------------------
    # [DownloadResearchDrive]
    #---------------------------------------------------------------------------
    def DownloadResearchDrive(self,*args):
        __func__= sys._getframe().f_code.co_name
        #Stop scanning by setting the global flag to False.

        self.q = multiprocessing.Queue()
        self.q.cancel_join_thread() # or else thread that puts data will not term
        gui = ResearchDriveGuiApp(self.q)
        t1 = multiprocessing.Process(target=self.GenerateData,
                                     args=(self.c,self.m, self.q,self.cdiName.get(),
                                           self.resrchDrvSrcDir.get(),
                                           self.resrchDrvTrgDir.get() ) )
        t2 = multiprocessing.Process(target=self.cat_outputfile,
                                     args=(self.c,self.m, self.q,
                                           self.resrchDrvSrcDir.get(),
                                           self.resrchDrvTrgDir.get() ) )
        self.m.printMesgStr("ResearchDrive Download: ",
                            self.c.get_B_Green(),"Starts...")
        t1.start()
        t2.start()
        gui.root.mainloop()
        t1.join()
        t2.join()

        return
    #---------------------------------------------------------------------------
    # [Copy/Move Fractions]
    #---------------------------------------------------------------------------
    def Copy_Move_Fractions(self,*args):
        __func__= sys._getframe().f_code.co_name
        #Stop scanning by setting the global flag to False.

        self.m.printMesgStr(   "Frames copy or move Function: ",self.c.get_B_Cyan(),__func__)
        self.m.printMesgAddStr(" Source Directory            : ",
                               self.c.getYellow(),self.data_path.get())
        self.m.printMesgAddStr(" Warp Target Directory       : ",
                               self.c.getGreen(),self.WarpTargetDir.get())
        self.m.printMesgAddStr(" How to handle Frames files  : ",
                               self.c.getMagenta(),self.file_frame_copy_move_handler)
        self.m.printMesgAddStr(" Multithreads copying        : ",
                               self.c.getCyan(),self.multiThreads_pst_handler)

        script_path = self.app_root.get()+os.path.sep+"Scripts"+os.path.sep+"Gatan_PC_K3"
        self.m.printMesgAddStr(" Moving to script folder: ",
                               c.getCyan(), script_path)

        os.chdir(script_path)
        os.system('pwd')

        #Creating the list for the FramesHandler_DriverCode_fileMover_Creator.sh
        in_list = []
        live_ ="sourcedir="+self.data_path.get()
        in_list.append(live_)
        live_ ="targetdir="+self.WarpTargetDir.get()
        in_list.append(live_)
        live_ =self.file_frame_copy_move_handler
        in_list.append(live_)
        live_ ="multiThreads="+self.multiThreads_pst_handler
        in_list.append(live_)
            
        CMD = ' '.join(in_list[:])
        self.m.printMesgAddStr(" sh FramesHandler_DriverCode_fileMover_Creator.sh ",
                               c.getCyan(), CMD)
        #Sleep a fraction of second to make sure that the files actually gets written
        time.sleep(0.5)
        # Call FramesHandler_DriverCode_fileMover_Creator python creator via OS
        os.system('sh FramesHandler_DriverCode_fileMover_Creator.sh %s'%CMD)
        #Sleep a fraction to make sure that file is created and can proceed
        time.sleep(0.5)
        #-----------------------------------------------------------------------
        # Creating the GUI object for the subprocess launch of DataManagerApp
        #-----------------------------------------------------------------------
        self.q3 = multiprocessing.Queue()
        self.q3.cancel_join_thread() # or else thread that puts data will not term
        gui = FramesWarpGuiApp(self.q3)
        t4 = multiprocessing.Process(target=self.LaunchFramesCopyMoveWarpHandler,
                                     args=(self.c,self.m, self.q3 ) )
        t4.start()
        gui.root.mainloop()
        t4.join()

        self.m.printMesgAddStr(" Moving back root folder: ",
                               c.getCyan(), self.app_root.get())
        os.chdir(self.app_root.get())
        os.system('pwd')

        return
    #---------------------------------------------------------------------------
    # [Copy Gain Reference]
    #---------------------------------------------------------------------------
    def Copy_Gain_Reference(self,*args):
        __func__= sys._getframe().f_code.co_name
        #Stop scanning by setting the global flag to False.

        self.m.printMesgStr(   "GainRef copy Function       : ",self.c.get_B_Cyan(),__func__)
        self.m.printMesgAddStr(" Gain Source Directory       : ",
                               self.c.getYellow(),self.GainSourceDir.get())
        self.m.printMesgAddStr(" Gain Target Directory       : ",
                               self.c.getGreen(),self.GainTargetDir.get())
        self.m.printMesgAddStr(" Gain Type to be extracted   : ",
                               self.c.getGreen(),self.GainTargetDir.get())
   

        #Creating the list for the FramesHandler_DriverCode_fileMover_Creator.sh
        in_list = []
        live_ ="sourcedir="+self.GainSourceDir.get()
        in_list.append(live_)
        live_ ="targetdir="+self.GainTargetDir.get()
        in_list.append(live_)
        live_ ="gain_type="+self.c.getGain_Type()
        in_list.append(live_)
            
        CMD = ' '.join(in_list[:])
        m.printMesgAddStr(" sh GainRefHandler_DriverCode_fileMover_Creator.sh ",
                          c.getCyan(), CMD)
        #Sleep a fraction of second to make sure that the files actually gets written
        time.sleep(0.5)
        # Call FramesHandler_DriverCode_fileMover_Creator python creator via OS
        os.system('pwd')
        os.system('sh GainRefHandler_DriverCode_fileMover_Creator.sh %s'%CMD)
        #Sleep a fraction to make sure that file is created and can proceed
        time.sleep(0.5)
        #-----------------------------------------------------------------------
        # Creating the GUI object for the subprocess launch of Gatan_PC_K3_GUIrApp
        #-----------------------------------------------------------------------
        self.q4 = multiprocessing.Queue()
        self.q4.cancel_join_thread() # or else thread that puts data will not term
        gui_GainRef = GainRefGuiApp(self.q4)
        t7 = multiprocessing.Process(target=self.LaunchGainRefCopyHandler,
                                     args=(self.c,self.m, self.q4 ) )
        t7.start()
        gui_GainRef.root.mainloop()
        t7.join()
        
        return
    #---------------------------------------------------------------------------
    # [Killers]
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [killCopyMoveFramesJobs]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # killCopyMoveFramesJobs class method to stop the downloading from remote
    # \param self     The self Object
    # \param c        The DataManage_common object
    # \param m        The messenger class object for logging
    # \param q        The subprocess Queue
    def killAllJobs(self,c,m,q):
        __func__= sys._getframe().f_code.co_name
        q.put("Function                             : "+__func__+"\n")
        q.put("Killing all active jobs: Research Drive Download\n")
        q.put("                         Copy/Move Fractions Handler\n")
        q.put("                         Copy GainRef Handler\n")
        q.put("\n")
        q.put("The Gatan_PC_K3_GUIr Application wil continue running\n")
        q.put("\n")
        q.put("Any of the Downloads, Copy/Move, jobs can be restarted any time.\n")
        os.chdir(self.app_root.get()+os.path.sep+"Scripts"+os.path.sep+"Gatan_PC_K3")
        os.system('sh Kill_Gatan_PC_K3_All_processors.sh')
        return
    #---------------------------------------------------------------------------
    # [killCopyMoveFramesJobs]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # killCopyMoveFramesJobs class method to stop the downloading from remote
    # \param self     The self Object
    # \param c        The DataManage_common object
    # \param m        The messenger class object for logging
    # \param q        The subprocess Queue
    def killCopyMoveFramesJobs(self,c,m,q):
        __func__= sys._getframe().f_code.co_name
        q.put("Function                             : "+__func__+"\n")
        q.put("Killing Copy/Move Fractions jobs: "+__func__+"\n")
        os.chdir(self.app_root.get()+os.path.sep+"Scripts"+os.path.sep+"Gatan_PC_K3")
        os.system('sh Kill_Gatan_PC_K3_fileMover_processors.sh')
        return
    #---------------------------------------------------------------------------
    # [killCopyGainRefJobs]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # killCopyGainRefJobs class method to stop the downloading from remote
    # \param self     The self Object
    # \param c        The DataManage_common object
    # \param m        The messenger class object for logging
    # \param q        The subprocess Queue
    def killCopyGainRefJobs(self,c,m,q):
        __func__= sys._getframe().f_code.co_name
        q.put("Function                             : "+__func__+"\n")
        q.put("Killing Copy Gain Reference jobs: "+__func__+"\n")
        os.chdir(self.app_root.get()+os.path.sep+"Scripts"+os.path.sep+"Gatan_PC_K3")
        os.system('sh Kill_Gatan_PC_K3_gainRefMover_processors.sh')
        return
    #---------------------------------------------------------------------------
    # [killRcloneJobs]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # StopDownloading class method to stop the downloading from remote
    # \param self     The self Object
    # \param c        The DataManage_common object
    # \param m        The messenger class object for logging
    # \param q        The subprocess Queue
    def killRcloneJobs(self,c,m,q):
        __func__= sys._getframe().f_code.co_name
        q.put("Killing Rclone jobs            : "+__func__+"\n")
        os.chdir(self.app_root.get()+os.path.sep+"Scripts"+ \
                 os.path.sep+"Research_drive_upload")
        #q.put(subprocess.check_output(['sh','Kill_rclone_processors.sh'])  )
        os.system('sh Kill_RC_processors.sh')
        os.chdir(self.app_root.get()+os.path.sep+"Scripts"+os.path.sep+"Gatan_PC_K3")
        return
    #---------------------------------------------------------------------------
    # [Stopers]
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [StopAll]
    #---------------------------------------------------------------------------
    def stopAll(self,*args):
        __func__= sys._getframe().f_code.co_name
        #Stop scanning by setting the global flag to False.
        global running
        running = False

        #TODO: here insert code to stop all processes in the GatanFramesGainApp

        self.q5 = multiprocessing.Queue()
        self.q5.cancel_join_thread() # or else thread that puts data will not term
        gui = StopAllGuiApp(self.q5)
        
        try:
            t10 = multiprocessing.Process(target=self.killAllJobs,
                                         args=(self.c, self.m, self.q5 ) )
            t10.start()
            t10.join()
        except:                     # <- naked except is a bad idea
            showerror("Stop All Jobs",
                      "There are no job currently running, No job to "+\
                      "terminate!. Try starting Start any of the running"+\
                      "job for downloading, Copy/Move Fractions or "+\
                      "Copy GainRef process\n")

        # Find which tab the button was pressed from
        self.m.printMesgStr("State Machine: ",self.c.get_B_Cyan()," device_locked")
        self.device.on_event('device_locked')
        self.device.state

        return
    #---------------------------------------------------------------------------
    # [StopCopyMoveFractions]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # StopCopyMoveFractions class method to stop the downloading from remote
    # \param self     The self Object
    # \param *args    The list of arguments from the command prompt
    def StopCopyMoveFractions(self,*args):
        __func__= sys._getframe().f_code.co_name

        script_path = self.app_root.get()+os.path.sep+"Scripts"+os.path.sep+"Gatan_PC_K3"
        self.m.printMesgAddStr(" Moving to script folder: ",
                               c.getCyan(), script_path)

        os.chdir(script_path)
        os.system('pwd')
        
        try:
            t6 = multiprocessing.Process(target=self.killCopyMoveFramesJobs,
                                         args=(self.c, self.m, self.q3 ) )
            t6.start()
            t6.join()
        except:                     # <- naked except is a bad idea
            showerror("Copy/Move Fractions",
                      "There are no job Copy/Move Fractions, No job to "+\
                      "terminate!. Try starting Start Copy/Move Fractions"+\
                      "process\n")

        self.m.printMesgAddStr(" Moving back root folder: ",
                               c.getCyan(), self.app_root.get())
        os.chdir(self.app_root.get())
        os.system('pwd')

        return
    #---------------------------------------------------------------------------
    # [StopCopyMoveFractions]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # StopCopyGainReference class method to stop the downloading from remote
    # \param self     The self Object
    # \param *args    The list of arguments from the command prompt
    def StopCopyGainReference(self,*args):
        __func__= sys._getframe().f_code.co_name

        try:
            t8 = multiprocessing.Process(target=self.killCopyGainRefJobs,
                                         args=(self.c, self.m, self.q4 ) )
            t8.start()
            t8.join()
        except:                     # <- naked except is a bad idea
            showerror("Copy Gain Reference",
                      "There are no job Copy Gain References, No job to "+\
                      "terminate!. Try starting Start Copy Gain Reference"+\
                      "process\n")

        return
    #---------------------------------------------------------------------------
    # [StopDownloading]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # StopDownloading class method to stop the downloading from remote
    # \param self     The self Object
    # \param *args    The list of arguments from the command prompt
    def StopDownloading(self,*args):
        __func__= sys._getframe().f_code.co_name

        try:
            t5 = multiprocessing.Process(target=self.killRcloneJobs,
                                         args=(self.c,self.m, self.q ) )

            t5.start()
            t5.join()
        except:                     # <- naked except is a bad idea
            showerror("Research Drive Download",
                      "There are no Downloading From RD, No job to "+\
                      "terminate!. Try starting Start Downloading From RD"+\
                      "process\n")
        return
    #---------------------------------------------------------------------------
    # [Checkers]
    #---------------------------------------------------------------------------
    def RcloneSetUp(self, c,m,q):
        __func__= sys._getframe().f_code.co_name
        q.put("-------------------------------------------------------------------------\n")
        q.put("RClone config setup \n")
        q.put("-------------------------------------------------------------------------\n")
        q.put("Function                       : "+__func__+"\n")

        q.put("Changing dir for cat_outputfile: "+self.app_root.get()+os.path.sep+"Scripts"+os.path.sep+"Research_drive_upload"+"\n")

        os.chdir(self.app_root.get()+os.path.sep+"Scripts"+os.path.sep+"Research_drive_upload")
        #q.put(subprocess.check_output(['ls'])  )
        q.put(subprocess.check_output(['pwd'])  )
        #time.sleep(2)

        q.put("-------------------------------------------------------------------------\n")
        q.put("Launching Rclone setup window...\n")
        q.put("-------------------------------------------------------------------------\n")
        os.system('xterm -bg black -fg white -geometry 180x40+50+50 -e \"rclone-1.47 config\"&')
        os.chdir(self.app_root.get()+os.path.sep+"Scripts"+os.path.sep+"Gatan_PC_K3")

        return
    #---------------------------------------------------------------------------
    # [setupRcloneConfig]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # setupRcloneConfig class method to check the inputs and then run the application
    # \param self     The self Object
    # \param *args    The list of arguments from the command prompt
    def setupRcloneConfig(self,*args):
        __func__= sys._getframe().f_code.co_name
        #-----------------------------------------------------------------------
        # Starting the statemachine
        #-----------------------------------------------------------------------
        self.q6 = multiprocessing.Queue()
        self.q6.cancel_join_thread() # or else thread that puts data will not term
        gui = RCloneSetupGuiApp(self.q6)
        t11 =  multiprocessing.Process(target=self.RcloneSetUp,
                                       args=(self.c,self.m, self.q6 ) )
        self.m.printMesgStr("RClone setup: ",
                            self.c.get_B_Green(),"Starts...")
        t11.start()
        gui.root.mainloop()
        t11.join()

        return
    #---------------------------------------------------------------------------
    # [checkInputsAndLaunchDataManagerApp]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # checkInputsAndLaunchDataManagerApp class method to check the inputs and then run the application
    # \param self     The self Object
    # \param *args    The list of arguments from the command prompt
    def checkInputsAndLaunchDataManagerApp(self,*args):
        __func__= sys._getframe().f_code.co_name
        #-----------------------------------------------------------------------
        # Starting the statemachine
        #-----------------------------------------------------------------------
        self.m.printMesgStr("State Machine: ",self.c.get_B_Green(),
                            " Start_computation")
        self.device.on_event('Start_computation')
        self.device.state
        #-----------------------------------------------------------------------
        # Check all of the inputs before launching driver code Poolifier
        #-----------------------------------------------------------------------
        #Changing dir to the application root directory to make sure that we are
        #at the right place this should already be from the down
        os.chdir(self.app_root.get())
        #-----------------------------------------------------------------------
        # Creating the GUI object for the subprocess launch of DataManagerApp
        #-----------------------------------------------------------------------
        self.q2 = multiprocessing.Queue()
        self.q2.cancel_join_thread() # or else thread that puts data will not term
        gui = DataManagerGuiApp(self.q2)
        t3 = multiprocessing.Process(target=self.LaunchDataManagerApp,
                                     args=(self.c,self.m, self.q2 ) )
        t3.start()
        gui.root.mainloop()
        t3.join()
        #-----------------------------------------------------------------------
        # Starting the StateMachine
        #-----------------------------------------------------------------------
        #Enable scanning by setting the global flag to True.
        global running
        running = True
        #self.scanning()

        #declared but not needed just yet may need to turn this into a self
        showGUI = True

        return
    #---------------------------------------------------------------------------
    # [LaunchGainRefCopyHandler]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # LaunchGainRefCopyHandler class method to launch the DataManager application
    # \param self     The self Object
    # \param c        DataManageApp common object
    # \param m        The mnessdeanger class for output messaging and loggin
    # \param q2       The multiprocessing Queue object
    def LaunchGainRefCopyHandler(self,c,m,q4):
        __func__= sys._getframe().f_code.co_name
        q4.put("Function                             : "+__func__+"\n")
        q4.put("Launching GainRef copy with gain type "+ \
               "gain_type=[{x0.m0, x0.m1, x1.m0, x1.m1}|x0.m0}] handler...\n")
        #FramesHandler_DriverCode_fileMover.py
        os.system('python3 GainRefHandler_DriverCode_fileMover.py')
        return
    #---------------------------------------------------------------------------
    # [LaunchFramesCopyMoveWarpHandler]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # LaunchFramesCopyMoveWarpHandler class method to launch the DataManager application
    # \param self     The self Object
    # \param c        DataManageApp common object
    # \param m        The mnessdeanger class for output messaging and loggin
    # \param q2       The multiprocessing Queue object
    def LaunchFramesCopyMoveWarpHandler(self,c,m,q3):
        __func__= sys._getframe().f_code.co_name
        q3.put("Function                             : "+__func__+"\n")
        q3.put("Launching Frames [{copy,move}|copy] "+ \
               "multiThreads=[{yes,no|no}] handler...\n")
        os.system('python3 FramesHandler_DriverCode_fileMover.py')
        return
    #---------------------------------------------------------------------------
    # [LaunchDataManagerApp]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # LaunchDataManagerApp class method to launch the DataManager application
    # \param self     The self Object
    # \param c        DataManageApp common object
    # \param m        The mnessdeanger class for output messaging and loggin
    # \param q2       The multiprocessing Queue object
    def LaunchDataManagerApp(self,c,m,q2):
        __func__= sys._getframe().f_code.co_name
        q2.put("Function                       : "+__func__+"\n")
        q2.put("Launching DataManager App...\n")
        os.system('python3 DataManage_preProcessing.py')
        return
    #---------------------------------------------------------------------------
    # [showAbout] method for quiting the application
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # GatanFramesGainApp class method to show the about popup
    # \param self     The self Object
    def showAbout(self):
        __func__= sys._getframe().f_code.co_name
        showinfo("About LDMaP-APP (Live Data Management and Processing Application for EM-Microscope data) Data Handler",
        ("This is LDMaP-APP (Live Data Management and Processing Application for EM-Microscope data) Data Handler v."+version+".\n\n"
         "If you use DataManage in your work, please cite the following paper:"\
         "This package is released under the Creative Commons " \
         "Attribution-NonCommercial-NoDerivs CC BY-NC-ND License " \
         "(http://creativecommons.org/licenses/by-nc-nd/3.0/)\n\n"
         "Please send comments, suggestions, and bug reports to " \
         "f.d.r.bonnet@biology.leidenuniv.nl or "\
         "fbonnet08@gmail.com"))
    #---------------------------------------------------------------------------
    # [showDocumentation] method for quiting the application
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # GatanFramesGainApp class method to show the documentation popup
    # \param self     The self Object
    def showDocumentation(self):
        __func__= sys._getframe().f_code.co_name
        showinfo("DataManage Documentation",
            "Please download the manual at https://sourceforge.net/projects/..... To be added later")

#-------------------------------------------------------------------------------
# Main code of the application DataManage
#-------------------------------------------------------------------------------
##\brief Python3 method.
# GatanFramesGainApp application main
if __name__ == '__main__':

    __func__= sys._getframe().f_code.co_name
    global version
    #printing the header
    version = DataManage_common.DataManage_version() # DataManage_version()
    #instantiating the common class
    c = DataManage_common.DataManage_common()# DataManage_common()
    #instantiating messaging class
    logfile = c.getLogGatan_PC_K3_GUI()  #getting the name of the global log file
    m = utils.messageHandler.messageHandler(logfile=logfile)# messageHandler(logfile = logfile)
    # printing the header of Application
    DataManage_header.print_Gatan_PC_K3_header(common=c, messageHandler=m) #print_Gatan_PC_K3_header(common=c,messageHandler=m)
    # getting stuff on screen
    m.printMesg("Message class instantiated, log: "+ c.getLogfileName())
    m.printMesg("Common class has been instantiated.")
    m.printCMesgCVal("DataManage Version  : ", c.get_B_White(),version,
                     c.get_B_Green())
    m.printMesgStr("Operating System:", c.get_B_Green(),system)
    m.printMesgStr("OS time stamp   :", c.get_B_Yellow(),sysver)
    m.printMesgStr("Release         :", c.get_B_Magenta(),release)
    m.printMesgStr("Kernel          :", c.get_B_Cyan(),platform)
    m.printMesgStr("Node            :", c.get_B_Yellow(),node)
    m.printMesgStr("Processor type  :", c.get_B_Magenta(),processor)
    m.printMesgStr("CPU cores count :", c.get_B_Green(),cpu_count)

    args = docopt(__doc__, version=version)

    if args['--noguiSplit'] == False:

        m.printMesgStr("Checking if     :",c.getCyan(),'target_* exists')
        if os.path.isfile('target_*'):
            m.printMesgAddStr(" target_* exists :",c.getRed(),'target_*')
            m.printMesgAddStr(" deleting       :",
                              c.getYellow(),'target_* files')
            os.system('rm target_*')
            m.printMesgAddStr(" status         :",c.get_B_Green(),'done.')
        else:
            m.printMesgAddStr(" target_* exists :",c.getGreen(),'NO')
            m.printMesgAddStr(" status          :",c.get_B_Green(),'Proceed...')
        
        # Create root window and initiate GUI interface
        root = tk.Tk()
        datamanageapp = GatanFramesGainApp(root, c=c, m=m)
        root.mainloop()
    else:
        #--use_gpu=USE_GPU
        if args['--use_gpu'] == "yes":
            use_gpu = True
            c.setUse_gpu(use_gpu)
        else:
            use_gpu = False
            c.setUse_gpu(use_gpu)

        #--set_gpu=SET_GPU
        if args['--set_gpu']:
            set_gpu = int(args['--set_gpu'])
            c.setSet_gpu(set_gpu)
        else:
            set_gpu = 0
            c.setSet_gpu(set_gpu)

        # --doBenchMarking
        if args['--doBenchMarking'] == True:
            doBenchMarking = True
            c.setBench_cpu(doBenchMarking)
            showGUI = False
            c.setShow_gui(showGUI)
        else:
            doBenchMarking = False
            showGUI = True
            c.setShow_gui(showGUI)

        # Call Gatan_PC_K3_GUI_live-Preprocessing
        if args['--noguiSplit'] == True:
            m.printMesg("Starting the Gatan_PC_K3_GUI...")
            '''
            Gatan_PC_K3_GUI(input_data_path   = input_data_path,
                                          input_gainmrc   = input_gainmrc,

                                          etc...
                                          
                                          doBenchMarking   = doBenchMarking,
                                          chimeraBin       = chimeraBin,
                                          showGUI          = showGUI,
                                          use_gpu          = c.getUse_gpu(),
                                          set_gpu          = c.getSet_gpu(),
                                          lib              = c.getLibfileName() )
            '''
        else:
            rc = c.get_RC_FAIL()
            #final ovrall return code
            getFinalExit(c,m,rc)
            m.printCMesg("Invalid entry for Gatan_PC_K3_GUI: Gatan_PC_K3_GUI.py!!",c.get_B_Red())
#---------------------------------------------------------------------------
# end of Gatan_PC_K3_GUI application
#---------------------------------------------------------------------------
