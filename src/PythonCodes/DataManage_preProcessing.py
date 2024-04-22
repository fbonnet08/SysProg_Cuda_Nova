'''!\file
   -- DataManage_preProcessing: (Python3 code) is a Python (NumPy/SciPy) application with a Tkinter GUI.
It is a software package for handling the data from EM-microscope and pre processing studied in structural biology, primarily electron cryo-microscopy (cryo-EM).

Please find the manual at https://sourceforge.net/projects/ldmap/

Author:
  F.D.R. Bonnet

This package is released under the Creative Commons
Attribution-NonCommercial-NoDerivs CC BY-NC-ND License
(http://creativecommons.org/licenses/by-nc-nd/3.0/)

Usage:
  DataManage_preProcessing.py [--nogui]
                              [--app_root=APP_ROOT]
                              [--data_path=DATA_path]
                              [--projectName=PROJECTNAME]
                              [--gainmrc=GAINMRC]
                              [--targetdir=TARGETDIR]
                              [--software=SOFTWARE]
                              [--movies_ext=MOVIES_EXT]
                              [--file_frame_copy_move_handler=FILE_FRAME_COPY_MOVE_HANDLER]
                              [--super_res=SUPER_RES]
                              [--motioncorr_dir=MOTIONCORR_DIR]
                              [--nframes=NFRAMES]
                              [--sph_abe=SPH_ABE]
                              [--amp_con=AMP_CON]
                              [--sze_pwr_spc=SZE_PWR_SPC]
                              [--minres=MINRES]
                              [--maxres=MAXRES]
                              [--mindef=MINDEF]
                              [--maxdef=MAXDEF]
                              [--defstep=DEFSTEP]
                              [--astigpenaltyover=ASTIGPENALTYOVER]
                              [--fm_dose=FM_DOSE]
                              [--bfactor=BFACTOR]
                              [--binning_factor=BINNING_FACTOR]
                              [--gain_rotation=GAIN_ROTATION]
                              [--gain_flip=GAIN_FLIP]
                              [--ctffind_dir=CTFFIND_DIR]
                              [--live_process=LIVE_PROCESS]
                              [--use_MotionCorr=USE_MOTIONCORR]
                              [--use_CTFfind=USE_CTFFIND]
                              [--file_pre_handler=F_PRE_HDLR]
                              [--file_organise_dir=FILE_ORGANISE_DIR]
                              [--file_pst_handler=FILE_PST_HANDLER]
                              [--use_gpu=USE_GPU]
                              [--gpu_list=GPU_LIST]
                              [--poolify=POOLIFY]
                              [--doBenchMarking]

NOTE: INPUT(s) is/are mandatory

Arguments:
  INPUTS                      Input For commputation is needed as shown below

Example:
   python3 DataManage_preProcessing.py --app_root=/home/frederic/Leiden/SourceCode/DataManage_project --projectName=20210401_2001-997_EPU-SP_Fred --data_path=/DriveA_SSD/scratch/frederic/ResearchDriveDownload --gainmrc= --targetdir=/DriveA_SSD/scratch/frederic/TargetDir --poolify=yes --software=EPU-2.10.5 --use_gpu=yes --gpu_list=0:1:2:3 --super_res=yes --fm_dose=1.0 --bfactor=150 --binning_factor=1 --gain_rotation=2 --gain_flip=2 --movies_ext=mrc --live_process=yes --use_MotionCorr=1 --use_CTFfind=1 --file_frame_copy_move_handler=move --file_pre_handler=preprocess --file_organise_dir=or --file_pst_handler=extract_data --motioncorr_dir=/opt/Devel_tools/MotionCor2_v1.3.1/MotionCor2_v1.3.1-Cuda102 --ctffind_dir=/opt/Devel_tools/CTFfind/ctffind-4.1.18/bin_compat/ctffind --doBenchMarking

Options:
 --nogui                        un DataManage from command-line mode.
 --app_root=APP_ROOT            Path to root of application 
 --data_path=DATA_PATH          Path to where the data is located
 --projectName=PROJECTNAME      Name of the project in consideration
 --gainmrc=GAINMRC              Gain reference to be used for this run
 --targetdir=TARGETDIR          Path to where data is to be processed
 --software=SOFTWARE            Software to be used {[EPU-2.10.5, SerialEM,TOMO4,TOMO5]|EPU-2.10.5}
 --movies_ext=MOVIES_EXT        Movie extension {[mrc,tiff]|mrc}
 --file_frame_copy_move_handler=FILE_FRAME_COPY_MOVE_HANDLER   Handler for data movement {[copy,move,rsync_copy,rsync_move]|copy}
 --super_res=SUPER_RES          Super Resolution {[yes,no]|yes}
 --motioncorr_dir=MOTIONCORR_DIR  Path to Motion correction
 --nframes=NFRAMES              [50] TODO: Include number frames unblur
 --sph_abe=SPH_ABE              [2.7] Sphererical Aberration
 --amp_con=AMP_CON              [0.10] Ampl constrast (from 0.07 06Nov21)
 --sze_pwr_spc=SZE_PWR_SPC      [512] Size of prower spectrum 
 --minres=MINRES                [30.0] Minimum resolution
 --maxres=MAXRES                [5.0] Maximum resolution
 --mindef=MINDEF                [5000] Minimum defocus "3100.0"06Nov21
 --maxdef=MAXDEF                [50000] Maximum defocus "6900.0"06Nov21
 --defstep=DEFSTEP              [500.0] Defocus search step
 --astigpenaltyover=ASTIGPENALTYOVER  [1500.0] Expected \(tolerated\) astig
 --fm_dose=FM_DOSE              Frames Dose {[0-inf]|1.0}
 --bfactor=BFACTOR              B-Factor applied to micrographs {[0-inf]|150}
 --binning_factor=BINNING_FACTOR  Binning factor applied or downscale {[0-inf]|1} 
 --gain_rotation=GAIN_ROTATION    Gain rotation applied {[0-inf]|2}
 --gain_flip=GAIN_FLIP            Gain flipping applied {[0-inf]|2}
 --ctffind_dir=CTFFIND_DIR        Path to CTF Find (v4.1.18 supported and Recommended)
 --live_process=LIVE_PROCESS      Live processing Gain flipping applied {[yes,no]|yes}
 --use_MotionCorr=USE_MOTIONCORR  Use Motion Correction {[yes,no]|yes}
 --use_CTFfind=USE_CTFFIND        Use CTFFind {[yes,no]|yes}
 --file_pre_handler=F_PRE_HDLR            Handler for preprocessing data or not
 --file_organise_dir=FILE_ORGANISE_DIR    Handler for organizing the files
 --file_pst_handler=FILE_PST_HANDLER      Handler for data extraction
 --use_gpu=USE_GPU            Use GPUs {[yes,no]|yes}
 --gpu_list=GPU_LIST          List of Gpus to be used {[{0:1:2:3}]|0}
 --poolify=POOLIFY            Create the pool {[yes,no]|yes}
 --doBenchMarking             Run DataManage in BenchMarking mode, if combined with   
 --help -h                    Show this help message and exit.
 --version                    Show version.
'''
#System tools
import sys
import tkinter as tk
import multiprocessing
import subprocess
from queue import Empty, Full
#from multiprocessing import Pool
#GUI imports
import tkinter
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
#appending the utils path
from DataManage_common import *
#platform, release  = whichPlatform()
sysver, platform, system, release, node, processor, cpu_count = whichPlatform()
#TODO: If statement will be removed after checking
sys.path.append(os.path.join(os.getcwd(), '.'))
sys.path.append(os.path.join(os.getcwd(), 'utils'))
sys.path.append(os.path.join(os.getcwd(), 'utils','StateMachine'))
#application imports
from docopt import docopt
#from DataManage_fileIO import *
from DataManage_header import print_header
#from messageHandler import *
import utils.messageHandler
from DataManage_descriptors import *  #bring in the documentation
#from simple_device import SimpleDevice
import utils.StateMachine.simple_device
import utils.CTFGraphLauncher
import utils.StopWatch
import utils.progressBar
import DataManage_config
import utils.Command_line
#import gui_ioChannel
################################################################################
#                                                                              
#                                                                              
#           DataManage application to manage and real process data on CPU-GPU   
#                          for given data sets from EM-microscope
#                                                                              
################################################################################
running = True  # Global flag
#-------------------------------------------------------------------------------
# [class] PostProcessingGuiApp
#-------------------------------------------------------------------------------
##\brief Python3 method.
# PostProcessingGuiApp application helper class
class PostProcessingGuiApp(object):
    def __init__(self,q):
        self.root = tk.Tk()
        self.root.geometry('500x250')
        self.root.title("DataManager Application (Post-Processing Handler)")
        self.text_wid = tk.Text(self.root,height=200,width=200)
        self.text_wid.pack(expand=1,fill=tk.BOTH)
        self.root.after(200,self.CheckQueuePoll(q))
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
# [class] PostProcessingGuiApp
#-------------------------------------------------------------------------------
##\brief Python3 method.
# PostProcessingGuiApp application helper class
class ShowResultsGuiApp(object):
    def __init__(self,q, c, m):
        self.c = c
        self.m = m
        self.root = tkinter.Tk() #tk.Tk()
        self.root.geometry('800x250')
        self.root.title("LDMaP-APP - Show Results Pre-Processing Handler")
        self.text_wid = tk.Text(self.root,height=100,width=600)
        #-----------------------------------------------------------------------
        #-----------------------------------------------------------------------
        # Set output canvas size setting up the window environemt.
        #-----------------------------------------------------------------------
        vscrollbar = tkinter.Scrollbar(self.root, orient=tkinter.VERTICAL)
        hscrollbar = tkinter.Scrollbar(self.root, orient=tkinter.HORIZONTAL)

        canvas = tkinter.Canvas(bg='lightblue', yscrollcommand=vscrollbar.set, xscrollcommand=hscrollbar.set)
        self.root.geometry('1580x260') #TODO: determine the geometry according to the image generated
        #self.root.geometry(geometry)
        frame = tkinter.Frame(canvas)

        vscrollbar.config(command=canvas.yview)
        vscrollbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)

        hscrollbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
        hscrollbar.config(command=canvas.xview)

        photo_boxplot = tkinter.PhotoImage(file='/home/frederic/Leiden/SourceCode/DataManage_project/doc/Application_Pics/boxplot_explanation.png')
        canvas_width = photo_boxplot.width()
        canvas_height = photo_boxplot.height()
        canvas.create_window(canvas_width,canvas_height,window=frame, anchor='nw')

        #os.system('pwd')
        #canvas.create_image(0,0,image='/home/frederic/Leiden/SourceCode/DataManage_project/doc/Application_Pics/boxplot_explanation.png', anchor=tkinter.NW)

        #-----------------------------------------------------------------------
        file = self.c.getTargetdir()+os.path.sep+self.c.getProjectName()+"_Pool"+os.path.sep+self.c.getProjectName()+"_Pool.asc"
        self.pool_dir = self.c.getTargetdir() + os.path.sep + self.c.getProjectName()
        self.asc_file = self.c.getProjectName() + "_Pool.asc"
        self.asc_file_full_path = self.pool_dir+os.path.sep+self.asc_file
        #-----------------------------------------------------------------------
        self.text_wid.pack(expand=1,fill=tk.BOTH)
        self.root.after(200,self.CheckQueuePoll(q))
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
# DataManage application
#-------------------------------------------------------------------------------
##\brief Python3 method.
# DataManagePreProcApp application class
class DataManagePreProcApp(object):
    """
    GUI Tkinter class for DataManage
    """
    ##\brief Python3 method.
    #DataManagePreProcApp application class constructor
    def __init__(self, parent):

        #function stamp
        __func__= sys._getframe().f_code.co_name
        #instantiating the common class
        self.c = DataManage_common()
        #instantiating messaging class
        logfile = self.c.getLogfileName()  #getting the name of the global log file
        self.m = utils.messageHandler.messageHandler(logfile = logfile)

        #Starting the StateMachine
        self.m.printMesg("State Machine SimpleDevice class instantiated, log: "+ c.getLogfileName())
        self.device = utils.StateMachine.simple_device.SimpleDevice(c, m)

        self.m.printMesgStr("State Machine: ",c.get_B_Green()," device_locked")
        self.device.on_event('device_locked')
        self.device.state

        # General Settings
        self.parent = parent
        self.parent.title("LDMaP-APP (Live Data Management and Processing Application for EM-Microscope data) Processing Handler v" + version)
        self.parent.option_add('*tearOff', False)

        #self.parent.geometry('500x250')

        self.myStyle = ttk.Style()
        self.myStyle.configure('DataManageApp.TButton', foreground='blue4',
                               font='Helvetica 9 bold')
        self.myStyle.configure('DataManageAppRed.TButton', foreground='red4',
                               font='Helvetica 9 bold')
        self.myStyle.configure('DataManageAppStd.TButton',
                               font='Helvetica 9')

        self.font_size_headers =  "Helvetica 10 bold"
        self.font_size_all_text = "Helvetica 10"
        self.font_size_all_text_bold = "Helvetica 10 bold"

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
        self.helpMenu.add_command(label="About DataManage",
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
        '''
        self.app_root         = tk.StringVar(value="/home/frederic/Leiden/SourceCode/DataManage_project")
        self.projectName      = tk.StringVar(value="supervisor_20200213_161112_k3_apof_benchmark")
        self.data_path        = tk.StringVar(value="/Drive_C/frederic/data/ApofTestProject/Contracted/") #"../Micrographs/*.mrc")
        self.gainRef          = tk.StringVar(value="/Drive_C/frederic/data/gainref_m0_2020-02-13--17-21-06.mrc")
        self.targetDirectory  = tk.StringVar(value="/Drive_C/frederic/data/LiveAppOut/DataManage_live_out")
        self.MotionCorrDir    = tk.StringVar(value="/opt/Devel_tools/MotionCor2_v1.3.1/")
        self.CTFfindDir       = tk.StringVar(value="/opt/Devel_tools/CTFfind/ctffind-4.1.18/bin_compat/")
        '''
        #[NeCEN]
        self.app_root         = tk.StringVar(value=DataManage_config.APP_ROOT)#"/home/frederic/Leiden/SourceCode/DataManage_project")
        self.projectName      = tk.StringVar(value="20210401_2001-997_EPU-SP_Fred")#20201015_1910-272_EPU-SP_Ludo
        self.data_path        = tk.StringVar(value="/DriveA_SSD/scratch/frederic/ResearchDriveDownload") #/data/local/frederic/ResearchDriveDownload#/data/krios1buffer#"
        self.gainRef          = tk.StringVar(value="")#/data/krios1buffer/gainrefs/2020-09-08--09-43-44_K3-20050033_Gain_Ref._x1.m1.mrc
        self.targetDirectory  = tk.StringVar(value="/DriveA_SSD/scratch/frederic/TargetDir")#/data/local/frederic/TargetDir
        self.MotionCorrDir    = tk.StringVar(value=DataManage_config.MOTIONCORR_DIR_DFLT)#"/opt/Devel_tools/MotionCor2_v1.3.1/MotionCor2_v1.3.1-Cuda102")
        self.CTFfindDir       = tk.StringVar(value=DataManage_config.CTFFIND_DIR_DFLT)#"/opt/Devel_tools/CTFfind/ctffind-4.1.18/bin_compat/ctffind")

        self.relionPath       = self.targetDirectory.get()+os.path.sep+     \
                                self.projectName.get()+"_Pool"+os.path.sep+ \
                                "Relion"+os.path.sep+                       \
                                "app_ctf_corrected_micrographs.star"

        self.framesAutopickIn = tk.StringVar(value=self.relionPath)
        #Preprocessing variables
        self.doBenchMarking = tk.BooleanVar(value=True)
        self.use_gpu        = tk.BooleanVar(value=True)
        self.gpu_list       = tk.StringVar(value="0:1:2:3")
        self.poolify        = tk.BooleanVar(value=False)
        #Preprocessing variables
        self.use_gpu_cl23d  = tk.BooleanVar(value=False)
        self.gpu_list_cl23d = tk.StringVar(value="0:1:2:3")
        self.doBenchMarking_cl23d = tk.BooleanVar(value=False)
        #Software to be used ComboBox
        self.software_SP    = tk.StringVar(value="EPU-2.10.5")
        self.software_TOMO  = tk.StringVar(value="TOMO4")
        self.super_res      = tk.BooleanVar(value=True)  #[{yes}|no]--->checkBox

        self.nframes        = tk.StringVar(value="50")   #Incl.? Nframes unblur
        self.sph_abe        = tk.StringVar(value="2.7")      #Sph. Aberration
        self.amp_con        = tk.StringVar(value="0.10")     #Ampl constrast
        self.sze_pwr_spc    = tk.StringVar(value="512")      #Size power Spect.
        self.minres         = tk.StringVar(value="30.0")     #Min resolution
        self.maxres         = tk.StringVar(value="5.0")      #Max resolution
        self.mindef         = tk.StringVar(value="5000")     #Min defocus
        self.maxdef         = tk.StringVar(value="50000")    #Max defocus
        self.defstep        = tk.StringVar(value="500.0")    #Defoc search step
        self.astigpenaltyover = tk.StringVar(value="1500.0") #Expected  astig

        self.fm_dose        = tk.StringVar(value="1.0") #--fm_dose=float
        self.bfactor        = tk.StringVar(value="150") #--bfactor=float
        self.binning_factor = tk.StringVar(value="1")   #--binning_factor=float
        self.gain_rotation  = tk.StringVar(value="2")   #--gain_rotation=int
        self.gain_flip      = tk.StringVar(value="2")   #--gain_flip=int
        # Particle extraction default values
        self.particle_min_diam_var = tk.StringVar(value="160")
        self.particle_max_diam_var = tk.StringVar(value="250")
        self.particle_box_sze      = tk.StringVar(value="400")
        #Class2D Launching input
        self.n_classes2D    = tk.StringVar(value="15")
        self.n_iterClss2D   = tk.StringVar(value="25")
        self.n_mskDClss2D   = tk.StringVar(value="200")
        self.n_poolClss2D   = tk.StringVar(value="3")
        #Class3D Launching input
        self.n_classes3D    = tk.StringVar(value="1")
        self.n_iterClss3D   = tk.StringVar(value="5")
        self.n_mskDClss3D   = tk.StringVar(value="200")
        self.n_poolClss3D   = tk.StringVar(value="3")
        self.symmetry3D     = tk.StringVar(value="C1")
        #Computational specs default variable 
        self.n_mpi_proc     = tk.StringVar(value="32")
        # Class2D
        self.n_mpi_proc_cl2d      = tk.StringVar(value="5")
        self.n_thrds_cl2d         = tk.StringVar(value="4")
        # Class3D
        self.n_mpi_proc_cl3d      = tk.StringVar(value="5")
        self.n_thrds_cl3d         = tk.StringVar(value="4")
        #Movie Extension #--movies_ext=tiff--->radioButton
        self.movies_ext           = tk.StringVar(value="mrc")
        #--live_process=yes--->checkBox
        self.live_process_hdl     = tk.BooleanVar(value=True)
        self.live_process         = "Null"
        #--use_unblur=0---> checkBox
        self.use_unblur_hdl       = tk.BooleanVar(value=False)
        self.use_unblur           = 0
        #--use_MotionCorr=0---> checkBox
        self.use_MotionCorr_hdl   = tk.BooleanVar(value=True)
        self.use_MotionCorr       = 0
        #--use_CTFfind=0---> checkBox
        self.use_CTFfind_hdl      = tk.BooleanVar(value=True)
        self.use_CTFfind          = 0
        #--file_pre_handler=preprocess--->CheckBox
        self.file_pre_handler_hdl = tk.BooleanVar(value=True)
        self.file_pre_handler     = tk.StringVar(value="preprocess")
        #--file_organise_dir=re---> Combo box
        self.file_organise_dir    = tk.StringVar(value="or")
        #--file_pst_handler=extract_data---> combo box
        self.file_pst_handler     = tk.StringVar(value="extract_data")
        #--file_frame_copy_move_handler=copy---> radioButton
        self.file_frame_copy_move_handler = tk.StringVar(value="copy")
        # Create split volume input frame
        self.splitframe = ttk.Frame(self.nb)
        #self.preprocframe = ttk.Frame(self.nb)

        self.splitframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.splitframe.columnconfigure(0, weight=1)
        self.splitframe.rowconfigure(   0, weight=1)

        #adding the frames to the main frame
        self.nb.add(self.splitframe, text='Micrographs and Gain Refs Input', underline=0, padding=10)
        #-----------------------------------------------------------------------
        # self.splitframe Frame
        #-----------------------------------------------------------------------
        #########################
        # Required Inputs
        #########################
        # ROW 0
        row = 0
        ttk.Label(self.splitframe, text="Required Inputs", font=self.font_size_headers).grid(column=1, row=row, sticky=tk.W)

        text_width = 100
        # ROW 1
        row = 1
        ttk.Label (self.splitframe, text="Application Root:", foreground="blue", font=self.font_size_all_text).grid(column=1 , row=row, sticky=tk.E)
        ttk.Entry (self.splitframe, width=text_width, textvariable=self.app_root, foreground="green", font=self.font_size_all_text).grid(column=2 , columnspan=9, row=row, sticky=(tk.W,tk.E) )
        ttk.Button(self.splitframe, text="Set LDMaP-App path", style='DataManageAppStd.TButton', command=(lambda: self.load_directory(self.app_root, "app_root"))) .grid(column=11, row=row, sticky=(tk.W,tk.E))#tk.W)
        self.c.setApp_root(self.app_root.get())

        # ROW 2
        row = 2
        ttk.Label(self.splitframe, text="Project Name:", foreground="blue", font=self.font_size_all_text).grid(column=1, row=row, sticky=tk.E)
        ttk.Entry(self.splitframe, width=text_width, textvariable=self.projectName, foreground="green", font=self.font_size_all_text).grid(column=2, columnspan=9,row=row, sticky=(tk.W,tk.E))
        ttk.Button(self.splitframe, text="Set Project Name", style='DataManageAppStd.TButton', command=(lambda: self.projectNameEntryActivator(self.projectName))) .grid(column=11, row=row, sticky=(tk.W,tk.E))#tk.W)
        self.c.setProjectName(self.projectName.get())

        # ROW 3
        row = 3
        ttk.Label(self.splitframe, text="Data Path:", foreground="blue", font=self.font_size_all_text).grid(column=1, row=row, sticky=tk.E)
        ttk.Entry(self.splitframe, width=text_width, textvariable=self.data_path, foreground="green", font=self.font_size_all_text).grid(column=2, columnspan=9,row=row, sticky=(tk.W,tk.E))
        ttk.Button(self.splitframe, text="Set Data Path", style='DataManageAppStd.TButton', command=(lambda: self.load_directory(self.data_path, "data_path"))).grid(column=11, row=row, sticky=(tk.W,tk.E))#tk.W)
        self.c.setData_path(self.data_path.get())

        # ROW 4
        row = 4
        ttk.Label(self.splitframe, text="Gain Refs  :", foreground="blue", font=self.font_size_all_text).grid(column=1, row=row, sticky=tk.E)
        ttk.Entry(self.splitframe, width=text_width, textvariable=self.gainRef, foreground="green", font=self.font_size_all_text).grid(column=2, columnspan=9, row=row, sticky=(tk.W,tk.E))
        ttk.Button(self.splitframe, text="Load Gain References", style='DataManageAppStd.TButton', command=(lambda: self.load_file(self.gainRef, "gainRef"))).grid(column=11, row=row, sticky=(tk.W,tk.E))#tk.W)
        self.c.setGainmrc(self.gainRef.get())

        # ROW 5
        row = 5
        ttk.Label(self.splitframe, text="Target Directory:", foreground="blue", font=self.font_size_all_text).grid(column=1, row=row, sticky=tk.E)
        ttk.Entry(self.splitframe, width=text_width, textvariable=self.targetDirectory, foreground="green", font=self.font_size_all_text).grid(column=2, columnspan=9, row=row, sticky=(tk.W, tk.E))
        ttk.Button(self.splitframe, text="Set Target Directory", style='DataManageAppStd.TButton', command=(lambda: self.load_directory(self.targetDirectory, "targetDirectory"))).grid(column=11, row=row, sticky=(tk.W,tk.E))#tk.W)
        self.c.setTargetdir(self.targetDirectory.get())

        # ROW 6
        row = 6
        ttk.Label(self.splitframe, text="Single Particle:", foreground="magenta4", font=self.font_size_all_text_bold).grid(column=1, row=row, sticky=tk.E)
        #SP software
        softwareSP=["EPU-2.10.5", "SerialEM"]
        # Radiobutton callback function
        def radCall_SP():
            radSel_SP=radVar_SP.get()
            if   radSel_SP == 0:
                self.software_SP = "EPU-2.10.5"
                self.c.setSoftware(self.software_SP)
                self.m.printMesgAddStr(" software_SP = ",
                                       self.c.getYellow(),self.c.getSoftware())
            elif radSel_SP == 1:
                self.software_SP = "SerialEM"
                self.c.setSoftware(self.software_SP)
                self.m.printMesgAddStr(" software_SP = ",
                                       self.c.getCyan(),self.c.getSoftware())

        radVar_SP = tk.IntVar()
        # Next we are selecting a non-existing index value for radVar_SP.
        radVar_SP.set(99)

        # Now we are creating all three Radiobutton widgets within one loop.
        curRad = tk.Radiobutton(self.splitframe, text=softwareSP[0], variable=radVar_SP, value=0, command=radCall_SP)
        curRad.grid(column=2, row=row, sticky=tk.W)
        curRad = tk.Radiobutton(self.splitframe, text=softwareSP[1], variable=radVar_SP, value=1, command=radCall_SP)
        curRad.grid(column=3, row=row, sticky=tk.W)

        ttk.Label(self.splitframe, text="Tomography:", foreground="magenta4", font=self.font_size_all_text_bold).grid(column=4, row=row, sticky=tk.E)
        #TOMO software
        softwareTOMO=["TOMO4", "TOMO5", "SerialEM"]
        # Radiobutton callback function
        def radCall_TOMO():
            radSel_TOMO=radVar_TOMO.get()
            if   radSel_TOMO == 0:
                self.software_TOMO = "TOMO4"
                self.c.setSoftware(self.software_TOMO)
                self.m.printMesgAddStr(" software_TOMO = ",
                                       self.c.getRed(),self.c.getSoftware())
            elif radSel_TOMO == 1:
                self.software_TOMO = "TOMO5"
                self.c.setSoftware(self.software_TOMO)
                self.m.printMesgAddStr(" software_TOMO = ",
                                       self.c.getRed(),self.c.getSoftware())
            elif radSel_TOMO == 2:
                self.software_TOMO = "SerialEM"
                self.c.setSoftware(self.software_TOMO)
                self.m.printMesgAddStr(" software_TOMO = ",
                                       self.c.getRed(),self.c.getSoftware())
        radVar_TOMO = tk.IntVar()
        # Next we are selecting a non-existing index value for radVar_TOMO.
        radVar_TOMO.set(98)

        # Now we are creating all three Radiobutton widgets within one loop.
        '''
        for col in range(len(softwareTOMO)):
            set_col = col+2
            curRad = 'rad' + str(col)
            curRad = tk.Radiobutton(self.splitframe, text=softwareTOMO[col], variable=radVar_TOMO, value=col, command=radCall_TOMO)
            curRad.grid(column=set_col, row=7, columnspan=9, sticky=tk.W)#columnspan=3,
        '''
        curRad = tk.Radiobutton(self.splitframe, text=softwareTOMO[0], variable=radVar_TOMO, value=0, command=radCall_TOMO)
        curRad.grid(column=5, row=row, sticky=tk.W)
        curRad = tk.Radiobutton(self.splitframe, text=softwareTOMO[1], variable=radVar_TOMO, value=1, command=radCall_TOMO)
        curRad.grid(column=6, row=row, sticky=tk.W)
        curRad = tk.Radiobutton(self.splitframe, text=softwareTOMO[2], variable=radVar_TOMO, value=2, command=radCall_TOMO)
        curRad.grid(column=7, row=row, sticky=tk.W)

        # ROW 7
        row = 7
        ttk.Label(self.splitframe, text="Movie Extension:", foreground="purple", font=self.font_size_all_text_bold).grid(column=1, row=row, sticky=tk.E)
        #Movie Extension
        moviesExt=["mrc", "tiff"]
        # Radiobutton callback function
        def radCall_MovieExt():
            radSel_MovieEXT=radVar_MovieEXT.get()
            if   radSel_MovieEXT == 0:
                self.movies_ext = "mrc"
                self.c.setMovies_ext(self.movies_ext)
                self.m.printMesgAddStr(" Movies ext = ",
                                       self.c.getYellow(),self.movies_ext)
            elif radSel_MovieEXT == 1:
                self.movies_ext = "tiff"
                self.c.setMovies_ext(self.movies_ext)
                self.m.printMesgAddStr(" Movies ext = ",
                                       self.c.getMagenta(),self.movies_ext)

        radVar_MovieEXT = tk.IntVar()
        # Next we are selecting a non-existing index value for radVar_MovieEXT.
        radVar_MovieEXT.set(97)

        # Now we are creating all three Radiobutton widgets within one loop.
        '''
        for col in range(len(moviesExt)):
            set_col = col+2
            curRad = 'rad' + str(col)
            curRad = tk.Radiobutton(self.splitframe, text=moviesExt[col], variable=radVar_MovieEXT, value=col, command=radCall_MovieExt)
            curRad.grid(column=set_col, row=row, columnspan=9, sticky=tk.W)#columnspan=3,
        '''
        curRad = tk.Radiobutton(self.splitframe, text=moviesExt[0], variable=radVar_MovieEXT, value=0, command=radCall_MovieExt)
        curRad.grid(column=2, row=row, sticky=tk.W)
        curRad = tk.Radiobutton(self.splitframe, text=moviesExt[1], variable=radVar_MovieEXT, value=1, command=radCall_MovieExt)
        curRad.grid(column=3, row=row, sticky=tk.W)

        ttk.Label(self.splitframe, text="Files Handler:", foreground="purple", font=self.font_size_all_text_bold).grid(column=4, row=row, sticky=tk.E)
        #Movie Extension
        fileMoveHandler=["rsync_copy", "rsync_move", "copy","move"]
        # Radiobutton callback function
        def radCall_file_frames_handler():
            radSel_FMVHdl=radVar_FileFMVHdl.get()
            if   radSel_FMVHdl == 0:
                self.file_frame_copy_move_handler = "rsync_copy"
                self.m.printMesgAddStr(" File handler = ",
                                       self.c.getGreen(),
                                       "Frames will be: "+self.c.get_B_Yellow()+\
                                       self.file_frame_copy_move_handler)
            elif radSel_FMVHdl == 1:
                self.file_frame_copy_move_handler = "rsync_move"
                self.m.printMesgAddStr(" File handler = ",
                                       self.c.getGreen(),
                                       "Frames will be: "+self.c.get_B_Cyan()+\
                                       self.file_frame_copy_move_handler)
            elif radSel_FMVHdl == 2:
                self.file_frame_copy_move_handler = "copy"
                self.m.printMesgAddStr(" File handler = ",
                                       self.c.getGreen(),
                                       "Frames will be: "+self.c.get_B_Blue()+\
                                       self.file_frame_copy_move_handler)
            elif radSel_FMVHdl == 3:
                self.file_frame_copy_move_handler = "move"
                self.m.printMesgAddStr(" File handler = ",
                                       self.c.getGreen(),
                                       "Frames will be: "+self.c.get_B_Magenta()+\
                                       self.file_frame_copy_move_handler)

        radVar_FileFMVHdl = tk.IntVar()
        # Next we are selecting a non-existing index value for radVar_FileFMVHdl.
        radVar_FileFMVHdl.set(95)

        # Now we are creating all three Radiobutton widgets within one loop.
        '''
        for col in range(len(fileMoveHandler)):
            set_col = col+2
            curRad = 'rad' + str(col)
            curRad = tk.Radiobutton(self.splitframe, text=fileMoveHandler[col], variable=radVar_FileFMVHdl, value=col, command=radCall_file_frames_handler)
            curRad.grid(column=set_col, row=row, columnspan=9, sticky=tk.W)#columnspan=6,
        '''
        curRad = tk.Radiobutton(self.splitframe, text=fileMoveHandler[0], variable=radVar_FileFMVHdl, value=0, command=radCall_file_frames_handler)
        curRad.grid(column=5, row=row, sticky=tk.W)
        curRad = tk.Radiobutton(self.splitframe, text=fileMoveHandler[1], variable=radVar_FileFMVHdl, value=1, command=radCall_file_frames_handler)
        curRad.grid(column=6, row=row, sticky=tk.W)
        curRad = tk.Radiobutton(self.splitframe, text=fileMoveHandler[2], variable=radVar_FileFMVHdl, value=2, command=radCall_file_frames_handler)
        curRad.grid(column=7, row=row, sticky=tk.W)
        curRad = tk.Radiobutton(self.splitframe, text=fileMoveHandler[3], variable=radVar_FileFMVHdl, value=3, command=radCall_file_frames_handler)
        curRad.grid(column=8, row=row, sticky=tk.W)

        #########################
        # Preprocessing Inputs
        #########################
        # ROW 8
        row = 8
        ttk.Label(self.splitframe, text="Preprocessing Inputs", font=self.font_size_headers).grid(column=1, row=row, sticky=tk.W)

        # ROW 9
        row = 9
        ttk.Label(self.splitframe, text="MotionCorr Directory:", foreground="blue", font=self.font_size_all_text).grid(column=1, row=row, sticky=tk.E)
        ttk.Entry(self.splitframe, width=text_width, textvariable=self.MotionCorrDir, foreground="green", font=self.font_size_all_text).grid(column=2, columnspan=9, row=row, sticky=(tk.W, tk.E))
        ttk.Button(self.splitframe, text="Set Motioncorr path   ", style='DataManageAppStd.TButton', command=(lambda: self.load_directory(self.MotionCorrDir, "MotionCorrDir"))).grid(column=11, row=row, sticky=(tk.W,tk.E))#tk.W)

        # ROW 10
        row = 10
        #fm_dose--->-FmDose 1.30
        ttk.Label(self.splitframe, text="Dose per Frame (e/A2):", font=self.font_size_all_text)                               .grid(column=1 , row=row, sticky=tk.E)
        ttk.Entry(self.splitframe, width=4, textvariable=self.fm_dose, foreground="gray", font=self.font_size_all_text)       .grid(column=2 , row=row, sticky=tk.W)
        #BFactor---> -Bft[{150,500}|150]: Binning factor---> -FtBin[{1,2}|1]
        ttk.Label(self.splitframe, text="Bfactor:", font=self.font_size_all_text)                                             .grid(column=3 , row=row, sticky=tk.E)
        ttk.Entry(self.splitframe, width=5, textvariable=self.bfactor, foreground="gray", font=self.font_size_all_text)       .grid(column=4 , row=row, sticky=tk.W)
        #Binning factor
        ttk.Label(self.splitframe, text="Binning factor:", font=self.font_size_all_text)                                      .grid(column=5 , row=row, sticky=tk.E)
        ttk.Entry(self.splitframe, width=2, textvariable=self.binning_factor, foreground="gray", font=self.font_size_all_text).grid(column=6 , row=row, sticky=tk.W)
        #Gain rotation ---> -RotGain[{0,90,180,270}|0]:Gain flip ---> -FlipGain [{0,1,2}|0]
        ttk.Label(self.splitframe, text="Gain rotation:", font=self.font_size_all_text)                                       .grid(column=7 , row=row, sticky=tk.E)
        ttk.Entry(self.splitframe, width=7, textvariable=self.gain_rotation, foreground="gray", font=self.font_size_all_text) .grid(column=8 , row=row, sticky=tk.W)
        # Gain Flip
        ttk.Label(self.splitframe, text="Gain flip:", font=self.font_size_all_text)                                           .grid(column=9 , row=row, sticky=tk.E)
        ttk.Entry(self.splitframe, width=3, textvariable=self.gain_flip, foreground="gray", font=self.font_size_all_text)     .grid(column=10, row=row, sticky=tk.W)

        # ROW 11
        row = 11
        ttk.Label(self.splitframe, text="CTFFind Directory:", foreground="blue", font=self.font_size_all_text)                                    .grid(column=1, row=row, sticky=tk.E)
        ttk.Entry(self.splitframe, width=30, textvariable=self.CTFfindDir, foreground="green", font=self.font_size_all_text)                      .grid(column=2, columnspan=9, row=row, sticky=(tk.W, tk.E))
        ttk.Button(self.splitframe, text="Set CTFFind path", style='DataManageAppStd.TButton', command=(lambda: self.load_directory(self.CTFfindDir, "CTFfindDir"))).grid(column=11, row=row, sticky=(tk.W,tk.E))#tk.W)

        # ROW 12
        row = 12
        #--sph_abe=SPH_ABE              [2.7] Sphererical Aberration
        ttk.Label(self.splitframe, text="Spherical Aberration:", font=self.font_size_all_text)                                  .grid(column=1 , row=row, sticky=tk.E)
        ttk.Entry(self.splitframe, width=4, textvariable=self.sph_abe, foreground="gray", font=self.font_size_all_text)         .grid(column=2 , row=row, sticky=tk.W)
        #--amp_con=AMP_CON              [0.10] Ampl constrast (from 0.07 06Nov21)
        ttk.Label(self.splitframe, text="Amplitude Contrast:", font=self.font_size_all_text)                                    .grid(column=3 , row=row, sticky=tk.E)
        ttk.Entry(self.splitframe, width=5, textvariable=self.amp_con, foreground="gray", font=self.font_size_all_text)         .grid(column=4 , row=row, sticky=tk.W)
        #--astigpenaltyover=ASTIGPENALTYOVER   [1500.0] Expected \(tolerated\) astig
        ttk.Label(self.splitframe, text="Expected Astigmatism:", font=self.font_size_all_text)                                  .grid(column=5 , row=row, sticky=tk.E)
        ttk.Entry(self.splitframe, width=7, textvariable=self.astigpenaltyover, foreground="gray", font=self.font_size_all_text).grid(column=6 , row=row, sticky=tk.W)

        # ROW 13
        row = 13
        #--minres=MINRES                [30.0] Minimum resolution
        text_width = 217
        ttk.Label(self.splitframe, text="Min Res [A]:", foreground="blue",font=self.font_size_all_text).grid(column=1, row=row, sticky=tk.E)
        ttk.Scale(self.splitframe, variable=self.minres, from_=10, to=200, orient=tk.HORIZONTAL, length=text_width).grid(column=2, columnspan=2, row=row, sticky=tk.W)
        text_width = 5
        ttk.Entry(self.splitframe, textvariable=self.minres, foreground="green", width=text_width, font=self.font_size_all_text).grid(column=4, columnspan=1, row=row, sticky=tk.W)
        #--mindef=MINDEF                [5000] Minimum defocus "3100.0"06Nov21
        text_width = 190
        ttk.Label(self.splitframe, text="Min Def [A]:", foreground="blue", font=self.font_size_all_text).grid(column=5, row=row, sticky=tk.E)
        ttk.Scale(self.splitframe, variable=self.mindef, from_=0, to=25000, orient=tk.HORIZONTAL, length=text_width).grid(column=6, columnspan=2, row=row, sticky=tk.W)
        text_width = 7
        ttk.Entry(self.splitframe, textvariable=self.mindef, foreground="green", width=text_width, font=self.font_size_all_text).grid(column=8, columnspan=1, row=row, sticky=tk.W)

        # ROW 14
        row = 14
        #--maxres=MAXRES                [5.0] Maximum resolution
        text_width = 217
        ttk.Label(self.splitframe, text="Max Res [A]:", foreground="blue",font=self.font_size_all_text).grid(column=1, row=row, sticky=tk.E)
        ttk.Scale(self.splitframe, variable=self.maxres, from_=1, to=20, orient=tk.HORIZONTAL, length=text_width).grid(column=2, columnspan=2, row=row, sticky=tk.W)
        text_width = 5
        ttk.Entry(self.splitframe, textvariable=self.maxres, foreground="green", width=text_width, font=self.font_size_all_text).grid(column=4, columnspan=1, row=row, sticky=tk.W)
        #--maxdef=MAXDEF                [50000] Maximum defocus "6900.0"06Nov21
        text_width = 190
        ttk.Label(self.splitframe, text="Max Def [A]:", foreground="blue", font=self.font_size_all_text).grid(column=5, row=row, sticky=tk.E)
        ttk.Scale(self.splitframe, variable=self.maxdef, from_=20000, to=100000, orient=tk.HORIZONTAL, length=text_width).grid(column=6, columnspan=2, row=row, sticky=tk.W)
        text_width = 7
        ttk.Entry(self.splitframe, textvariable=self.maxdef, foreground="green", width=text_width, font=self.font_size_all_text).grid(column=8, columnspan=1, row=row, sticky=tk.W)

        # ROW 15
        row = 15
        #--sze_pwr_spc=SZE_PWR_SPC      [512] Size of prower spectrum
        text_width = 217
        ttk.Label(self.splitframe, text="Size of Power Spectrum [pix]:", foreground="blue",font=self.font_size_all_text).grid(column=1, row=row, sticky=tk.E)
        ttk.Scale(self.splitframe, variable=self.sze_pwr_spc, from_=64, to=1024, orient=tk.HORIZONTAL, length=text_width).grid(column=2, columnspan=2, row=row, sticky=tk.W)
        text_width = 5
        ttk.Entry(self.splitframe, textvariable=self.sze_pwr_spc, foreground="green", width=text_width, font=self.font_size_all_text).grid(column=4, columnspan=1, row=row, sticky=tk.W)
        #--defstep=DEFSTEP              [500.0] Defocus search step
        text_width = 190
        ttk.Label(self.splitframe, text="Defocus step [A]:", foreground="blue", font=self.font_size_all_text).grid(column=5, row=row, sticky=tk.E)
        ttk.Scale(self.splitframe, variable=self.defstep, from_=200, to=2000, orient=tk.HORIZONTAL, length=text_width).grid(column=6, columnspan=2, row=row, sticky=tk.W)
        text_width = 7
        ttk.Entry(self.splitframe, textvariable=self.defstep, foreground="green", width=text_width, font=self.font_size_all_text).grid(column=8, columnspan=1, row=row, sticky=tk.W)

        # ROW 17
        row = 17
        ttk.Label(self.splitframe, text="Detector Mode and computing:", foreground="purple", font=self.font_size_all_text_bold).grid(column=1, row=row, sticky=tk.E)
        
        ttk.Checkbutton(self.splitframe, text="Super Resolution", variable=self.super_res, command=self.SuperResCheckButtonActivator).grid(column=2, row=row, sticky=tk.W)
        ttk.Checkbutton(self.splitframe, text="Preprocess", command=self.preProcessCheckButtonActivator, variable=self.file_pre_handler_hdl).grid(column=3, row=row, sticky=tk.W)
        ttk.Checkbutton(self.splitframe, text="Live Process", command=self.liveProcessCheckButtonActivator, variable=self.live_process_hdl)     .grid(column=4, row=row, sticky=tk.W)
        ttk.Checkbutton(self.splitframe, text="Motion Correction", command=self.motionCorrCheckButtonActivator,variable=self.use_MotionCorr_hdl).grid(column=5, row=row, sticky=tk.W)
        ttk.Checkbutton(self.splitframe, text="CTFfind", command=self.CTFFindCheckButtonActivator, variable=self.use_CTFfind_hdl)               .grid(column=6, row=row, sticky=tk.W)

        # ROW 20
        row = 20
        ttk.Label(self.splitframe, text="Organise Files into Directories:", foreground="purple", font=self.font_size_all_text_bold).grid(column=1, row=row, sticky=tk.E)
        #Movie Extension
        fileOrganiseDir=["Organise Pool", "Recreate Pool","Do Nothing"]
        # Radiobutton callback function
        def radCall_file_organise_dir():
            radSel_FODir=radVar_FileOrgDir.get()
            if   radSel_FODir == 0:
                self.file_organise_dir = "or"
                self.m.printMesgAddStr(" Data Structure = ",
                                       self.c.getGreen(),"Organise files in the Pool: "+ \
                                       self.file_organise_dir)
            elif radSel_FODir == 1:
                self.file_organise_dir = "re"
                self.m.printMesgAddStr(" Data Structure = ",
                                       self.c.getYellow(),
                                       "Recreate/ReOrganise files back to orignal Pool " \
                                       "configuration: "+ \
                                       self.file_organise_dir)
            elif radSel_FODir == 2:
                self.file_organise_dir = "no"
                self.m.printMesgAddStr(" Data Structure = ",
                                       self.c.getCyan(),
                                       "Do nothing: leave as it is: "+\
                                       self.file_organise_dir)

        radVar_FileOrgDir = tk.IntVar()
        # Next selecting a non-existing index value for radVar_FileOrgDir.
        radVar_FileOrgDir.set(96)

        # Now we are creating all three Radiobutton widgets within one loop.
        '''
        for col in range(len(fileOrganiseDir)):
            set_col = col+2
            curRad = 'rad' + str(col)
            curRad = tk.Radiobutton(self.splitframe, text=fileOrganiseDir[col], variable=radVar_FileOrgDir, value=col, command=radCall_file_organise_dir)
            curRad.grid(column=set_col, row=row, columnspan=5, sticky=tk.W)
        '''
        curRad = tk.Radiobutton(self.splitframe, text=fileOrganiseDir[0], variable=radVar_FileOrgDir, value=0, command=radCall_file_organise_dir)
        curRad.grid(column=2, row=row, columnspan=5, sticky=tk.W)
        curRad = tk.Radiobutton(self.splitframe, text=fileOrganiseDir[1], variable=radVar_FileOrgDir, value=1, command=radCall_file_organise_dir)
        curRad.grid(column=3, row=row, columnspan=5, sticky=tk.W)
        curRad = tk.Radiobutton(self.splitframe, text=fileOrganiseDir[2], variable=radVar_FileOrgDir, value=2, command=radCall_file_organise_dir)
        curRad.grid(column=4, row=row, columnspan=5, sticky=tk.W)

        # ROW 21
        row = 21
        ttk.Label(self.splitframe, text="Extract Data:", foreground="purple", font=self.font_size_all_text_bold).grid(column=1, row=row, sticky=tk.E)
        #Movie Extension
        filePstHandler=["From Files", "no"]
        # Radiobutton callback function
        def radCall_file_pst_handler():
            radSel_FPSTHdl=radVar_FilePstHdl.get()
            if   radSel_FPSTHdl == 0:
                self.file_pst_handler = "extract_data"
                self.m.printMesgAddStr(" File post-handler = ",
                                       self.c.getGreen(),
                                       "Extract data from files: from the files and the preprocessing computation"+\
                                       self.file_pst_handler)
            elif radSel_FPSTHdl == 1:
                self.file_pst_handler = "no"
                self.m.printMesgAddStr(" File post-handler = ",
                                       self.c.getGreen(),
                                       "No action taken: no data extraction to be performed leave as is"+\
                                       self.file_pst_handler)

        radVar_FilePstHdl = tk.IntVar()
        # Next we are selecting a non-existing index value for radVar_FilePstHdl.
        radVar_FilePstHdl.set(95)

        # Now we are creating all three Radiobutton widgets within one loop.
        '''
        for col in range(len(filePstHandler)):
            set_col = col+2
            curRad = 'rad' + str(col)
            curRad = tk.Radiobutton(self.splitframe, text=filePstHandler[col], variable=radVar_FilePstHdl, value=col, command=radCall_file_pst_handler)
            curRad.grid(column=set_col, row=row, columnspan=6, sticky=tk.W)
        '''
        curRad = tk.Radiobutton(self.splitframe, text=filePstHandler[0], variable=radVar_FilePstHdl, value=0, command=radCall_file_pst_handler)
        curRad.grid(column=2, row=row, columnspan=6, sticky=tk.W)
        curRad = tk.Radiobutton(self.splitframe, text=filePstHandler[1], variable=radVar_FilePstHdl, value=1, command=radCall_file_pst_handler)
        curRad.grid(column=3, row=row, columnspan=6, sticky=tk.W)

        #########################
        # Computational Inputs
        #########################
        # ROW 22
        row = 22
        ttk.Label(self.splitframe, text="Computational Inputs", font=self.font_size_headers).grid(column=1, row=row, sticky=tk.W)

        # ROW 23
        row = 23
        ttk.Label(self.splitframe, text="BenchMarking:", font=self.font_size_all_text).grid(column=1, row=row, sticky=tk.E)
        ttk.Checkbutton(self.splitframe,text=logfile,
                        command=self.BenchMarkingCheckButtonActivator, variable=self.doBenchMarking).grid(column=2, row=row, sticky=tk.W)

        # ROW 24
        row = 24
        ttk.Label(self.splitframe, text="Use GPU:", font=self.font_size_all_text)                                      .grid(column=1, row=row, sticky=tk.E)
        ttk.Checkbutton(self.splitframe, text="CUDA C", command=self.UseGpuCheckButtonActivator, variable=self.use_gpu).grid(column=2, row=row, sticky=tk.W)
        ttk.Label(self.splitframe, text="GPU[{0:1:2:3}}|0]:", font=self.font_size_all_text)                            .grid(column=3, row=row, sticky=tk.W)
        ttk.Entry(self.splitframe, width=7, textvariable=self.gpu_list, font=self.font_size_all_text)                  .grid(column=4, row=row, sticky=tk.W)
        ttk.Label(self.splitframe, text="Create Project Pool:", foreground="darkgreen", font=self.font_size_all_text_bold)                     .grid(column=5, row=row, sticky=tk.W)
        ttk.Checkbutton(self.splitframe, command=self.PoolifyCheckButtonActivator, variable=self.poolify)              .grid(column=6, row=row, sticky=tk.W)

        #########################
        # Lower buttons
        #########################
        # ROW 25
        row = 25
        # Quit button
        ttk.Button(self.splitframe, text="Quit", style='DataManageAppRed.TButton', command=self.quit)                 .grid(column=1, row=row, sticky=tk.W)
        # Stop button
        ttk.Button(self.splitframe, text="Stop RUN", style='DataManageApp.TButton', command=self.stopRun)             .grid(column=2, row=row, sticky=(tk.W,tk.E))#tk.W)
        # Show astigmatism reults
        ttk.Button(self.splitframe, text="Show Astg", style='DataManageApp.TButton', command=self.showCTFAstigResults).grid(column=3, row=row, sticky=(tk.W,tk.E))#tk.W)
        # Show astigmatism reults
        ttk.Button(self.splitframe, text="Show Graphs", style='DataManageApp.TButton', command=self.showPreprocGraphs).grid(column=4, row=row, sticky=(tk.W,tk.E))#tk.W)
        # Check Inputs and RUN button
        ttk.Button(self.splitframe, text="Check->RUN Pre-Proc", style='DataManageApp.TButton', command=self.checkInputsAndRun).grid(column=5, columnspan=2, row=row, sticky=(tk.W,tk.E))#tk.W)
        # Check Inputs and RUN button
        ttk.Button(self.splitframe, text="Launch Processing", style='DataManageApp.TButton', command=self.checkInputsAndLaunchPostProcessingApp).grid(column=10, columnspan=3, row=row, sticky=(tk.W,tk.E))#tk.E)

        # Setup grid with padding
        for child in self.splitframe.winfo_children(): child.grid_configure(padx=5, pady=5)
        #-----------------------------------------------------------------------
        # Binding upon return
        #-----------------------------------------------------------------------
        self.parent.bind("<Return>",self.checkInputsAndRun)
        #self.parent.bind("<Return>",self.checkInputsAndLaunchPostProcessingApp)
    #---------------------------------------------------------------------------
    # class methods
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [Launchers]
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [LaunchPostProcessingApp]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # LaunchPostProcessingApp class method to launch the DataManager application
    # \param self     The self Object
    # \param c        DataManageApp common object
    # \param m        The mnessdeanger class for output messaging and loggin
    # \param q2       The multiprocessing Queue object
    def LaunchPreprocGraphs(self,c,m,q):
        __func__= sys._getframe().f_code.co_name
        q.put("Launching DataManage Show Results for Pre-Processing LDMaP-App...\n")
        q.put("Function                       : "+__func__+"\n")
        q.put("CTF-File to be plotted         : "+self.asc_file_full_path+"\n")
        
        PARAMS = []
        live_ ="--app_root="+self.c.getApp_root()
        PARAMS.append(live_)
        live_ ="--projectName="+self.c.getProjectName()
        PARAMS.append(live_)
        live_ ="--data_path="+self.c.getData_path()
        PARAMS.append(live_)
        live_ ="--targetdir="+self.c.getTargetdir()
        PARAMS.append(live_)
        live_ ="--software="+self.c.getSoftware()
        PARAMS.append(live_)
        live_ = "--movies_ext=" + self.c.getMovies_ext()
        PARAMS.append(live_)
        live_ ="--ctf_asc_file="+self.asc_file
        PARAMS.append(live_)

        CMD = ' '.join(PARAMS)
        #print(CMD)
        os.system('python3 DataManage_graphShowResults.py %s'%CMD)
        return
    #---------------------------------------------------------------------------
    # [showPreprocGraphs]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # showPreprocGraphs class method to check the inputs and then run the
    # application
    # \param self     The self Object
    # \param *args    The list of arguments from the command prompt
    def showPreprocGraphs(self,*args):
        __func__= sys._getframe().f_code.co_name
        #-----------------------------------------------------------------------
        # Starting the statemachine
        #-----------------------------------------------------------------------
        self.m.printMesgStr("State Machine: ",self.c.get_B_Green(),
                            " Start_computation")
        self.device.on_event('Start_computation')
        self.device.state
        #-----------------------------------------------------------------------
        # Seting the fields values into the c object before passing
        #-----------------------------------------------------------------------
        self.doBenchMarking.get()
        self.c.setApp_root(self.app_root.get())
        self.c.setProjectName(self.projectName.get())
        self.c.setData_path(self.data_path.get())
        self.c.setTargetdir(self.targetDirectory.get())
        # Setting the asc file to be plotted
        self.asc_file = self.c.getProjectName() + "_Pool.asc"
        self.asc_file_full_path = self.c.getTargetdir() + os.path.sep + self.c.getProjectName() + os.path.sep + self.asc_file
        #-----------------------------------------------------------------------
        # Check all of the inputs before launching driver code Poolifier
        #-----------------------------------------------------------------------
        #Changing dir to the application root directory to make sure that we are
        #at the right place this should already be from the down
        os.chdir(self.app_root.get())
        #-----------------------------------------------------------------------
        # Creating the GUI object for the subprocess launch of DataManagerApp
        #-----------------------------------------------------------------------
        self.q3 = multiprocessing.Queue()
        self.q3.cancel_join_thread() # or else thread that puts data will not term
        gui = ShowResultsGuiApp(self.q3, self.c,self.m)
        t3 = multiprocessing.Process(target=self.LaunchPreprocGraphs,
                                     args=(self.c,self.m, self.q3 ) )
        t3.start()
        gui.root.mainloop()
        t3.join()
        #-----------------------------------------------------------------------
        # Starting the StateMachine
        #-----------------------------------------------------------------------
        global running
        running = True

        #declared but not needed just yet may need to turn this into a self
        showGUI = True

        return
    #---------------------------------------------------------------------------
    # [LaunchPostProcessingApp]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # LaunchPostProcessingApp class method to launch the DataManager application
    # \param self     The self Object
    # \param c        DataManageApp common object
    # \param m        The mnessdeanger class for output messaging and loggin
    # \param q2       The multiprocessing Queue object
    def LaunchCTFAstigResults(self,c,m,q):
        __func__= sys._getframe().f_code.co_name
        q.put("Launching DataManage Show Results for Pre-Processing LDMaP-App...\n")
        q.put("Function                       : "+__func__+"\n")
        q.put("CTF-File to be plotted         : "+self.asc_file_full_path+"\n")
        
        PARAMS = []
        live_ ="--app_root="+self.c.getApp_root()
        PARAMS.append(live_)
        live_ ="--projectName="+self.c.getProjectName()
        PARAMS.append(live_)
        live_ ="--data_path="+self.c.getData_path()
        PARAMS.append(live_)
        live_ ="--targetdir="+self.c.getTargetdir()
        PARAMS.append(live_)
        live_ ="--software="+self.c.getSoftware()
        PARAMS.append(live_)
        live_ ="--ctf_asc_file="+self.asc_file
        PARAMS.append(live_)

        CMD = ' '.join(PARAMS)
        #print(CMD)
        os.system('python3 DataManage_ctfShowResults.py %s'%CMD)
        return
    #---------------------------------------------------------------------------
    # [showCTFAstigResults]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # showCTFAstigResults class method to check the inputs and then run the application
    # \param self     The self Object
    # \param *args    The list of arguments from the command prompt
    def showCTFAstigResults(self,*args):
        __func__= sys._getframe().f_code.co_name
        #-----------------------------------------------------------------------
        # Starting the statemachine
        #-----------------------------------------------------------------------
        self.m.printMesgStr("State Machine: ",self.c.get_B_Green(),
                            " Start_computation")
        self.device.on_event('Start_computation')
        self.device.state
        #-----------------------------------------------------------------------
        # Seting the fields values into the c object before passing
        #-----------------------------------------------------------------------
        self.doBenchMarking.get()
        self.c.setApp_root(self.app_root.get())
        self.c.setProjectName(self.projectName.get())
        self.c.setData_path(self.data_path.get())
        self.c.setTargetdir(self.targetDirectory.get())
        # Setting the asc file to be plotted
        self.asc_file = self.c.getProjectName() + "_Pool.asc"
        self.asc_file_full_path = self.c.getTargetdir() + os.path.sep + self.c.getProjectName() + os.path.sep + self.asc_file
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
        gui = ShowResultsGuiApp(self.q2, self.c,self.m)
        t2 = multiprocessing.Process(target=self.LaunchCTFAstigResults,
                                     args=(self.c,self.m, self.q2 ) )
        t2.start()
        gui.root.mainloop()
        t2.join()
        #-----------------------------------------------------------------------
        # Starting the StateMachine
        #-----------------------------------------------------------------------
        global running
        running = True

        #declared but not needed just yet may need to turn this into a self
        showGUI = True

        return
    #---------------------------------------------------------------------------
    # [LaunchPostProcessingApp]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # LaunchPostProcessingApp class method to launch the DataManager application
    # \param self     The self Object
    # \param c        DataManageApp common object
    # \param m        The mnessdeanger class for output messaging and loggin
    # \param q3       The multiprocessing Queue object
    def LaunchPostProcessingApp(self, c, m, q3):
        __func__= sys._getframe().f_code.co_name
        q3.put("Function                       : "+__func__+"\n")
        q3.put("Launching DataManage Post-Processing App...\n")

        PARAMS = []
        live_ ="--app_root="+c.getApp_root()
        PARAMS.append(live_)
        live_ ="--projectName="+c.getProjectName()
        PARAMS.append(live_)
        live_ ="--data_path="+c.getData_path()
        PARAMS.append(live_)
        live_ ="--targetdir="+c.getTargetdir()
        PARAMS.append(live_)
        live_ ="--software="+str(c.getSoftware())
        PARAMS.append(live_)
        if self.doBenchMarking.get() == True:
            live_ ="--doBenchMarking"
            PARAMS.append(live_)

        CMD = ' '.join(PARAMS)
        os.system('python3 DataManage_pstProcessing.py %s'%CMD)
        return
    #---------------------------------------------------------------------------
    # [checkInputsAndLaunchPostProcessingApp]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # checkInputsAndLaunchPostProcessingApp class method to check the inputs and then run the application
    # \param self     The self Object
    # \param *args    The list of arguments from the command prompt
    def checkInputsAndLaunchPostProcessingApp(self,*args):
        __func__= sys._getframe().f_code.co_name
        #-----------------------------------------------------------------------
        # Starting the statemachine
        #-----------------------------------------------------------------------
        self.m.printMesgStr("State Machine: ",self.c.get_B_Green(),
                            " Start_computation")
        self.device.on_event('Start_computation')
        self.device.state
        #-----------------------------------------------------------------------
        # Seting the fields values into the c object before passing
        #-----------------------------------------------------------------------
        self.doBenchMarking.get()
        self.c.setApp_root(self.app_root.get())
        self.c.setProjectName(self.projectName.get())
        self.c.setData_path(self.data_path.get())
        self.c.setTargetdir(self.targetDirectory.get())
        # Setting the asc file to be plotted
        self.asc_file = self.c.getProjectName() + "_Pool.asc"
        self.asc_file_full_path = self.c.getTargetdir() + os.path.sep + self.c.getProjectName() + os.path.sep + self.asc_file
        #-----------------------------------------------------------------------
        # Check all of the inputs before launching driver code Poolifier
        #-----------------------------------------------------------------------
        #Changing dir to the application root directory to make sure that we are
        #at the right place this should already be from the down
        os.chdir(self.app_root.get())
        #-----------------------------------------------------------------------
        # Creating the GUI object for the subprocess launch of DataManagerApp
        #-----------------------------------------------------------------------
        self.q3 = multiprocessing.Queue()
        self.q3.cancel_join_thread() # or else thread that puts data will not term
        gui = PostProcessingGuiApp(self.q3)
        t3 = multiprocessing.Process(target=self.LaunchPostProcessingApp,
                                     args=(self.c,self.m, self.q3 ) )
        t3.start()
        gui.root.mainloop()
        t3.join()
        #-----------------------------------------------------------------------
        # Starting the StateMachine
        #-----------------------------------------------------------------------
        global running
        running = True

        #declared but not needed just yet may need to turn this into a self
        showGUI = True

        return
    #---------------------------------------------------------------------------
    # [Setters]
    #---------------------------------------------------------------------------
    #--file_pre_handler=preprocess---> combox box
    def set_file_pre_handler(self,file_pre_handler_hdl):
        #print("file_pre_handler_hdl: ",file_pre_handler_hdl.get()
        if file_pre_handler_hdl.get()==True: self.file_pre_handler="preprocess"
        if file_pre_handler_hdl.get()==False: self.file_pre_handler="lookupdata"
    #--live_process=yes--->checkBox
    def set_live_process(self,live_process_hdl):
        #print("live_process_hdl: ",live_process_hdl.get())
        if (live_process_hdl.get() == True): self.live_process = "yes"
        if (live_process_hdl.get() == False): self.live_process = "no"
    #--use_MotionCorr=0---> checkBox
    def set_use_MotionCorr(self,use_MotionCorr_hdl):
        #print("use_MotionCorr_hdl.get(): ", use_MotionCorr_hdl.get())
        if (use_MotionCorr_hdl.get() == True): self.use_MotionCorr = 1
        if (use_MotionCorr_hdl.get() == False): self.use_MotionCorr = 0
    #--use_CTFfind=0---> checkBox
    def set_use_CTFfind(self,use_CTFfind_hdl):
        #print("use_CTFfind_hdl.get(): ", use_CTFfind_hdl.get())
        if (use_CTFfind_hdl.get() == True): self.use_CTFfind = 1
        if (use_CTFfind_hdl.get() == False): self.use_CTFfind = 0
    #--use_unblur=0---> checkBox
    def set_use_unblur(self,use_unblur_hdl):
        #print("use_unblur_hdl: ", use_unblur_hdl.get())
        if (use_unblur_hdl.get() == True): self.use_unblur = 1
        if (use_unblur_hdl.get() == False): self.use_unblur = 0
    #---------------------------------------------------------------------------
    # [browsePath]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # DataManageApp class method to browse the file and load the chimera binary
    # \param self                  The self Object
    # \param chimeraBin            Chimera binary
    def browsePath(self,chimeraBin):
        __func__= sys._getframe().f_code.co_name
        options =  {}
        options['title'] = "LDMaP-App - Chimera path to binary"
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
    # DataManageApp class method to load file
    # \param self                  The self Object
    # \param fileNameStringVar     Strin variable file name
    def load_file(self, fileNameStringVar, which_file):
        __func__= sys._getframe().f_code.co_name
        options =  {}
        # options['filetypes'] = [ ("All files", ".*"), ("MRC map", ".map,.mrc") ]      # THIS SEEMS BUGGY...
        options['title'] = "LDMaP-App - Select data file"
        if which_file == "gainRef":
            options['title'] = "LDMaP-App - Select a Gain Reference"
        fname = askopenfilename(**options)
        if fname:
            try:
                fileNameStringVar.set(fname)
                if which_file == "gainRef":
                    self.m.printMesgAddStr(" Gain Reference selected = ", self.c.getGreen(), fname)
            except:                     # <- naked except is a bad idea
                showerror("Open Source File", "Failed to read file\n'%s'" % fname)
            return
    #---------------------------------------------------------------------------
    # [load_files]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # DataManageApp class method to load the two inpout maps files
    # \param self                  The self Object
    # \param fileNameStringVar1    Input string for Volume 1
    # \param fileNameStringVar2    Input string for Volume 2
    def load_files(self, fileNameStringVar1, fileNameStringVar2):
        __func__= sys._getframe().f_code.co_name
        options =  {}
        options['title'] = "LDMaP-App - Select data files"
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
    # DataManageApp class method to load the two inpout maps files
    # \param self                  The self Object
    # \param fileNameStringVar1    Input string for Volume 1
    def load_directory(self, fileNameStringVar1, which_dir):
        __func__= sys._getframe().f_code.co_name
        options =  {}
        options['title'] = "LDMaP-App - Select Directory"
        if which_dir == "app_root":
            options['title'] = "LDMaP-App - Select Application Root Directory"
        if which_dir == "data_path":
            options['title'] = "LDMaP-App - Select Data Path Directory"
        if which_dir == "targetDirectory":
            options['title'] = "LDMaP-App - Select Target Directory"
        if which_dir == "MotionCorrDir":
            options['title'] = "LDMaP-App - Select Motion Correction Directory"
        if which_dir == "CTFfindDir":
            options['title'] = "LDMaP-App - Select CTF Find Directory"

        fname = askdirectory(**options)
        if fname:
            try:
                fileNameStringVar1.set(fname)
                if which_dir == "app_root":
                    self.m.printMesgAddStr(" Application Root Directory selected = ", self.c.getGreen(), fname)
                if which_dir == "data_path":
                    self.m.printMesgAddStr(" Data Path Directory selected = ", self.c.getGreen(), fname)
                if which_dir == "targetDirectory":
                    self.m.printMesgAddStr(" Target Directory selected = ", self.c.getGreen(), fname)
                if which_dir == "MotionCorrDir":
                    self.m.printMesgAddStr(" MotionCorr Directory selected = ", self.c.getGreen(), fname)
                if which_dir == "CTFfindDir":
                    self.m.printMesgAddStr(" CTF Find Directory selected = ", self.c.getGreen(), fname)
            except:                     # <- naked except is a bad idea
                showerror("Open Source Files", "Failed to load Directory\n'%s'" % fname)
            return
    #---------------------------------------------------------------------------
    # [Scanning]
    #---------------------------------------------------------------------------
    def scanning(self):
        __func__= sys._getframe().f_code.co_name
        if running:  # Only do this if the Stop button has not been clicked
            #print("hello")
            #-------------------------------------------------------------------
            # Starting the timers for the preprocessing computation
            #-------------------------------------------------------------------
            #creating the timers and starting the stop watch...
            if self.c.getBench_cpu():
                stopwatch = utils.StopWatch.createTimer()    #createTimer()
                utils.StopWatch.StartTimer(stopwatch)        #StartTimer(stopwatch)
            #-------------------------------------------------------------------
            # Starting preprocessing with DataManage_live-PreProcessing
            #-------------------------------------------------------------------
            #Instert code here for the wrarper code of the data live processing
            my_time = time.asctime()    #clock_gettime(threading.get_ident())
            m.printMesgAddStr(" The preprocessing computation is starting: ",
                              c.getRed(),str(my_time))

            in_list = []
            live_ ="--app_root="+self.c.getApp_root()
            in_list.append(live_)
            live_ ="--projectName="+self.c.getProjectName()
            in_list.append(live_)
            live_ ="--data_path="+self.c.getData_path()
            in_list.append(live_)
            live_ ="--gainmrc="+self.c.getGainmrc()
            in_list.append(live_)
            live_ ="--targetdir="+self.c.getTargetdir()
            in_list.append(live_)
            live_ ="--poolify="+self.c.getPoolify()
            in_list.append(live_)
            live_ ="--software="+str(self.c.getSoftware())
            in_list.append(live_)
            live_ ="--use_gpu="+self.c.getUse_gpu()
            in_list.append(live_)
            live_ ="--gpu_list="+self.c.getGpu_list()
            in_list.append(live_)
            live_ ="--super_res="+self.c.getSuper_Res()
            in_list.append(live_)

            live_ ="--nframes="+str(c.getNframes())
            in_list.append(live_)
            live_ ="--sph_abe="+str(c.getSph_Abe())
            in_list.append(live_)
            live_ ="--amp_con="+str(c.getAmp_Con())
            in_list.append(live_)
            live_ ="--sze_pwr_spc="+str(c.getSze_Pwr_Spc())
            in_list.append(live_)
            live_ ="--minres="+str(c.getMinRes())
            in_list.append(live_)
            live_ ="--maxres="+str(c.getMaxRes())
            in_list.append(live_)
            live_ ="--mindef="+str(c.getMinDef())
            in_list.append(live_)
            live_ ="--maxdef="+str(c.getMaxDef())
            in_list.append(live_)
            live_ ="--defstep="+str(c.getDefStep())
            in_list.append(live_)
            live_ ="--astigpenaltyover="+str(c.getAstigPenaltyOver())
            in_list.append(live_)

            live_ ="--fm_dose="+str(self.c.getFMDose())
            in_list.append(live_)
            live_ ="--bfactor="+str(self.c.getBfactor())
            in_list.append(live_)
            live_ ="--binning_factor="+str(self.c.getBinning_factor())
            in_list.append(live_)
            live_ ="--gain_rotation="+str(self.c.getGain_rotation())
            in_list.append(live_)
            live_ ="--gain_flip="+str(self.c.getGain_flip())
            in_list.append(live_)
            live_ ="--movies_ext="+self.c.getMovies_ext()
            in_list.append(live_)
            live_ ="--live_process="+self.c.getLive_process()
            in_list.append(live_)
            live_ ="--use_unblur="+str(self.c.getUse_unblur())
            in_list.append(live_)
            live_ ="--use_MotionCorr="+str(self.c.getUse_MotionCorr())
            in_list.append(live_)
            live_ ="--use_CTFfind="+str(self.c.getUse_CTFfind())
            in_list.append(live_)
            live_ = "--file_frame_copy_move_handler="+self.c.getFile_frame_copy_move_handler()
            in_list.append(live_)
            live_ ="--file_pre_handler="+self.c.getFile_pre_handler()
            in_list.append(live_)
            live_ ="--file_organise_dir="+self.c.getFile_organise_dir()
            in_list.append(live_)
            live_ ="--file_pst_handler="+self.c.getFile_pst_handler()
            in_list.append(live_)
            live_ = "--motioncorr_dir="+self.c.getMotionCorr_dir()
            in_list.append(live_)
            live_ = "--ctffind_dir="+self.c.getCTFFind_dir()
            in_list.append(live_)
            live_ =""+self.c.getdoBenchMarking_marker()
            in_list.append(live_)
            
            CMD = ' '.join(in_list[:])
            m.printMesgAddStr(" python3 DataManage_live-PreProcessing.py ",
                              c.getCyan(), CMD)

            '''
            python3 DataManage_live-PreProcessing.py
            (done) --app_root=/home/frederic/Leiden/SourceCode/DataManage_project
            (done) --projectName=supervisor_20200213_161112_k3_apof_benchmark
            (done) --data_path=/Drive_C/frederic/data/ApofTestProject/Contracted
            (done) --gainmrc=/Drive_C/frederic/data/gainref_m0_2020-02-13--17-21-06.mrc
            (done) --targetdir=/Drive_C/frederic/data/LiveAppOut/DataManage_live_out
            (done) --fm_dose=1.0--->textfield
            (done) --bfactor=150 int--->textfield
            (done) --binning_factor=1 int--->textfield
            (done) --gain_rotation=0 int--->textfield
            (done) --gain_flip=0 int--->textfield
            (done) --software=epu                    ---> Already have
            (done) --use_gpu=no                      ---> checkBox
            (done) --gpu_list=0:1:2                  ---> checkBox
            (done) --poolify=no                      ---> checkBox
            (done) --super_res=yes                   ---> checkBox
            (done) --movies_ext=tiff                 ---> radioButton
            (done) --live_process=yes                ---> checkBox
            (done) --use_unblur=0                    ---> checkBox
            (done) --use_MotionCorr=0                ---> checkBox
            (done) --use_CTFfind=0                   ---> checkBox
            (done) --file_pre_handler=preprocess     ---> combox box
            (done) --file_organise_dir=re            ---> Combo box 
            (done) --file_pst_handler=extract_data   ---> combo box
            (done) --motioncorr_dir=/opt/Devel_tools/MotionCor2_v1.3.1/
            (done) --ctffind_dir=/opt/Devel_tools/CTFfind/ctffind-4.1.18/bin_compat/
            (done) --doBenchMarking                  ---> checkBox
            '''

            time.sleep(10)

            # Call DataManage_live-PreProcessing
            os.system('python3 DataManage_live-PreProcessing.py %s'%CMD)

            #-------------------------------------------------------------------
            # Reporting time taken to instantiate and strip innitial star file
            #-------------------------------------------------------------------
            if self.c.getBench_cpu():
                utils.StopWatch.StopTimer_secs(stopwatch)#StopTimer_secs(stopwatch)
                info = c.GetFrameInfo()
                self.m.printBenchMap("data_path",self.c.getRed(),
                                     self.data_path.get(), info, __func__)
                self.m.printBenchTime_cpu("Read data_path file",
                                          self.c.getBlue(), stopwatch,
                                          info, __func__)
        #Creating the recursive loop for computation
        #self.m.printMesgStr("Scanning: ",self.c.get_B_Red()," TODO: do not forget to remove statement in the scanning method for the continueous loop")
        root.after(1000, self.scanning)

    #---------------------------------------------------------------------------
    # [Starters]
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [start_ParticlePick]
    #---------------------------------------------------------------------------
    def start_ParticlePick(self,*args):
        __func__= sys._getframe().f_code.co_name
        #Stop scanning by setting the global flag to False.
        global running
        running = True
        # Find which tab the button was pressed from
        self.m.printMesgStr("State Machine: ",self.c.get_B_Green()," Start_computation")
        self.device.on_event('Start_computation')
        self.device.state

        return
    #---------------------------------------------------------------------------
    # [startRun_Class2D]
    #---------------------------------------------------------------------------
    def startRun_Class2D(self,*args):
        __func__= sys._getframe().f_code.co_name
        #Stop scanning by setting the global flag to False.
        global running
        running = True
        # Find which tab the button was pressed from
        self.m.printMesgStr("State Machine: ",self.c.get_B_Green()," Start_computation")
        self.device.on_event('Start_computation')
        self.device.state

        return
    #---------------------------------------------------------------------------
    # [Stopers]
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [StopRun]
    #---------------------------------------------------------------------------
    def stopRun(self,*args):
        __func__= sys._getframe().f_code.co_name
        #Stop scanning by setting the global flag to False.
        global running
        running = False
        # Find which tab the button was pressed from
        self.m.printMesgStr("State Machine: ",self.c.get_B_Cyan()," device_locked")
        self.device.on_event('device_locked')
        self.device.state

        return
    #---------------------------------------------------------------------------
    # [CheckButton-Activators]
    #---------------------------------------------------------------------------
    def BenchMarkingCheckButtonActivator(self):
        if self.doBenchMarking.get() == False:
            self.m.printMesgAddStr(" BenchMarking = ",
                                   self.c.getMagenta(), str(self.doBenchMarking.get())+" will not be computed" )
        if self.doBenchMarking.get() == True:
            self.m.printMesgAddStr(" BenchMarking = ",
                                   self.c.getYellow(), str(self.doBenchMarking.get())+" Will be computed")
        return

    def motionCorrCheckButtonActivator(self):
        if self.use_MotionCorr_hdl.get() == False:
            self.m.printMesgAddStr(" Motion Correction = ",
                                   self.c.getCyan(), str(self.use_MotionCorr_hdl.get())+" will not be computed" )
        if self.use_MotionCorr_hdl.get() == True:
            self.m.printMesgAddStr(" Motion Correction = ",
                                   self.c.getGreen(), str(self.use_MotionCorr_hdl.get())+" Will be computed")
        return

    def CTFFindCheckButtonActivator(self):
        if self.use_CTFfind_hdl.get() == False:
            self.m.printMesgAddStr(" CTFFind = ",
                                   self.c.getCyan(), str(self.use_CTFfind_hdl.get())+" will not be estimated" )
        if self.use_CTFfind_hdl.get() == True:
            self.m.printMesgAddStr(" CTFFind = ",
                                   self.c.getGreen(), str(self.use_CTFfind_hdl.get())+" Will be estimated")
        return

    def UseGpuCheckButtonActivator(self):
        if self.use_gpu.get() == False:
            self.m.printMesgAddStr(" Use GPU = ",
                                   self.c.getMagenta(), "will not be used" )
        if self.use_gpu.get() == True:
            self.m.printMesgAddStr(" Use GPU = ",
                                   self.c.getGreen(), "Will be used")
        return

    def PoolifyCheckButtonActivator(self):
        if self.poolify.get() == False:
            self.m.printMesgAddStr(" Project Pool = ",
                                   self.c.getCyan(), "will not be created")
        if self.poolify.get() == True:
            self.m.printMesgAddStr(" Project Pool = ",
                                   self.c.getGreen(), "Will be created")
        return

    def SuperResCheckButtonActivator(self):
        if self.super_res.get() == False:
            self.m.printMesgAddStr(" Super Resolution = ",
                                   self.c.getCyan(), "Non super resolution no factor 1/2 applied on the pixel size")
        if self.super_res.get() == True:
            self.m.printMesgAddStr(" Super Resolution = ", self.c.getGreen(), "Super resolution a factor 1/2 applied on the pixel size")
        return

    def liveProcessCheckButtonActivator(self):
        if self.live_process_hdl.get() == False:
            self.m.printMesgAddStr(" Live Process = ",
                                   self.c.getCyan(), "Computation will be done in static" )
        if self.live_process_hdl.get() == True:
            self.m.printMesgAddStr(" Live Process = ",
                                   self.c.getGreen(), "Computation will be executed in a live process mode" )
        return

    def preProcessCheckButtonActivator(self):
        if self.file_pre_handler_hdl.get() == False:
            self.m.printMesgAddStr(" Pre Process = ",
                                   self.c.getCyan(), "The data will not be pre-processed" )
        if self.file_pre_handler_hdl.get() == True:
            self.m.printMesgAddStr(" Pre Process = ",
                                   self.c.getGreen(), "The data will be pre-processed" )
        return
    #---------------------------------------------------------------------------
    # [checkInputsAndRun]
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # DataManageApp class method to check the inputs and then run the application
    # \param self     The self Object
    # \param *args    The list of arguments from the command prompt
    def checkInputsAndRun(self,*args):
        __func__= sys._getframe().f_code.co_name
        #-----------------------------------------------------------------------
        # Starting the statemachine
        #-----------------------------------------------------------------------
        self.m.printMesgStr("State Machine: ",self.c.get_B_Green()," Start_computation")
        self.device.on_event('Start_computation')
        self.device.state
        #-----------------------------------------------------------------------
        # Check all of the inputs before launching driver code Poolifier
        #-----------------------------------------------------------------------
        if self.data_path.get() == "":
            showerror("Check Inputs","'Micrographs' are not set. Select a Frames to analyze.")
            return
        else:
            try:
                input_data_path = self.data_path.get()
                self.m.printMesgStr("Directory: ", self.c.getGreen(), input_data_path)
                #TODO: need to implement the method for checking the validity of the directory
            except:
                showerror("Check Inputs", "The Micrographs could not be read.")
                return
        '''
        if self.gainRef.get() == "":
            showerror("Check Inputs", "'gainRef' is not set. Select a Gain Reference to analyze.")
            #return
        else:
            try:
                input_gainmrc = self.gainRef.get()
                self.m.printMesgStr("gainref: ", self.c.getYellow(), input_gainmrc)
            except:
                showerror("Check Inputs", "The Gain Reference file could not be read.")
                return
        '''
        if self.gainRef.get() != "":
            try:
                input_gainmrc = self.gainRef.get()
                self.m.printMesgStr("gainref: ", self.c.getYellow(), input_gainmrc)
            except:
                showerror("Check Inputs", "The Gain Reference file could not be read.")
                return
        #-----------------------------------------------------------------------
        # Setting the values into self object for the live processing input
        #-----------------------------------------------------------------------
        self.set_live_process(self.live_process_hdl)
        #--use_MotionCorr=0---> checkBox
        self.set_use_MotionCorr(self.use_MotionCorr_hdl)
        #--use_CTFfind=0---> checkBox
        self.set_use_CTFfind(self.use_CTFfind_hdl)
        #--use_unblur=0---> checkBox
        self.set_use_unblur(self.use_unblur_hdl)
        #--file_pre_handler=preprocess---> checkBox
        self.set_file_pre_handler(self.file_pre_handler_hdl)
        #-----------------------------------------------------------------------
        # Setting all input values into the DataManageCommon object
        #-----------------------------------------------------------------------
        input_app_root = self.app_root.get()
        self.c.setApp_root(input_app_root)
        input_projectName = self.projectName.get()
        self.c.setProjectName(input_projectName)
        input_data_path = self.data_path.get()
        self.c.setData_path(input_data_path)
        input_gainmrc = self.gainRef.get()
        self.c.setGainmrc(input_gainmrc)
        input_targetdir = self.targetDirectory.get()
        self.c.setTargetdir(input_targetdir)
        #software = self.software_SP
        software = self.c.getSoftware #self.software_SP
        #self.c.setSoftware(software)
        #logical setter doBenchMarking = self.doBenchMarking.get()    
        self.c.setBench_cpu(self.doBenchMarking.get())
        #Marker setter
        if self.doBenchMarking.get() == True: doBenchMarking= "--doBenchMarking"
        if self.doBenchMarking.get() == False: doBenchMarking = ""
        self.c.setdoBenchMarking_marker(doBenchMarking)
        #useGpu = self.use_gpu.get()
        if self.use_gpu.get() == True:  useGpu = "yes"
        if self.use_gpu.get() == False: useGpu = "no"
        self.c.setUse_gpu(useGpu)
        #gpuList = self.gpu_list.get()
        gpuList = self.gpu_list.get()
        self.c.setGpu_list(gpuList)
        # poolify = self.poolify.get()
        if self.poolify.get() == True:  poolify = "yes"
        if self.poolify.get() == False: poolify = "no"
        self.c.setPoolify(poolify)
        #superRes = self.super_res.get()
        if self.super_res.get() == True: superRes = "yes"
        if self.super_res.get() == False: superRes = "no"
        self.c.setSuper_Res(superRes)
        #--movies_ext=tiff---> radioButton
        movies_ext = self.movies_ext
        self.c.setMovies_ext(movies_ext)
        #--live_process=yes---> checkBox
        live_process = self.live_process
        self.c.setLive_process(live_process)
        #--use_MotionCorr=0---> checkBox
        use_MotionCorr = self.use_MotionCorr
        self.c.setUse_MotionCorr(use_MotionCorr)
        #--use_CTFfind=0---> checkBox
        use_CTFfind = self.use_CTFfind
        self.c.setUse_CTFfind(use_CTFfind)
        #--use_unblur=0---> checkBox
        use_unblur = self.use_unblur
        self.c.setUse_unblur(use_unblur)

        #--file_frame_copy_move_handler=copy---> radioButton
        file_frame_copy_move_handler = self.file_frame_copy_move_handler
        self.c.setFile_frame_copy_move_handler(file_frame_copy_move_handler)

        #--file_pre_handler=preprocess---> checkBox
        file_pre_handler = self.file_pre_handler
        self.c.setFile_pre_handler(file_pre_handler)
        #--file_organise_dir=re---> Combo box
        file_organise_dir = self.file_organise_dir
        self.c.setFile_organise_dir(file_organise_dir)
        #--file_pst_handler=extract_data---> combo box
        file_pst_handler = self.file_pst_handler
        self.c.setFile_pst_handler(file_pst_handler)
        #--motioncorr_dir=MOTIONCORR_DIR---> textbox ttk.Entry
        motioncorr_dir = self.MotionCorrDir.get()
        self.c.setMotionCorr_dir(motioncorr_dir)
        #--ctffind_dir=CTFFIND_DIR---> textbox ttk.Entry
        ctffind_dir = self.CTFfindDir.get()
        self.c.setCTFFind_dir(ctffind_dir)
        #--nframes=NFRAMES              [50] TODO: Include number frames unblur
        nframes = self.nframes.get()
        self.c.setNframes(nframes=nframes)
        #--sph_abe=SPH_ABE              [2.7] Sphererical Aberration
        sph_abe = self.sph_abe.get()
        self.c.setSph_Abe(sph_abe=sph_abe)
        #--amp_con=AMP_CON              [0.10] Ampl constrast (from 0.07 06Nov21)
        amp_con = self.amp_con.get()
        self.c.setAmp_Con(amp_con=amp_con)
        #--sze_pwr_spc=SZE_PWR_SPC      [512] Size of prower spectrum
        sze_pwr_spc = self.sze_pwr_spc.get()
        self.c.setSze_Pwr_Spc(sze_pwr_spc=sze_pwr_spc)
        #--minres=MINRES                [30.0] Minimum resolution
        minres = self.minres.get()
        self.c.setMinRes(minres=minres)
        #--maxres=MAXRES                [5.0] Maximum resolution
        maxres = self.maxres.get()
        self.c.setMaxRes(maxres=maxres)
        #--mindef=MINDEF                [5000] Minimum defocus "3100.0"06Nov21
        mindef = self.mindef.get()
        self.c.setMinDef(mindef=mindef)
        #--maxdef=MAXDEF                [50000] Maximum defocus "6900.0"06Nov21
        maxdef = self.maxdef.get()
        self.c.setMaxDef(maxdef=maxdef)
        #--defstep=DEFSTEP              [500.0] Defocus search step
        defstep = self.defstep.get()
        self.c.setDefStep(defstep=defstep)
        #--astigpenaltyover=ASTIGPENALTYOVER   [1500.0] Expected \(tolerated\) astig
        astigpenaltyover = self.astigpenaltyover.get()
        self.c.setAstigPenaltyOver(astigpenaltyover=astigpenaltyover)
        #--fm_dose=string--->textfield
        fm_dose = self.fm_dose.get()
        self.c.setFMDose(fm_dose)
        #--bfactor=150 int--->textfield
        bfactor = self.bfactor.get()
        self.c.setBfactor(bfactor)
        #--binning_factor=1 int--->textfield
        binning_factor = self.binning_factor.get()
        self.c.setBinning_factor(binning_factor)
        #--gain_rotation=0 int--->textfield
        gain_rotation = self.gain_rotation.get()
        self.c.setGain_rotation(gain_rotation)
        #--gain_flip=0 int--->textfield
        gain_flip = self.gain_flip.get()
        self.c.setGain_flip(gain_flip)
        #-----------------------------------------------------------------------
        # Starting the StateMachine
        #-----------------------------------------------------------------------
        global running
        running = True
        self.scanning()

        #declared but not needed just yet may need to turn this into a self
        showGUI = True

        return
    #---------------------------------------------------------------------------
    # [quit] method for quiting the application
    #---------------------------------------------------------------------------
    ##\brief Python3 method.
    # DataManageApp class method to quit the application
    # \param self     The self Object
    def quit(self):
        self.m.printCMesg("DataManage exits without computing.", self.c.get_B_Red())
        self.m.printMesgInt("Return code: ", self.c.get_B_Red(),self.c.get_RC_STOP())
        exit(self.c.get_RC_STOP())

    ##\brief Python3 method.
    # DataManageApp class method to show the about popup
    # \param self     The self Object
    def showAbout(self):
        __func__= sys._getframe().f_code.co_name
        showinfo("About LDMaP-App",
        ("This is DataManage part of LDMaP-App package, v."+version+".\n\n"
         "If you use DataManage in your work, please cite the following:\n"\
         "https://sourceforge.net/projects/ldmap/\n"\
         "This package is released under the Creative Commons " \
         "Attribution-NonCommercial-NoDerivs CC BY-NC-ND License " \
         "(http://creativecommons.org/licenses/by-nc-nd/3.0/)\n\n"
         "Please send comments, suggestions, and bug reports to " \
         "fbonnet08@gmail.com"))

    ##\brief Python3 method.
    # DataManageApp class method to show the documentation popup
    # \param self     The self Object
    def showDocumentation(self):
        __func__= sys._getframe().f_code.co_name
        showinfo("DataManage Documentation and manual",
            "can be download from https://sourceforge.net/projects/ldmap/")

#-------------------------------------------------------------------------------
# Main code of the application DataManage
#-------------------------------------------------------------------------------
##\brief Python3 method.
# DataManagePreProcApp application main
if __name__ == '__main__':

    __func__= sys._getframe().f_code.co_name
    global version
    #printing the header
    version = DataManage_version()
    #instantiating the common class
    c = DataManage_common()
    #getting the return code for the start of the execution
    rc = c.get_RC_SUCCESS()
    #instantiating messaging class
    logfile = c.getLogfileName()  #getting the name of the global log file
    m = utils.messageHandler.messageHandler(logfile = logfile)#messageHandler(logfile = logfile)
    #printing the header of Application
    print_header(common=c, messageHandler=m)
    #getting stuff on screen
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

    if args['--nogui'] == False:

        # Create root window and initiate GUI interface
        root = tk.Tk()
        datamanageapp = DataManagePreProcApp(root)
        root.mainloop()
    else:
        #-----------------------------------------------------------------------
        # Command line handler for now. TODO: need to immplement class
        #-----------------------------------------------------------------------
        #--app_root=APP_ROOT {required}
        if args['--app_root']:
            APP_ROOT = args['--app_root']
        else:
            m.printCMesg("A root path for program is required.",c.get_B_Red())
            m.printMesgAddStr("app_root: ",c.getGreen(),args['--app_root'])
            m.printMesgInt("Return code: ",c.get_B_Red(),c.get_RC_FAIL())
            exit(c.get_RC_FAIL())

        #--projectName=PROJECTNAME Name of the project folder
        if args['--projectName']:
            PROJECTNAME = args['--projectName']
        else:
            m.printCMesg("A projectName for program is required.",c.get_B_Red())
            m.printMesgAddStr("projectName: ",c.getGreen(),args['--projectName'])
            m.printMesgInt("Return code: ",c.get_B_Red(),c.get_RC_FAIL())
            exit(c.get_RC_FAIL())

        #--data_path=DATA_PATH {required}
        if args['--data_path']:
            DATA_PATH = args['--data_path']
        else:
            m.printCMesg("A data path for program is required.",c.get_B_Red())
            m.printMesgAddStr("data_path: ",c.getGreen(),args['--data_path'])
            m.printMesgInt("Return code: ",c.get_B_Red(),c.get_RC_FAIL())
            exit(c.get_RC_FAIL())

        #--gainmrc=GAINMRC {required}
        if args['--gainmrc']:
            GAINMRC = args['--gainmrc']
        else:
            m.printCMesg("A gainref mrc is required generally.",c.get_B_Red())
            m.printMesgAddStr("--gainmrc: ",c.getGreen(),args['--gainmrc'])
            m.printMesgInt("Return code: ",c.get_B_Red(),c.get_RC_FAIL())
            m.printCMesg("Proceeding wihtout a gainreference.",c.get_B_Red())
            GAINMRC = "no_input_GAINMRC"
            #exit(c.get_RC_FAIL())

        #--targetdir=TARGETDIR {required}
        if args['--targetdir']:
            TARGETDIR = args['--targetdir']
        else:
            m.printCMesg("A target dir path for program is required.",c.get_B_Red())
            m.printMesgAddStr("--targetdir: ",c.getGreen(),args['--targetdir'])
            m.printMesgInt("Return code: ",c.get_B_Red(),c.get_RC_FAIL())
            exit(c.get_RC_FAIL())

        #--software=SOFTWARE {epu,SerialEM|dflt=epu}
        if args['--software']:
            SOFTWARE = args['--software']
        else:
            SOFTWARE = "epu"
            m.printCMesg("Acquisition software not set --> EPU",c.get_B_Yellow())
            m.printMesgAddStr("--software: ",c.getGreen(),SOFTWARE)
            m.printMesgInt("Return code: ",c.get_B_Yellow(),c.get_RC_WARNING())
        #---------------------------------------------------------------------------
        # Setting the variable into the common class
        #---------------------------------------------------------------------------
        c.setApp_root(APP_ROOT)
        c.setProjectName(PROJECTNAME)
        c.setData_path(DATA_PATH)
        c.setGainmrc(GAINMRC)
        c.setTargetdir(TARGETDIR)
        c.setSoftware(SOFTWARE)
        #---------------------------------------------------------------------------
        # Setting the variable into the common class
        #---------------------------------------------------------------------------
        l = utils.Command_line.Command_line(args, c, m)

        #--movies_ext=MOVIES_EXT     Movies extension [{mrc|tiff}|dflt=mrc}]
        l.createArgs_movies_ext()
        #--file_frame_copy_move_handler=FILE_FRAME_COPY_MOVE_HANDLER [{copy,move,rsync_copy,rsync_move}|copy]
        l.createArgs_file_frame_copy_move_handler()
        #--super_res=SUPER_RES       Super resolution mode [{yes|no}|dflt=no}]
        l.createArgs_super_res()
        #--motioncorr_dir=MOTIONCORR_DIR     Path to Motion Corr executable
        l.createArgs_motioncorr_dir()

        #--nframes=NFRAMES              [50] TODO: Include number frames unblur
        l.createArgs_Nframes()
        #--sph_abe=SPH_ABE              [2.7] Sphererical Aberration
        l.createArgs_Sph_Abe()
        #--amp_con=AMP_CON              [0.10] Ampl constrast (from 0.07 06Nov21)
        l.createArgs_Amp_Con()
        #--sze_pwr_spc=SZE_PWR_SPC      [512] Size of prower spectrum
        l.createArgs_Sze_Pwr_Spc()
        #--minres=MINRES                [30.0] Minimum resolution
        l.createArgs_MinRes()
        #--maxres=MAXRES                [5.0] Maximum resolution
        l.createArgs_MaxRes()
        #--mindef=MINDEF                [5000] Minimum defocus "3100.0"06Nov21
        l.createArgs_MinDef()
        #--maxdef=MAXDEF                [50000] Maximum defocus "6900.0"06Nov21
        l.createArgs_MaxDef()
        #--defstep=DEFSTEP              [500.0] Defocus search step
        l.createArgs_DefStep()
        #--astigpenaltyover=ASTIGPENALTYOVER   [1500.0] Expected \(tolerated\) astig
        l.createArgs_AstigPenaltyOver()

        #--fm_dose=FM_DOSE            Dose per frames [{float}|1.0]
        l.createArgs_fm_dose()
        #--bfactor=BFACTOR                Bfactor
        l.createArgs_bfactor()
        #--binning_factor=BINNING_FACTOR  Binning factor
        l.createArgs_binning_factor()
        #--gain_rotation=GAIN_ROTATION    cntr-clockwise [{0:0,1:90,2:180,3:270}|0]
        l.createArgs_gain_rotation()
        #--gain_flip=GAIN_FLIP            [{0:no, 1:flip UpDown, 2:flip LR}|0]
        l.createArgs_gain_flip()
        #--ctffind_dir=CTFFIND_DIR           Path to CTFFind executable
        l.createArgs_ctffind_dir()
        #--live_process=LIVE_PROCESS Live processing [{yes|no}|dflt=no}]
        l.createArgs_live_process()
        #--use_MotionCorr=USE_MCORR {yes:1|dflt=no:0}
        l.createArgs_use_MotionCorr()
        #--use_CTFfind=USE_CTFFIND {yes:1|dflt=no:0}
        l.createArgs_use_CTFfind()
        #--file_pre_handler=F_PRE_HDLR
        l.createArgs_file_pre_handler()
        #--file_organise_dir=F_ORGS_DIR
        l.createArgs_file_organise_dir()
        #--file_pst_handler=F_PST_HDLR
        l.createArgs_file_pst_handler()
        #--use_gpu=USE_GPU
        l.createArgs_use_gpu()
        #--gpu_list=GPU_LIST
        l.createArgs_gpu_list()
        #--poolify=POOLIFY    Poolify the data {yes|no} [dflt=yes]
        l.createArgs_poolify()
        # --doBenchMarking
        l.createArgs_doBenchMarking()

        # Call DataManage_live-PreProcessing
        if args['--nogui'] == True:
            m.printMesg("Starting non GUI DataManage_live-PreProcessing...")
            #-------------------------------------------------------------------
            # Starting the timers for the preprocessing computation
            #-------------------------------------------------------------------
            #creating the timers and starting the stop watch...
            stopwatch = utils.StopWatch.createTimer()
            if c.getBench_cpu():
                utils.StopWatch.StartTimer(stopwatch)
            #-------------------------------------------------------------------
            # Starting preprocessing with DataManage_live-PreProcessing
            #-------------------------------------------------------------------
            #Instert code here for the wrarper code of the data live processing
            my_time = time.asctime()    #clock_gettime(threading.get_ident())
            m.printMesgAddStr(" The preprocessing computation is starting: ",
                              c.getRed(),str(my_time))

            in_list = []
            live_ ="--app_root="+c.getApp_root()
            in_list.append(live_)
            live_ ="--projectName="+c.getProjectName()
            in_list.append(live_)
            live_ ="--data_path="+c.getData_path()
            in_list.append(live_)
            live_ ="--gainmrc="+c.getGainmrc()
            in_list.append(live_)
            live_ ="--targetdir="+c.getTargetdir()
            in_list.append(live_)
            live_ ="--poolify="+c.getPoolify()
            in_list.append(live_)
            live_ ="--software="+c.getSoftware()
            in_list.append(live_)
            live_ ="--use_gpu="+c.getUse_gpu()
            in_list.append(live_)
            live_ ="--gpu_list="+c.getGpu_list()
            in_list.append(live_)
            live_ ="--super_res="+c.getSuper_Res()
            in_list.append(live_)

            live_ ="--nframes="+str(c.getNframes())
            in_list.append(live_)
            live_ ="--sph_abe="+str(c.getSph_Abe())
            in_list.append(live_)
            live_ ="--amp_con="+str(c.getAmp_Con())
            in_list.append(live_)
            live_ ="--sze_pwr_spc="+str(c.getSze_Pwr_Spc())
            in_list.append(live_)
            live_ ="--minres="+str(c.getMinRes())
            in_list.append(live_)
            live_ ="--maxres="+str(c.getMaxRes())
            in_list.append(live_)
            live_ ="--mindef="+str(c.getMinDef())
            in_list.append(live_)
            live_ ="--maxdef="+str(c.getMaxDef())
            in_list.append(live_)
            live_ ="--defstep="+str(c.getDefStep())
            in_list.append(live_)
            live_ ="--astigpenaltyover="+str(c.getAstigPenaltyOver())
            in_list.append(live_)
            
            live_ ="--fm_dose="+str(c.getFMDose())
            in_list.append(live_)
            live_ ="--bfactor="+str(c.getBfactor())
            in_list.append(live_)
            live_ ="--binning_factor="+str(c.getBinning_factor())
            in_list.append(live_)
            live_ ="--gain_rotation="+str(c.getGain_rotation())
            in_list.append(live_)
            live_ ="--gain_flip="+str(c.getGain_flip())
            in_list.append(live_)
            live_ ="--movies_ext="+c.getMovies_ext()
            in_list.append(live_)
            live_ ="--live_process="+c.getLive_process()
            in_list.append(live_)
            live_ ="--use_unblur="+str(c.getUse_unblur())
            in_list.append(live_)
            live_ ="--use_MotionCorr="+str(c.getUse_MotionCorr())
            in_list.append(live_)
            live_ ="--use_CTFfind="+str(c.getUse_CTFfind())
            in_list.append(live_)
            live_ = "--file_frame_copy_move_handler="+c.getFile_frame_copy_move_handler()
            in_list.append(live_)
            live_ ="--file_pre_handler="+c.getFile_pre_handler()
            in_list.append(live_)
            live_ ="--file_organise_dir="+c.getFile_organise_dir()
            in_list.append(live_)
            live_ ="--file_pst_handler="+c.getFile_pst_handler()
            in_list.append(live_)
            live_ = "--motioncorr_dir="+c.getMotionCorr_dir()
            in_list.append(live_)
            live_ = "--ctffind_dir="+c.getCTFFind_dir()
            in_list.append(live_)
            live_ =""+c.getdoBenchMarking_marker()
            in_list.append(live_)
            
            CMD = ' '.join(in_list[:])
            m.printMesgAddStr(" python3 DataManage_live-PreProcessing.py ",
                              c.getCyan(), CMD)

            #-------------------------------------------------------------------
            # [Sleeping] sleeping time between scans
            #-------------------------------------------------------------------
            #time.sleep(10)
            SLEEP_TIME = 10
            m.printMesg("Sleeping before launching DataManage_live-PreProcessing...")
            if float(SLEEP_TIME) < 1:
                m.printMesgAddStr(" No Sleep between each sweeps: ", c.getGreen(),
                                  str(SLEEP_TIME) + " seconds")
            if float(SLEEP_TIME) >= 1:
                progressBar = utils.progressBar.ProgressBar()
                for i in range(int(SLEEP_TIME)):
                    time.sleep(1)
                    progressBar.update(1, int(SLEEP_TIME))
                    progressBar.printEv()
                m.printLine()
                progressBar.resetprogressBar()

            # Call DataManage_live-PreProcessing
            '''
            DataManage_live-PreProcessing(input_data_path   = input_data_path,
                                          input_gainmrc     = input_gainmrc,

                                          etc...
                                          
                                          doBenchMarking   = doBenchMarking,
                                          chimeraBin       = chimeraBin,
                                          showGUI          = showGUI,
                                          use_gpu          = c.getUse_gpu(),
                                          set_gpu          = c.getSet_gpu(),
                                          lib              = c.getLibfileName() )
            '''
            #TODO: unlock the system call once it works
            os.system('python3 DataManage_live-PreProcessing.py %s'%CMD)
            #-------------------------------------------------------------------
            # Reporting time taken to instantiate and strip innitial star file
            #-------------------------------------------------------------------
            if c.getBench_cpu():
                utils.StopWatch.StopTimer_secs(stopwatch)
                info = c.GetFrameInfo()
                m.printBenchMap("data_path",c.getRed(),
                                     c.getData_path(), info, __func__)
                m.printBenchTime_cpu("Read data_path file",
                                          c.getBlue(), stopwatch,
                                          info, __func__)
            #-------------------------------------------------------------------
            # [Final] ovrall return code
            #-------------------------------------------------------------------
            getFinalExit(c,m,rc)
        else:
            rc = c.get_RC_FAIL()
            #final ovrall return code
            getFinalExit(c,m,rc)
            m.printCMesg("Invalid entry for DataManage_live-PreProcessing: DataManage_live-preProcessing.py!!",c.get_B_Red())
#-------------------------------------------------------------------------------
# end of DataManage application
#-------------------------------------------------------------------------------
