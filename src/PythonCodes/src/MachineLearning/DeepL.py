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
import src.PythonCodes.src.MachineLearning.CNN
# Definiiton of the constructor
class DeepL:
    #---------------------------------------------------------------------------
    # [Constructor] for the
    #---------------------------------------------------------------------------
    # Constructor
    def __init__(self, c, m, db_action="train_dataset_from_db"):
        __func__= sys._getframe().f_code.co_name
        self.rc = 0
        self.c = c
        self.m = m
        self.app_root = self.c.getApp_root()
        self.m.printMesgStr("Instantiating the class       :", self.c.getGreen(), "DeepL")

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

        self.database_path = os.path.dirname(self.database_full_path)
        self.c.setDatabase_path(self.database_path)

        self.database_full_path_no_ext = os.path.splitext(self.database_full_path)[0]
        self.c.setDatabase_full_path_no_ext(self.database_full_path_no_ext)
        self.database_file = os.path.basename(self.database_full_path)
        self.c.setDatabase_file(self.database_file)
        self.basename = self.database_file.split('.')[0]
        self.c.setFile_basename(self.basename)
        self.c.setDatabase_name(str(self.basename).replace(' ', '_'))
        ext = ""
        if len(os.path.splitext(self.database_full_path)) > 1: ext = os.path.splitext(self.database_full_path)[1]

        # print("basename: ", basename)
        if self.c.getDebug() == 1:
            self.m.printMesgAddStr("Database directory name    --->: ", self.c.getMagenta(), self.database_path)
            self.m.printMesgAddStr("database_full_path_no_ext  --->: ", self.c.getMagenta(), self.database_full_path_no_ext)
            self.m.printMesgAddStr("Database file              --->: ", self.c.getMagenta(), self.database_file)
            self.m.printMesgAddStr("Database name              --->: ", self.c.getMagenta(), self.c.getDatabase_name())
            self.m.printMesgAddStr("File basename              --->: ", self.c.getMagenta(), self.basename)
            self.m.printMesgAddStr("ext                        --->: ", self.c.getMagenta(), ext)
        # [end-if] self.c.getDebug() == 1:
        self.file_csv = self.basename+self.ext_csv
        self.c.setCSV_file(self.file_csv)
        self.file_asc = self.basename+self.ext_asc
        self.c.setPlot_asc(self.file_asc)

        if self.c.getDebug() == 1:
            self.m.printMesgAddStr("the csv file is then       --->: ", self.c.getMagenta(), self.file_csv)
            # printing the files:
            self.printFileNames()
        # [end-if] self.c.getDebug() == 1:
        #-----------------------------------------------------------------------
        # [Instantiating] Instantiating the CNN Convolution Neural Network class
        #-----------------------------------------------------------------------
        rc = self.check_torch_cuda()
        #-----------------------------------------------------------------------
        # [INstantiating] Instantiating the CNN Convolution Neural Network class
        #-----------------------------------------------------------------------
        if rc == self.c.get_RC_SUCCESS():
            self.convNeurNet = src.PythonCodes.src.MachineLearning.CNN.CNN(c, m, db_action=db_action)




        #-----------------------------------------------------------------------
        # [Constructor-end] end of the constructor
        #-----------------------------------------------------------------------
    #---------------------------------------------------------------------------
    # [Checkers]
    #---------------------------------------------------------------------------
    def check_torch_cuda(self):
        rc = self.c.get_RC_SUCCESS()
        __func__ = sys._getframe().f_code.co_name
        self.m.printMesgStr("Checking the Torch environment:", self.c.getGreen(), __func__)

        run_lambda = self.run
        cuda_runtime_version = self.get_running_cuda_version(run_lambda)
        self.m.printMesgAddStr("Cuda runtime version       --->: ", self.c.getCyan(), cuda_runtime_version)

        cudnn_verion = self.get_cudnn_version(run_lambda=run_lambda)
        self.m.printMesgAddStr("Cudnn probable version     --->: ", self.c.getYellow(), cudnn_verion)

        gpu_info = self.get_gpu_info(run_lambda=run_lambda)
        self.m.printMesgAddStr("Gpu info                   --->: ", self.c.getMagenta(), gpu_info)

        gpu_number = self.get_device_number(gpu_info)
        self.m.printMesgAddStr("Gpu device                 --->: ", self.c.getMagenta(), gpu_number)

        if isinstance(int(gpu_number), int):
            if int(gpu_number) >= 0:
                use_gpu = True
                self.c.setUse_gpu(use_gpu=use_gpu)
                self.c.setSet_gpu(gpu_number)
        # [end-if] statement isinstance(gpu_number, int):

        return rc
    #---------------------------------------------------------------------------
    # [Driver]
    #---------------------------------------------------------------------------
    def run(self, command):
        """Return (return-code, stdout, stderr)."""
        shell = True if type(command) is str else False
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell)
        raw_output, raw_err = p.communicate()
        rc = p.returncode
        #print(self.c.platform)
        if self.c.get_system() == 'Windows':
            enc = 'oem'
        else:
            enc = locale.getpreferredencoding()
        output = raw_output.decode(enc)
        err = raw_err.decode(enc)
        return rc, output.strip(), err.strip()
    def run_and_parse_first_match(self, run_lambda, command, regex):
        """Run command using run_lambda, returns the first regex match if it exists."""
        rc, out, _ = run_lambda(command)
        if rc != 0:
            return None
        match = re.search(regex, out)
        if match is None:
            return None
        return match.group(1)
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
    def get_device_number(self, gpu_info):
        device_number = 0
        if ':' in gpu_info:
            if 'GPU' in gpu_info:
                device_number = str(gpu_info.split(':')[0]).split('GPU')[1].strip()
        return device_number
    def get_running_cuda_version(self, run_lambda):
        return self.run_and_parse_first_match(run_lambda, 'nvcc --version', r'release .+ V(.*)')

    def get_nvidia_smi(self):
        # Note: nvidia-smi is currently available only on Windows and Linux
        smi = 'nvidia-smi'
        if self.c.get_system() == 'Windows':
            system_root = os.environ.get('SYSTEMROOT', 'C:\\Windows')
            program_files_root = os.environ.get('PROGRAMFILES', 'C:\\Program Files')
            legacy_path = os.path.join(program_files_root, 'NVIDIA Corporation', 'NVSMI', smi)
            new_path = os.path.join(system_root, 'System32', smi)
            smis = [new_path, legacy_path]
            for candidate_smi in smis:
                if os.path.exists(candidate_smi):
                    smi = '"{}"'.format(candidate_smi)
                    break
        return smi

    def get_nvidia_driver_version(self, run_lambda):
        if self.c.get_system() == 'Darwin':
            cmd = 'kextstat | grep -i cuda'
            return self.run_and_parse_first_match(run_lambda, cmd,
                                                  r'com[.]nvidia[.]CUDA [(](.*?)[)]')
        smi = self.get_nvidia_smi()
        return self.run_and_parse_first_match(run_lambda, smi, r'Driver Version: (.*?) ')

    def get_gpu_info(self, run_lambda):
        if self.c.get_system() == 'Darwin' or (TORCH_AVAILABLE and hasattr(torch.version, 'hip') and torch.version.hip is not None):
            if TORCH_AVAILABLE and torch.cuda.is_available():
                if torch.version.hip is not None:
                    prop = torch.cuda.get_device_properties(0)
                    if hasattr(prop, "gcnArchName"):
                        gcnArch = " ({})".format(prop.gcnArchName)
                    else:
                        gcnArch = "NoGCNArchNameOnOldPyTorch"
                else:
                    gcnArch = ""
                return torch.cuda.get_device_name(None) + gcnArch
            return None
        smi = self.get_nvidia_smi()
        uuid_regex = re.compile(r' \(UUID: .+?\)')
        rc, out, _ = run_lambda(smi + ' -L')
        if rc != 0:
            return None
        # Anonymize GPUs by removing their UUID
        return re.sub(uuid_regex, '', out)

    def get_cudnn_version(self, run_lambda):
        """Return a list of libcudnn.so; it's hard to tell which one is being used."""
        if self.c.get_system() == 'Windows':
            system_root = os.environ.get('SYSTEMROOT', 'C:\\Windows')
            cuda_path = os.environ.get('CUDA_PATH', "%CUDA_PATH%")
            where_cmd = os.path.join(system_root, 'System32', 'where')
            cudnn_cmd = '{} /R "{}\\bin" cudnn*.dll'.format(where_cmd, cuda_path)
        elif self.c.get_system() == 'Darwin':
            # CUDA libraries and drivers can be found in /usr/local/cuda/. See
            # https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html#install
            # https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installmac
            # Use CUDNN_LIBRARY when cudnn library is installed elsewhere.
            cudnn_cmd = 'ls /usr/local/cuda/lib/libcudnn*'
        else:
            cudnn_cmd = 'ldconfig -p | grep libcudnn | rev | cut -d" " -f1 | rev'
        rc, out, _ = run_lambda(cudnn_cmd)
        # find will return 1 if there are permission errors or if not found
        if len(out) == 0 or (rc != 1 and rc != 0):
            l = os.environ.get('CUDNN_LIBRARY')
            if l is not None and os.path.isfile(l):
                return os.path.realpath(l)
            return None
        files_set = set()
        for fn in out.split('\n'):
            fn = os.path.realpath(fn)  # eliminate symbolic links
            if os.path.isfile(fn):
                files_set.add(fn)
        if not files_set:
            return None
        # Alphabetize the result because the order is non-deterministic otherwise
        files = sorted(files_set)
        if len(files) == 1:
            return files[0]
        result = '\n'.join(files)
        return result

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
    # [Checkers]
    #---------------------------------------------------------------------------
    def test_torch_cuda(self):
        rc = self.c.get_RC_SUCCESS()
        __func__ = sys._getframe().f_code.co_name
        self.m.printMesgStr("Testing torch cuda            :", self.c.getGreen(), __func__)

        # checking the torch environment
        dtype = torch.float
        device = torch.device("cuda:0")

        # Create random input and output data
        x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
        y = torch.sin(x)

        # Randomly initialize weights
        a = torch.randn((), device=device, dtype=dtype)
        b = torch.randn((), device=device, dtype=dtype)
        c = torch.randn((), device=device, dtype=dtype)
        d = torch.randn((), device=device, dtype=dtype)

        rc, zerolead = self.extract_ZeroLead(numpy.power(10,7))
        learning_rate = 1e-6
        for t in range(3000):
            # Forward pass: compute predicted y
            y_pred = a + b * x + c * x ** 2 + d * x ** 3

            # Compute and print loss
            loss = (y_pred - y).pow(2).sum().item()
            if t % 100 == 99:
                self.m.printMesgAddStr("The iterator t ---->: "+\
                                       self.c.getGreen()+str(t).zfill(len(str(zerolead)))+self.c.getBlue()+
                                       "    --->: ",
                                       self.c.getYellow(), str('{:012.8f}'.format(float(loss))).zfill(len(str(zerolead))))
                #print(t, loss)

            # Backprop to compute gradients of a, b, c, d with respect to loss
            grad_y_pred = 2.0 * (y_pred - y)
            grad_a = grad_y_pred.sum()
            grad_b = (grad_y_pred * x).sum()
            grad_c = (grad_y_pred * x ** 2).sum()
            grad_d = (grad_y_pred * x ** 3).sum()

            # Update weights using gradient descent
            a -= learning_rate * grad_a
            b -= learning_rate * grad_b
            c -= learning_rate * grad_c
            d -= learning_rate * grad_d
        # [end-loop] for t in ragne(2000)
        #print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
        msg = f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3'
        self.m.printMesgAddStr("Results ---->: ", self.c.getYellow(), str(msg))

        return rc
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
