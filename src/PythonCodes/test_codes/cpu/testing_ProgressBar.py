import os
import sys
import time
# path extension
sys.path.append(os.path.join(os.getcwd(), '.'))
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append(os.path.join(os.getcwd(), '..', '..',))
sys.path.append(os.path.join(os.getcwd(), '..', '..', '..'))
sys.path.append(os.path.join(os.getcwd(), '..', '..', '..', '..'))
sys.path.append(os.path.join(os.getcwd(), '.', 'src', 'PythonCodes'))
sys.path.append(os.path.join(os.getcwd(), '.', 'src', 'PythonCodes', 'utils'))

print(os.getcwd())

import src.PythonCodes.DataManage_common
import src.PythonCodes.utils.messageHandler
import src.PythonCodes.utils.StopWatch
import src.PythonCodes.utils.progressBar

if __name__ == "__main__":
    __func__= sys._getframe().f_code.co_name
    global version
    version = src.PythonCodes.DataManage_common.DataManage_version()
    c = src.PythonCodes.DataManage_common.DataManage_common()
    rc = c.get_RC_SUCCESS()
    # Getting the log file
    logfile = c.getLogfileName()
    m = src.PythonCodes.utils.messageHandler.messageHandler(logfile = logfile)

    n_sleep = 10
    progressBar = src.PythonCodes.utils.progressBar.ProgressBar()
    for i in range(n_sleep):
        time.sleep(1)
        progressBar.update(1, n_sleep)
        progressBar.printEv()
    # [end-loop]
    m.printLine()

    # ---------------------------------------------------------------------------
    # [Final] overall return code
    # ---------------------------------------------------------------------------
    # Final exit statements
    src.PythonCodes.DataManage_common.getFinalExit(c, m, rc)
    # ---------------------------------------------------------------------------
    # End of testing script
    # ---------------------------------------------------------------------------

