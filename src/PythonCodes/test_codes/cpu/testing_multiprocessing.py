import multiprocessing
import subprocess
import os
import sys
import tkinter as tk
from queue import Empty, Full
# ################################################################################
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


def LaunchGainRefCopyHandler(name, q1):
    __func__= sys._getframe().f_code.co_name
    q1.put("Function                             : "+__func__+"\n")
    q1.put("Launching GainRef copy with gain type " + \
           "gain_type=[{x0.m0, x0.m1, x1.m0, x1.m1}|x0.m0}] handler...\n")
    #FramesHandler_DriverCode_fileMover.py
    #GainRefHandler_DriverCode_fileMover.py
    print("Hello ", name)
    #os.system('python3 -c "print(\"Hello multiprocessing ...\")" ')
    return
# ################################################################################
# Main
# ################################################################################
if __name__ == '__main__':

    root = tk.Tk()
    print("This is a multiprocessing test with sub rpocess")

    q1 = multiprocessing.Queue()
    q1.cancel_join_thread()
    gui_GainRef = GainRefGuiApp(q1)

    t1 = multiprocessing.Process(target=LaunchGainRefCopyHandler, args=('LaunchGainRefCopyHandler', q1,))
    t1.start()
    t1.join()
    root.mainloop()

