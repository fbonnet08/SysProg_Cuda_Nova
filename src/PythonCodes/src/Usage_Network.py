#!/usr/bin/env python3
'''!\file
   -- DataManage addon: (Python3 code) class for handling some of the command line
      of the common variables
      \author Frederic Bonnet
      \date 27th of February 2024

      Universite de Perpignan March 2024

Name:
---
Command_line: class Usage_Network for extracting network transfers between
two interfaces.

Description of classes:
---
This class generates an object that contains and handlers

Requirements (system):
---
* sys
* psutil
* time
* os
* pandas
* datetime
'''
# System imports
import sys
import psutil
import time
import os
import pandas as pd
from datetime import datetime
# Path extension
sys.path.append(os.path.join(os.getcwd(), '.'))
sys.path.append(os.path.join(os.getcwd(), '..'))
#Application imports
#import src.PythonCodes.DataManage_common
# Global variables
UPDATE_DELAY = 1     # (secs)
# Definiiton of the constructor
class Usage_Network:
    # Constructor
    def __init__(self, c, m):
        __func__= sys._getframe().f_code.co_name
        self.rc = 0
        self.c = c
        self.m = m
        self.app_root = self.c.getApp_root()  #app_root
        self.m.printMesgStr("Instantiating the class : ", self.c.getGreen(), "Usage_Network")

# get_size(self, bytes)
    def get_size(self, bytes):
        # Returns size of bytes in a nice format
        for unit in ['', 'K', 'M', 'G', 'T', 'P']:
            if bytes < 1024:
                return f"{bytes:.2f}{unit}B"
            bytes /= 1024
    # extract_speed(self, speed_string)
    def extract_speed(self, speed_string):
        # Split the string based on the last occurrence of "/"
        parts = speed_string.split("/")[-1].strip()
        #print("parts before: ", parts)
        if parts == "s": parts = speed_string.split("B/s")[0].strip()
        numerical_value = 0
        unit = "B/s"
        #print("parts after: ", parts)
        # Check if the string contains KB/s
        if "KB/s" in parts:
            numerical_value = float(parts.split("KB/s")[0])
            unit = "KB/s"
        # Check if the string contains MB/s
        elif "MB/s" in parts:
            numerical_value = float(parts.split("MB/s")[0])
            unit = "MB/s"
        elif "K" in parts:
            numerical_value = float(parts.split("K")[0])
            unit = "KB/s"
        elif "M" in parts:
            numerical_value = float(parts.split("M")[0])
            unit = "MB/s"
        elif "s" in parts:
            # Assuming it is in B/s by default if no KB/s or MB/s is found
            numerical_value = float(parts.split("s")[0])
            unit = "B/s"

        return numerical_value, unit
    # get_MBytess(self, speed, units)
    def get_MBytess(self, speed, units):
        speed_MBs = 0
        if units == "B/s":
            speed_MBs = speed / 1000000
        if units == "KB/s":
            speed_MBs = speed / 1000
        if units == "MB/s":
            speed_MBs = speed

        return speed_MBs
    # Usage_NetworkInterfaces(self, )
    def Usage_NetworkInterfaces_whileLoop(self, log_file):
        rc = self.c.get_RC_SUCCESS()
        # get the network I/O stats from psutil on each network interface
        # by setting `pernic` to `True`
        io = psutil.net_io_counters(pernic=True)
        
        cnt = 0
        while True:
            cnt += 1
            # sleep for `UPDATE_DELAY` seconds
            time.sleep(UPDATE_DELAY)
            # get the network I/O stats again per interface
            io_2 = psutil.net_io_counters(pernic=True)
            # initialize the data to gather (a list of dicts)
            data = []
            for iface, iface_io in io.items():
                # new - old stats gets us the speed
                upload_speed, download_speed = io_2[iface].bytes_sent - iface_io.bytes_sent, io_2[iface].bytes_recv - iface_io.bytes_recv
                data.append({
                    "iface": iface,
                    "Download": self.get_size(io_2[iface].bytes_recv),
                    "Upload": self.get_size(io_2[iface].bytes_sent),
                    "Upload Speed": f"{self.get_size(upload_speed / UPDATE_DELAY)}/s",
                    "Download Speed": f"{self.get_size(download_speed / UPDATE_DELAY)}/s",
                })
            # update the I/O stats for the next iteration
            io = io_2
            # construct a Pandas DataFrame to print stats in a cool tabular style
            df = pd.DataFrame(data)
            # sort values per column, feel free to change the column
            df.sort_values("Download", inplace=True, ascending=False)
            # clear the screen based on your OS
            os.system("cls") if "nt" in os.name else os.system("clear")
            # print the stats
            self.m.printCMesgVal("" , self.c.getYellow(), df.to_string())
            
            ethernet_data = next((item for item in data if item["iface"] == "Ethernet"), None)
            self.m.printCMesgVal("Ethr --->: ", self.c.getMagenta(), ethernet_data)
            wlan_data = next((item for item in data if item["iface"] == "WLAN"), None)
            self.m.printCMesgVal("WLAN --->: ", self.c.getRed(), wlan_data)
            vEthernet_data = next((item for item in data if item["iface"] == "vEthernet (WSL)"), None)
            self.m.printCMesgVal("vEthr(WSL) --->: ", self.c.getRed(), vEthernet_data)

            if ethernet_data != None:
                upload_speed = ethernet_data["Upload Speed"]
                download_speed = ethernet_data["Download Speed"]
                self.m.printCMesgVal("Ethernet Upload Speed   --->: ", self.c.getGreen(), upload_speed)
                self.m.printCMesgVal("Ethernet Download Speed --->: ", self.c.getYellow(), download_speed)

                # Extracting only the numerical part
                upload_value, upload_unit = self.extract_speed(upload_speed)
                self.m.printCMesgValMesg(self.c.getBlue(),"upload    Numerical value   : ",
                                         self.c.getGreen(), '{:03.4f}'.format(upload_value), " Unit: "+self.c.getMagenta()+str(upload_unit))
                download_value, download_unit = self.extract_speed(download_speed)
                self.m.printCMesgValMesg(self.c.getBlue(),"Downsload Numerical value   : ",
                                         self.c.getYellow(), '{:03.4f}'.format(download_value), " Unit: "+self.c.getMagenta()+str(download_unit))

                upload_MBs = self.get_MBytess(upload_value, upload_unit)
                download_MBs = self.get_MBytess(download_value, download_unit)

                self.m.printCMesgValMesg(self.c.getBlue(),"upload    MBs               : ",
                                         self.c.getGreen(), '{:03.4f}'.format(upload_MBs), self.c.getMagenta()+" MB/s")
                self.m.printCMesgValMesg(self.c.getBlue(),"Downsload MBs               : ",
                                         self.c.getYellow(), '{:03.4f}'.format(download_MBs), self.c.getMagenta()+" MB/s")
            else:
                self.m.printMesg("Ethernet interface not found in the data.")

            if wlan_data != None:
                upload_speed = wlan_data["Upload Speed"]
                download_speed = wlan_data["Download Speed"]
                self.m.printCMesgVal("WLAN upload Speed       --->: ", self.c.getGreen(), upload_speed)
                self.m.printCMesgVal("WLAN Download Speed     --->: ", self.c.getYellow(), download_speed)

                # Extracting only the numerical part
                upload_value, upload_unit = self.extract_speed(upload_speed)
                self.m.printCMesgValMesg(self.c.getBlue(),"upload    Numerical value   : ",
                                         self.c.getGreen(), '{:03.4f}'.format(upload_value), " Unit: "+self.c.getMagenta()+str(upload_unit))
                download_value, download_unit = self.extract_speed(download_speed)
                self.m.printCMesgValMesg(self.c.getBlue(),"Downsload Numerical value   : ",
                                         self.c.getYellow(), '{:03.4f}'.format(download_value), " Unit: "+self.c.getMagenta()+str(download_unit))

                upload_MBs = self.get_MBytess(upload_value, upload_unit)
                download_MBs = self.get_MBytess(download_value, download_unit)

                self.m.printCMesgValMesg(self.c.getBlue(),"upload    MBs               : ",
                                         self.c.getGreen(), '{:03.4f}'.format(upload_MBs), self.c.getMagenta()+" MB/s")
                self.m.printCMesgValMesg(self.c.getBlue(),"Downsload MBs               : ",
                                         self.c.getYellow(), '{:03.4f}'.format(download_MBs), self.c.getMagenta()+" MB/s")
            else:
                self.m.printMesg("WLAN interface not found in the data.")

            #end of the if block
            # Now writing output to file
            # "2021-09-29_15:08:58","11.380353101754516E+21"
            current_time = datetime.now()
            format_time = current_time.strftime('%Y-%m-%d_%H:%M:%S')
            msg = ""
            try:
                csv_file = open(log_file, 'a')
                msg = "\"" + str(format_time)                     + "\"" + "," + \
                      "\"" + str('{:03.4f}'.format(upload_MBs))   + "\"" + "," + \
                      "\"" + str('{:03.4f}'.format(download_MBs)) + "\"" + "\n"
                csv_file.write(msg)
                #FileList =    csv_file.read().splitlines()
            except IOError:
                rc = self.c.get_RC_FAIL()
                self.m.printCMesg("Cannot open or no such file : "+csv_file, self.c.get_B_Red())
                self.c.getFinalExit(self.c, self.m,rc)
                exit(rc)
            csv_file.close()
            #if (cnt == ncnt): break
            # End of while loop
        
        return rc
    def Usage_NetworkInterfaces_onePass(self, log_file):
        rc = self.c.get_RC_SUCCESS()
        # get the network I/O stats from psutil on each network interface
        # by setting `pernic` to `True`
        io = psutil.net_io_counters(pernic=True)
        
        # sleep for `UPDATE_DELAY` seconds
        time.sleep(UPDATE_DELAY)
        # get the network I/O stats again per interface
        io_2 = psutil.net_io_counters(pernic=True)
        # initialize the data to gather (a list of dicts)
        data = []
        for iface, iface_io in io.items():
            # new - old stats gets us the speed
            upload_speed, download_speed = io_2[iface].bytes_sent - iface_io.bytes_sent, io_2[iface].bytes_recv - iface_io.bytes_recv
            data.append({
                "iface": iface,
                "Download": self.get_size(io_2[iface].bytes_recv),
                "Upload": self.get_size(io_2[iface].bytes_sent),
                "Upload Speed": f"{self.get_size(upload_speed / UPDATE_DELAY)}/s",
                "Download Speed": f"{self.get_size(download_speed / UPDATE_DELAY)}/s",
            })
        # update the I/O stats for the next iteration
        io = io_2
        # construct a Pandas DataFrame to print stats in a cool tabular style
        df = pd.DataFrame(data)
        # sort values per column, feel free to change the column
        df.sort_values("Download", inplace=True, ascending=False)
        # clear the screen based on your OS
        os.system("cls") if "nt" in os.name else os.system("clear")
        # print the stats
        self.m.printCMesgVal("" , self.c.getYellow(), df.to_string())
        # End of while loop
        ethernet_data = next((item for item in data if item["iface"] == "Ethernet"), None)
        self.m.printCMesgVal("Ethr --->: " , self.c.getMagenta(), ethernet_data)
        wlan_data = next((item for item in data if item["iface"] == "WLAN"), None)
        self.m.printCMesgVal("wlan --->: " , self.c.getRed(), wlan_data)
        
        if ethernet_data:
            upload_speed = ethernet_data["Upload Speed"]
            download_speed = ethernet_data["Download Speed"]
            self.m.printCMesgVal("Ethernet Upload Speed   --->: " , self.c.getYellow(), upload_speed)
            self.m.printCMesgVal("Ethernet Download Speed --->: " , self.c.getYellow(), download_speed)
            # Extracting only the numerical part
            upload_value, upload_unit = self.extract_speed(upload_speed)
            self.m.printCMesgValMesg(self.c.getBlue(),"upload    Numerical value   : ",
                                     self.c.getYellow(), upload_value, "Unit: "+str(upload_unit))
            download_value, download_unit = self.extract_speed(download_speed)
            self.m.printCMesgValMesg(self.c.getBlue(),"Downsload Numerical value   : ",
                                     self.c.getYellow(), download_value, "Unit: "+str(download_unit))
            
            upload_MBs = self.get_MBytess(upload_value, upload_unit)
            download_MBs = self.get_MBytess(download_value, download_unit)
            
            self.m.printCMesgValMesg(self.c.getBlue(),"upload    MBs               : ",
                                     self.c.getYellow(), upload_MBs, " MB/s")
            self.m.printCMesgValMesg(self.c.getBlue(),"Downsload MBs               : ",
                                     self.c.getYellow(), download_MBs, " MB/s")
        else:
            self.m.printMesg("Ethernet interface not found in the data.")
        # end of If block
        # Now writting the output file
        current_time = datetime.now()
        format_time = current_time.strftime('%Y-%m-%d_%H:%M:%S')
        msg = ""
        try:
            csv_file = open(log_file, 'a')
            msg = "\"" + str(format_time)                     + "\"" + "," + \
                  "\"" + str('{:03.4f}'.format(upload_MBs))   + "\"" + "," + \
                  "\"" + str('{:03.4f}'.format(download_MBs)) + "\"" + "\n"
            csv_file.write(msg)
            #FileList =    csv_file.read().splitlines()
        except IOError:
            rc = self.c.get_RC_FAIL()
            self.m.printCMesg("Cannot open or no such file : "+csv_file, self.c.get_B_Red())
            self.c.getFinalExit(self.c, self.m,rc)
            exit(rc)
        csv_file.close()
        return rc
#---------------------------------------------------------------------------
# end of Usage_Network
#---------------------------------------------------------------------------
