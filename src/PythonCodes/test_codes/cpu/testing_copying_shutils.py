import shutil
import psutil
import time
import os
import sys
from pathlib import Path

def copy_with_network_monitoring(src_file, dest_file):
    # Get the initial network usage
    initial_network_usage = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
    
    # Copy the file
    shutil.copy(src_file, dest_file)
    
    # Get the final network usage
    final_network_usage = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
    
    # Calculate the network usage during the file copy
    network_usage = final_network_usage - initial_network_usage
    
    print(f"Total network usage during file copy: {network_usage} bytes")

def monitor_network_usage(src_file, dest_file, interval=1):
    # Get the initial network usage
    initial_network_usage = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
    
    # Copy the file while monitoring network usage per second
    shutil.copy(src_file, dest_file)
    
    # Monitor network usage per second
    while True:
        time.sleep(interval)
        current_network_usage = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
        network_usage = current_network_usage - initial_network_usage
        print(f"Network usage in the last {interval} seconds: {network_usage} bytes")
def copy_with_progress(src_file, dst_file, buffer_size=1024):
    total_size = os.path.getsize(src_file)
    bytes_sent = 0
    print("total_size : ", total_size)
    dst = open(dst_file, 'w')
    with open(src_file, 'r') as src: #, open(dst_file, 'w') as dst:
        while True:
            buffer = src.read(buffer_size)
            if not buffer:
                break
            dst.write(buffer)
            bytes_sent += len(buffer)
            print(f"Progress: {bytes_sent}/{total_size} bytes ({(bytes_sent/total_size)*100:.2f}%)", end='\r')
    
    print("\nFile copy completed.")

if __name__ == "__main__":
    #source_file = "/path/to/source/file"
    #destination_file = "/path/to/destination/file"
    current_dir = Path(__file__).parent.absolute()
    source_file = os.path.join(os.getcwd(),'C:\\','Users','Frederic','Desktop','cuda_12.3.2_546.12_windows.exe')
    dir_to = os.path.join(os.getcwd(), 'A:\\','Frederic')

    print ("current directory is: ", current_dir)
    print ("File to copy        : ", source_file)
    print ("copy file to        : ", dir_to)



# Option 1: Monitor network usage per second during the copy process
    #monitor_network_usage(source_file, dir_to)
    #copy_with_network_monitoring(source_file, dir_to)
    copy_with_progress(source_file, dir_to)
    # Option 2: Only print total network usage after the copy process completes
    # copy_with_network_monitoring(source_file, destination_file)
