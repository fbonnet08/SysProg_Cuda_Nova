
import psutil
import time
import os
import pandas as pd

UPDATE_DELAY = 1 # in seconds

def get_size(bytes):
    """
    Returns size of bytes in a nice format
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}B"
        bytes /= 1024

def extract_speed(speed_string):
    # Split the string based on the last occurrence of "/"
    parts = speed_string.split("/")[-1].strip()
    print("parts before: ", parts)
    if parts == "s": parts = speed_string.split("B/s")[0].strip()
    numerical_value = 0
    unit = "B/s"
    print("parts after: ", parts)
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

def get_MBytess(speed, units):
    speed_MBs = 0
    if units == "B/s":
        speed_MBs = speed / 1000000
    if units == "KB/s":
        speed_MBs = speed / 1000
    if units == "MB/s":
        speed_MBs = speed

    return speed_MBs

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
            "Download": get_size(io_2[iface].bytes_recv),
            "Upload": get_size(io_2[iface].bytes_sent),
            "Upload Speed": f"{get_size(upload_speed / UPDATE_DELAY)}/s",
            "Download Speed": f"{get_size(download_speed / UPDATE_DELAY)}/s",
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
    print(df.to_string())
    if (cnt == 5): break


ethernet_data = next((item for item in data if item["iface"] == "Ethernet"), None)

print(ethernet_data)

if ethernet_data:
    upload_speed = ethernet_data["Upload Speed"]
    download_speed = ethernet_data["Download Speed"]
    print("Ethernet Upload Speed:", upload_speed)
    print("Ethernet Download Speed:", download_speed)

    # Extracting only the numerical part
    upload_value, upload_unit = extract_speed(upload_speed)
    print("upload    Numerical value:", upload_value, "Unit:", upload_unit)
    download_value, download_unit = extract_speed(download_speed)
    print("Downsload Numerical value:", download_value, "Unit:", download_unit)

    upload_MBs = get_MBytess(upload_value, upload_unit)
    download_MBs = get_MBytess(download_value, download_unit)

    print("upload    MBs            :", upload_MBs, "MB/s")
    print("Downsload MBs            :", download_MBs, "MB/s")

else:
    print("Ethernet interface not found in the data.")


