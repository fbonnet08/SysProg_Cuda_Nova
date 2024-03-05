# Scripts to initiaite the starting procedure of the
# virtual machine environment using PowerShell
# first star the OMV6 on local network
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' startvm "OMV6_x64"
# 1. first we start the virtual machine parefeu without the --type headless switch
# IP: 192.168.1.97
# Unising the sleeping command to wait for the machine to boot properly before starting the others
# Start-Sleep [-Seconds*] <Int32> [<CommonParameters>]
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' startvm "Parefeu"
Start-Sleep -Seconds 15
# Starting the DMZ server
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' startvm "ServeurDMZ_Sem2"
# 2. We can now start the DNS server
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' startvm "ServerDNS"
Start-Sleep -Seconds 20
# STarting the DHCP server
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' startvm "ServerDHCP"
Start-Sleep -Seconds 15
# Now straing the NFS server
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' startvm "ServerNFS"
Start-Sleep -Seconds 15
# NOw starting the Proxy server
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' startvm "ProxeyWeb_Sem2"
Start-Sleep -Seconds 15
# Now starting the mail services
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' startvm "ServerSMTP"
Start-Sleep -Seconds 15
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' startvm "ServerSMTP_DMZ"
Start-Sleep -Seconds 15
# Finally start the OpenMedivaul in the demilitarized zone simultaneously as Alice
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' startvm "OMV6_x64_DMZ"
# at last we start the client Alice on the DHCP server
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' startvm "Alice_sem2"

