# Script to stop the VM in the reverse order to the starting order

# First we can stop the OMV6 on the local network
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' controlvm "OMV6_x64" poweroff
# Finally start the OpenMedivaul in the demilitarized zone simultaneously as Alice
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' controlvm "OMV6_x64_DMZ" poweroff
# at last we start the client Alice on the DHCP server
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' controlvm "Alice_sem2" poweroff
# Now starting the mail services
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' controlvm "ServerSMTP" poweroff
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' controlvm "ServerSMTP_DMZ" poweroff
# NOw starting the Proxy server
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' controlvm "ProxeyWeb_Sem2" poweroff
# Now straing the NFS server
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' controlvm "ServerNFS" poweroff
# STarting the DHCP server
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' controlvm "ServerDHCP" poweroff
# 2. We can now start the DNS server
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' controlvm "ServerDNS" poweroff
# Starting the DMZ server
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' controlvm "ServeurDMZ_Sem2" poweroff
# 1. first we start the virtual machine parefeu without the --type headless switch
& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' controlvm "Parefeu" poweroff
#& 'C:\Program Files\Oracle\VirtualBox\VBoxManage.exe' controlvm "Parefeu" poweroff # --type headless




