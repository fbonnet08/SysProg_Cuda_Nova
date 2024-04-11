# mounting the drives
Write-Host ""
# Synology
$remoteComputer = "192.168.1.155"
IF (Test-Connection -BufferSize 32 -Count 1 -ComputerName $remoteComputer -Quiet) {
    Write-Host "The remote Synology-DS1821+ NAS is Online"
    # Synology
    net use A: \\$remoteComputer\homes lupf_Synology24! /user:$remoteComputer\frederic
    net use B: \\$remoteComputer\homes n%MN^H5Z         /user:$remoteComputer\alice
    net use D: \\$remoteComputer\homes v251LlHY         /user:$remoteComputer\bob
} Else { Write-Host "The remote Synology-DS1821+ is Down  ---> No share folders to be mounted" }
# QNAP
$remoteComputer = "192.168.1.59"
IF (Test-Connection -BufferSize 32 -Count 1 -ComputerName $remoteComputer -Quiet) {
    Write-Host "The remote QNAP-T431P NAS is Online"
    # QNAP
    net use E: \\$remoteComputer\Frederic lupf_Qnap24!   /user:$remoteComputer\frederic
    net use F: \\$remoteComputer\Alice    #  lupf_Alice24!   /user:$remoteComputer\alice
    net use H: \\$remoteComputer\Bob      #  lupf_Bob24!   /user:$remoteComputer\bob
} Else { Write-Host "The remote QNAP-T431P is Down        ---> No share folders to be mounted" }
#OpenMediaVault
$remoteComputer = "192.168.1.68"
IF (Test-Connection -BufferSize 32 -Count 1 -ComputerName $remoteComputer -Quiet) {
    Write-Host "The remote OpenMediaVault NAS is Online"
    #OpenMediaVault
    net use I: \\$remoteComputer\Frederic lupf_Openmediavault23   /user:$remoteComputer\frederic
    net use J: \\$remoteComputer\Alice    # lupf_Alice24!   /user:$remoteComputer\alice
    net use K: \\$remoteComputer\Bob      # lupf_Alice24!   /user:$remoteComputer\alice
    net use M: \\$remoteComputer\Jean     # lupf_Alice24!   /user:$remoteComputer\alice
    net use N: \\$remoteComputer\PoolM1   # lupf_Alice24!   /user:$remoteComputer\alice
} Else { Write-Host "The remote OpenMediaVault NAS is Down ---> No share folders to be mounted"
Write-Host "" }





