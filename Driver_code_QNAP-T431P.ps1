Import-Module BitsTransfer
function My-Copy-File {
    param( $from, $to)
    Write-Host $from
    Write-Host $to
    $fileSize = (Get-Item -Path $from).Length
    $ffile = [io.file]::OpenRead($from)
    #$tofile = [io.file]::OpenWrite($to)
    Write-Progress -Activity "Copying file" -status "$from -> $to" -PercentComplete 0
    $ncount = 3
    try {
        [byte[]]$buff = new-object byte[] 4096
        [long]$total = [int]$count = 0            #Start-Sleep -Seconds 1
        do {
            $count = $ffile.Read($buff, 0, $buff.Length)
            #$tofile.Write($buff, 0, $count)
            $total += $count
            if ($total % 1mb -eq 0) {
                Write-Progress -Activity "Copying file" -status "$from -> $to" `
                   -PercentComplete ([long]($total * 100 / $ffile.Length))
            }
        } while ($count -gt 0) }
    finally {
        #$ffile.Dispose()
        #$tofile.Dispose()
        Write-Progress -Activity "Copying file" -Status "Ready" -Completed
    }
}
################################################################################
# first unmount the drives in case they are already mounted
Write-Host "First start by umoiunting the drives to be sure"
.\Scripts\Windows\UmountShareDrives.ps1
# mounting the drives
Write-Host "Next we remount the drive and start the procedure"
.\Scripts\Windows\MountShareDrives.ps1
################################################################################
# Getting the data from the copymounting the drives
Write-Host "Launching NetworkDriverMain.py ..."
################################################################################
$f_copy_from = "C:\Users\Frederic\Desktop\cuda_12.3.2_546.12_windows.exe"
################################ QNAP-T431P ####################################
$f_copy_to   = "E:\Frederic"
$csv_file    = "cp_file_NetUse_QNAP-T431P.csv"

start powershell {python .\NetworkDriverMain.py --with_while_loop=yes --csvfile=cp_file_NetUse_QNAP-T431P.csv}

Write-Host "Transfering data to storage QNAP-T431P ..."
Start-Sleep -Seconds 10
Start-BitsTransfer -Source $f_copy_from -Destination $f_copy_to -Description "Copy" -DisplayName "$f_copy_from -> $f_copy_to ---> QNAP-T431P"

# once the file copy is done we stop the
Start-Sleep -Seconds 20
Get-Process *python|Stop-Process

# Next we graph the result using the graph command
Write-Host "Graphing the data for the transfer ..."
start powershell {python .\NetworkDriverMain.py --csvfile=cp_file_NetUse_QNAP-T431P.csv}
