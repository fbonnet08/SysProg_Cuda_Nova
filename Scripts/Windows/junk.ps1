
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
            #python .\NetworkDriverMain.py --with_while_loop=no --plot_asc=copy_file_Synology.asc
        } while ($count -gt 0) }
    finally {
        #$ffile.Dispose()
        #$tofile.Dispose()
        Write-Progress -Activity "Copying file" -Status "Ready" -Completed
    }
}
My-Copy-File "C:\Users\Frederic\Desktop\cuda_12.3.2_546.12_windows.exe" "C:\Users\Frederic\OneDrive\UVPD-Perpignan\SourceCodes\CLionProjects\SysProg-Cuda-Nova"



$ScriptBlock = {
    #python .\NetworkDriverMain.py --with_while_loop=yes --plot_asc=$asc_file
    ls > test.txt
}
# Start the background job
Start-Job -ScriptBlock $ScriptBlock




$from = "C:\Users\Frederic\Desktop\cuda_12.3.2_546.12_windows.exe"
$to = "C:\Users\Frederic\OneDrive\UVPD-Perpignan\SourceCodes\CLionProjects\SysProg-Cuda-Nova"

Write-Progress -Activity "Copying file" -status "$from -> $to" -PercentComplete 0

[long]$total = [int]$count = 0
do {
    Start-Sleep -Seconds 1
    $count ++
    #$total += $count
    Write-Host "$count"
    #if ($total % 1mb -eq 0) {
    Write-Progress -Activity "Copying file" -status "$from -> $to" `
                       -PercentComplete ([long]($count * 100 / 10 ))
    #}
    #python .\NetworkDriverMain.py --with_while_loop=no --plot_asc=copy_file_Synology.asc
    #$total += $count
}while  ($count -ne 10) #(Copy-Item "$from" -Destination "$to")

$cnt=0
while ($cnt -ne 20 ){
    Start-Sleep -Seconds 1
    $cnt++
    python .\NetworkDriverMain.py --with_while_loop=no --plot_asc=copy_file_Synology.asc
}


do {
    $count ++
    #$total += $count
    #Write-Host "$count"
    #if ($total % 1mb -eq 0) {
    Write-Progress -Activity "Copying file" -status "$from -> $to" `
                       -PercentComplete ([long]($count * 100 / $ncount ))
    #}
    python .\NetworkDriverMain.py --with_while_loop=no --plot_asc=copy_file_Synology.asc
    #$total += $count
} while (Copy-Item "$from" -Destination "$to") }


$from = "C:\Users\Frederic\Desktop\cuda_12.3.2_546.12_windows.exe"
$to = "C:\Users\Frederic\OneDrive\UVPD-Perpignan\SourceCodes\CLionProjects\SysProg-Cuda-Nova"

Write-Progress -Activity "Copying file" -status "$from -> $to" -PercentComplete 0


Copy-Item "$from" -Destination "$to"

function Copy-File {
    param( [string]$from, [string]$to)
    #$ffile = [io.file]::OpenRead($from)
    #$tofile = [io.file]::OpenWrite($to)
    Write-Progress -Activity "Copying file" -status "$from -> $to" -PercentComplete 0

    Copy-Item "$from" -Destination "$to"
    #try {
    #    [byte[]]$buff = new-object byte[] 4096
    #    [long]$total = [int]$count = 0
    #    do {
    #        $count = $ffile.Read($buff, 0, $buff.Length)
    #        $tofile.Write($buff, 0, $count)
    #        $total += $count
    #        if ($total % 1mb -eq 0) {
    #            Write-Progress -Activity "Copying file" -status "$from -> $to" `
    #               -PercentComplete ([long]($total * 100 / $ffile.Length))
    #        }
    #    } while ($count -gt 0)
    #}
    #finally {
    #    $ffile.Dispose()
    #    $tofile.Dispose()
    #    Write-Progress -Activity "Copying file" -Status "Ready" -Completed
    #}
}

$conns = net use
foreach ($con in $conns) {
    if($con -match '\\\\MyServerB\\C\$') {
        net use '\\MyServerB\C$' /delete
    }
}

\\192.168.1.155\homes

