$WshShell = New-Object -ComObject WScript.Shell

# Full path to the project folder
$projectPath = (Get-Item -Path ".").FullName

# Create a shortcut for the batch script
$shortcutBat = $WshShell.CreateShortcut("$projectPath\Sweeta Environment (CMD).lnk")
$shortcutBat.TargetPath = "cmd.exe"
$shortcutBat.Arguments = "/k `"$projectPath\activate_env.bat`""
$shortcutBat.WorkingDirectory = $projectPath
$shortcutBat.IconLocation = "C:\Windows\System32\cmd.exe,0"
$shortcutBat.Description = "Opens the conda environment for Sweeta (CMD)"
$shortcutBat.Save()

# Create a shortcut for the PowerShell script
$shortcutPs = $WshShell.CreateShortcut("$projectPath\Sweeta Environment (PowerShell).lnk")
$shortcutPs.TargetPath = "powershell.exe"
$shortcutPs.Arguments = "-NoExit -ExecutionPolicy Bypass -File `"$projectPath\activate_env.ps1`""
$shortcutPs.WorkingDirectory = $projectPath
$shortcutPs.IconLocation = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe,0"
$shortcutPs.Description = "Opens the conda environment for Sweeta (PowerShell)"
$shortcutPs.Save()

# Create a direct shortcut for the GUI application
$shortcutGui = $WshShell.CreateShortcut("$projectPath\Sweeta (GUI).lnk")
$shortcutGui.TargetPath = "cmd.exe"
$shortcutGui.Arguments = "/k `"conda activate py312aiwatermark && python $projectPath\remwmgui.py`""
$shortcutGui.WorkingDirectory = $projectPath
$shortcutGui.Description = "Launches the Sweeta graphical interface directly"
$shortcutGui.Save()

Write-Host "Shortcuts created successfully in the folder $projectPath" -ForegroundColor Green
Write-Host "You can move them to your desktop or Start menu." -ForegroundColor Yellow 