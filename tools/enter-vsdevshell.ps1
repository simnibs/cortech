$vs = & "${Env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -products * -latest -format json | ConvertFrom-Json
Import-Module "$($vs.InstallationPath)\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell $vs.InstanceId -DevCmdArguments '-arch=x64'