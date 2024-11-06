for /f "delims=" %%i in ('"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -products * -latest -property installationPath') do set VS_INSTALL_PATH=%%i

set VCVARSALL="%VS_INSTALL_PATH%\VC\Auxiliary\Build\vcvarsall.bat"
REM echo VCVARSALL="%VS_INSTALL_PATH%\VC\Auxiliary\Build\vcvarsall.bat">> %GITHUB_ENV%

%VCVARSALL% x64