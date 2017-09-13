@echo off

set arg1=%1
set arg2=%2

for %%a in (*.*) do call :Sub %%a
goto :eof

:Sub
set Name=%1
for /F "tokens=1 delims=_" %%b in ('echo %Name%') do set Folder=%arg2%%%b
if not exist %Folder% md %Folder%
move %1 %Folder%