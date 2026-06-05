@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
echo CL=[%CL%]
echo CXX=[%CXX%]
echo CC=[%CC%]
where cl
