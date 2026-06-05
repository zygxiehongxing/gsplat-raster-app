@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=C:\cuda118\bin;%PATH%
set TMPDIR=%TEMP%\nvcc_probe
if exist "%TMPDIR%" rmdir /S /Q "%TMPDIR%"
mkdir "%TMPDIR%"
cd /d "%TMPDIR%"

echo int main(){return 0;} > test.cu

nvcc -allow-unsupported-compiler -Xcompiler=/allow-unsupported-compiler -Xcompiler=/D__NV_NO_HOST_COMPILER_CHECK -Xcompiler=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -c test.cu -o test.obj 2>&1

echo.
echo === dryrun ===
nvcc -allow-unsupported-compiler -Xcompiler=/allow-unsupported-compiler --dryrun -c test.cu -o test.obj 2>&1
