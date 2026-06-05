@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=C:\cuda118\bin;%PATH%
nvcc --version
nvcc -allow-unsupported-compiler -Xcompiler=/allow-unsupported-compiler -Xcompiler=/D__NV_NO_HOST_COMPILER_CHECK -Xcompiler=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -c -o nul nul 2>&1
