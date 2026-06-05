@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "NINJA=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "CMAKE=C:\Program Files\CMake\bin\cmake.exe"
set "PATH=C:\cuda118\bin;%PATH%"
set "BLD=C:\code\gsplat-raster-app\build"
if exist "%BLD%" rmdir /S /Q "%BLD%"
"%CMAKE%" -S C:\code\gsplat-raster-app -B "%BLD%" -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CXX_COMPILER=cl ^
  -DCMAKE_CUDA_COMPILER=C:/cuda118/bin/nvcc.exe ^
  -DCMAKE_PREFIX_PATH=C:/code/3dgs-osg-viewer/vcpkg_installed/x64-windows ^
  -DVCPKG_INSTALLED_DIR=C:/code/3dgs-osg-viewer/vcpkg_installed ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DGSPLAT_USE_CUDA_TOOLCHAIN=OFF
if errorlevel 1 exit /b 1
"%CMAKE%" --build "%BLD%" --target gsplat_osg_app
exit /b %ERRORLEVEL%
