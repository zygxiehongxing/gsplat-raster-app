@echo off

setlocal EnableExtensions



set "BLD=C:\code\gsplat-raster-app\build"

set "VCPKG_BIN=C:\code\3dgs-osg-viewer\vcpkg_installed\x64-windows\bin"

set "PATH=%BLD%;%VCPKG_BIN%;C:\cuda118\bin;%PATH%"
if not defined GSPLAT_CUDA_RASTER_TIMING set "GSPLAT_CUDA_RASTER_TIMING=1"




if not exist "%BLD%\gsplat_osg_app.exe" (

    echo Missing %BLD%\gsplat_osg_app.exe — run scripts\configure_and_build.bat first.

    exit /b 1

)



"%BLD%\gsplat_osg_app.exe" %*

