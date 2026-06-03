@echo off
setlocal EnableExtensions
cd /d %~dp0\..

set "VCPKG_TOOLCHAIN="
if defined VCPKG_ROOT if exist "%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" (
    set "VCPKG_TOOLCHAIN=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"
)
if not defined VCPKG_TOOLCHAIN if exist "C:\vcpkg\scripts\buildsystems\vcpkg.cmake" (
    set "VCPKG_TOOLCHAIN=C:\vcpkg\scripts\buildsystems\vcpkg.cmake"
)
if not defined VCPKG_TOOLCHAIN if exist "%USERPROFILE%\vcpkg\scripts\buildsystems\vcpkg.cmake" (
    set "VCPKG_TOOLCHAIN=%USERPROFILE%\vcpkg\scripts\buildsystems\vcpkg.cmake"
)
set "VCPKG_TRIPLET=x64-windows"
if defined GSPLAT_VCPKG_TRIPLET set "VCPKG_TRIPLET=%GSPLAT_VCPKG_TRIPLET%"
set "GSPLAT_VCPKG_ARGS="
if defined VCPKG_TOOLCHAIN (
    set "GSPLAT_VCPKG_ARGS=-DCMAKE_TOOLCHAIN_FILE=%VCPKG_TOOLCHAIN% -DVCPKG_TARGET_TRIPLET=%VCPKG_TRIPLET%"
    echo INFO: using vcpkg toolchain: %VCPKG_TOOLCHAIN%
    echo INFO: vcpkg triplet: %VCPKG_TRIPLET%
) else (
    echo INFO: vcpkg toolchain not found, using system packages
)

set "GSPLAT_OSG_ARGS="
set "GSPLAT_OSG_VCPKG_INSTALLED=%~dp0..\..\3dgs-osg-viewer\vcpkg_installed"
if not exist "%GSPLAT_OSG_VCPKG_INSTALLED%\%VCPKG_TRIPLET%\share\unofficial-osg\unofficial-osg-config.cmake" (
    set "GSPLAT_OSG_VCPKG_INSTALLED=%~dp0..\3dgs-osg-viewer\vcpkg_installed"
)
if exist "%GSPLAT_OSG_VCPKG_INSTALLED%\%VCPKG_TRIPLET%\share\unofficial-osg\unofficial-osg-config.cmake" (
    set "GSPLAT_OSG_PREFIX=%GSPLAT_OSG_VCPKG_INSTALLED%\%VCPKG_TRIPLET%"
    set "GSPLAT_OSG_ARGS=-DVCPKG_INSTALLED_DIR=%GSPLAT_OSG_VCPKG_INSTALLED% -DCMAKE_PREFIX_PATH=%GSPLAT_OSG_PREFIX%"
    echo INFO: reusing 3dgs vcpkg_installed: %GSPLAT_OSG_PREFIX%
)

set "VSROOT=C:\Program Files\Microsoft Visual Studio\18"
set "VSCOMM="
for %%E in (BuildTools Community Professional Enterprise) do (
    if exist "%VSROOT%\%%E\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VSCOMM=%VSROOT%\%%E"
        goto :vs_ok
    )
)
echo ERROR: Visual Studio not found
exit /b 1

:vs_ok
call "%VSCOMM%\VC\Auxiliary\Build\vcvarsall.bat" amd64 -vcvars_ver=14.44
if errorlevel 1 exit /b 1
set "CUDA118=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
if exist "%CUDA118%\bin\nvcc.exe" (
  set "CUDA_PATH=%CUDA118%"
  if exist "C:\cuda118\bin\nvcc.exe" (
    set "PATH=C:\cuda118\bin;%CUDA118%\bin;%PATH%"
  ) else (
    set "PATH=%CUDA118%\bin;%PATH%"
  )
)

set "NINJA=%VSCOMM%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if not exist "%NINJA%" (
    for /f "delims=" %%N in ('where ninja 2^>nul') do set "NINJA=%%N" & goto :ninja_ok
    echo ERROR: ninja not found
    exit /b 1
)
:ninja_ok

if not exist build mkdir build
cd build
if "%GSPLAT_CLEAN_BUILD%"=="1" (
    if exist CMakeCache.txt del /f CMakeCache.txt
)

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DGSPLAT_USE_CUDA_TOOLCHAIN=ON ^
  -DGSPLAT_BUILD_RENDER_PLY=ON ^
  -DGSPLAT_BUILD_OSG_APP=ON ^
  %GSPLAT_VCPKG_ARGS% ^
  %GSPLAT_OSG_ARGS%
if errorlevel 1 exit /b 1

"%NINJA%" gsplat_render_ply
if errorlevel 1 exit /b 1

"%NINJA%" gsplat_osg_app 2>nul
if not errorlevel 1 (
    echo OK: %CD%\gsplat_osg_app.exe
)

echo.
echo OK: %CD%\gsplat_render_ply.exe
endlocal
