@echo off

setlocal EnableExtensions

rem One-shot configure + build for gsplat_osg_app (cmd.exe only, not PowerShell).



set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"

set "NINJA=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"

set "CMAKE=C:\Program Files\CMake\bin\cmake.exe"

set "SRC=C:\code\gsplat-raster-app"

set "BLD=C:\code\gsplat-raster-app\build"

set "VCPKG=C:\code\3dgs-osg-viewer\vcpkg_installed\x64-windows"



set "CUDA118=C:\cuda118\bin\nvcc.exe"

set "CUDA13=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"

set "USE_CUDA118=OFF"

set "CMAKE_EXTRA="

if exist "%CUDA118%" (

    set "USE_CUDA118=ON"

    set "CUDA_PATH=C:\cuda118"

    set "PATH=C:\cuda118\bin;%PATH%"

) else if exist "%CUDA13%" (

    set "CMAKE_EXTRA=-DGSPLAT_USE_CUDA_TOOLCHAIN=OFF -DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe"

    set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%"

) else (

    echo No CUDA toolkit: need C:\cuda118 or CUDA v13.3

    exit /b 1

)



if "%USE_CUDA118%"=="ON" (
    call "%VCVARS%" -vcvars_ver=14.44
) else (
    call "%VCVARS%"
)
if errorlevel 1 exit /b 1



set "HOST_CL="
rem CUDA 11.8 + VS2026: use older MSVC 14.4x toolset as nvcc host (14.51 STL breaks cicc).
set "_VS18_MSVC=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC"
if exist "%_VS18_MSVC%" (
    for /f "delims=" %%I in ('dir /b /ad /o-n "%_VS18_MSVC%\14.4*" 2^>nul') do (
        if exist "%_VS18_MSVC%\%%I\bin\Hostx64\x64\cl.exe" set "HOST_CL=%_VS18_MSVC%\%%I\bin\Hostx64\x64\cl.exe" & goto :got_cl
    )
)
for %%V in (2022 2019) do (
    if not defined HOST_CL (
        for /f "delims=" %%I in ('dir /s /b "C:\Program Files\Microsoft Visual Studio\%%V\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe" 2^>nul') do set "HOST_CL=%%I" & goto :got_cl
    )
)
if not defined HOST_CL for /f "delims=" %%I in ('where cl 2^>nul') do set "HOST_CL=%%I" & goto :got_cl
:got_cl
if not defined HOST_CL (
    echo cl.exe not found. Install VS2022 Build Tools ^(C++^) for CUDA 11.8 host, or run vcvars64.
    exit /b 1
)
for %%I in ("%HOST_CL%") do set "HOST_CL=%%~sI"
echo Using CUDA host compiler: %HOST_CL%



set "CL="

set "CXX="

set "CC="

set "CUDAHOSTCXX=%HOST_CL%"



if exist "%BLD%" rmdir /S /Q "%BLD%"



echo === Configure (USE_CUDA118=%USE_CUDA118%, host=%HOST_CL%) ===

if "%USE_CUDA118%"=="ON" (
    "%CMAKE%" -S "%SRC%" -B "%BLD%" -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl -DCMAKE_PREFIX_PATH="%VCPKG%" -DVCPKG_INSTALLED_DIR=C:/code/3dgs-osg-viewer/vcpkg_installed -DVCPKG_TARGET_TRIPLET=x64-windows -DGSPLAT_USE_CUDA_TOOLCHAIN=ON "-DCMAKE_CUDA_HOST_COMPILER=%HOST_CL%"
) else (
    "%CMAKE%" -S "%SRC%" -B "%BLD%" -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl -DCMAKE_PREFIX_PATH="%VCPKG%" -DVCPKG_INSTALLED_DIR=C:/code/3dgs-osg-viewer/vcpkg_installed -DVCPKG_TARGET_TRIPLET=x64-windows %CMAKE_EXTRA% "-DCMAKE_CUDA_HOST_COMPILER=%HOST_CL%"
)



if errorlevel 1 (

    echo === configure failed ===

    exit /b 1

)



echo === Build gsplat_osg_app ===

"%CMAKE%" --build "%BLD%" --target gsplat_osg_app

set "RC=%ERRORLEVEL%"

if "%RC%"=="0" (

    echo === Deploy runtime DLLs ===

    xcopy /Y /Q "%VCPKG%\bin\*.dll" "%BLD%\" >nul

    if exist "C:\cuda118\bin\cudart64_110.dll" xcopy /Y /Q "C:\cuda118\bin\cudart64_110.dll" "%BLD%\" >nul

    echo.

    echo === OK: %BLD%\gsplat_osg_app.exe ===

    echo Run: %BLD%\gsplat_osg_app.exe your.ply 2000000

) else (

    echo === build failed ===

)

exit /b %RC%


