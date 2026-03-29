@echo off
echo =======================================================
echo FLUXION CMake C++ Verilator Extension Build Script
echo =======================================================

echo Checking for CMake...
cmake --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: CMake not found. Install from cmake.org
    pause
    exit /b 1
)

echo [1/2] Configuring CMake project...
cmake -B build -S .

if %errorlevel% neq 0 (
    echo [ERROR] CMake configuration failed.
    echo Please ensure you have C++ Build Tools installed.
    pause
    exit /b %errorlevel%
)

echo [2/2] Building Verilator Extension...
cmake --build build --config Release

if %errorlevel% neq 0 (
    echo [ERROR] Compilation failed.
    pause
    exit /b %errorlevel%
)

echo Build complete.
echo =======================================================
pause
