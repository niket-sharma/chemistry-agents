@echo off
REM Build script for chemistry-agents package on Windows

echo Chemistry Agents Package Builder
echo ==================================

if "%1"=="clean" (
    echo Cleaning build artifacts...
    if exist build rmdir /s /q build
    if exist dist rmdir /s /q dist
    if exist chemistry_agents.egg-info rmdir /s /q chemistry_agents.egg-info
    if exist src\chemistry_agents.egg-info rmdir /s /q src\chemistry_agents.egg-info
    echo ✅ Cleanup completed
    goto end
)

if "%1"=="build" (
    echo Building package...
    python build_package.py build
    goto end
)

if "%1"=="test" (
    echo Running tests...
    python build_package.py test
    goto end
)

if "%1"=="install" (
    echo Installing locally...
    python build_package.py install
    goto end
)

if "%1"=="all" (
    echo Running full build pipeline...
    python build_package.py all
    goto end
)

echo Usage: build_package.bat [clean^|build^|test^|install^|all]
echo.
echo Commands:
echo   clean     - Clean build artifacts
echo   build     - Build the package
echo   test      - Run tests
echo   install   - Install package locally
echo   all       - Run complete build pipeline

:end