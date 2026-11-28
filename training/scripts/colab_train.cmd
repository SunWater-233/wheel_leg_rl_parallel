@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0colab_train.ps1" %*
exit /b %ERRORLEVEL%
