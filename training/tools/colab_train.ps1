# Compatibility wrapper forwarding to training/scripts/colab_train.ps1
& (Join-Path (Split-Path $PSScriptRoot -Parent) 'scripts\colab_train.ps1') @args
exit $LASTEXITCODE
