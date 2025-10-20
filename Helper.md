## 🧾 PowerShell -  Fie Kernel

Lass dir alle laufenden Python-Kerne samt PID anzeigen:

Get-Process python | Select-Object Id, ProcessName, Path, StartTime


Wenn du siehst, dass mehrere python.exe laufen, kannst du prüfen, welcher davon dein Kernel ist.
Ein Jupyter-Kernel wird immer mit -m ipykernel_launcher gestartet, das findest du so:

Get-WmiObject Win32_Process -Filter "Name = 'python.exe'" |
  Where-Object { $_.CommandLine -match "ipykernel_launcher" } |
  Select-Object ProcessId, CommandLine