@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  AUREXIS AWS CONTROLLER
REM  One bat file for the entire EC2 training workflow.
REM  Drop this in the SAME folder as your aurexis .py files
REM  and your aurexis-key.pem file.
REM ============================================================

cd /d "%~dp0"

set IP_FILE=_ec2_ip.txt
set KEY_FILE=aurexis-key.pem

REM ===== Verify ssh is on PATH =====
where ssh >nul 2>nul
if errorlevel 1 (
    echo.
    echo [FATAL] 'ssh' is not on PATH.
    echo.
    echo On Windows 10/11, install OpenSSH Client:
    echo   Settings ^> Apps ^> Optional features ^> Add a feature
    echo   Search 'OpenSSH Client', install it, restart this bat.
    echo.
    pause
    exit /b 1
)

REM ===== Verify key file =====
if not exist "%KEY_FILE%" (
    echo.
    echo [FATAL] %KEY_FILE% not found in this folder:
    echo   %CD%
    echo.
    echo This is the SSH private key file you downloaded when you
    echo created the EC2 instance. Drop it here and try again.
    echo.
    pause
    exit /b 1
)

REM ===== Fix key file permissions (Windows is fussy) =====
icacls "%KEY_FILE%" /inheritance:r >nul 2>nul
icacls "%KEY_FILE%" /grant:r "%username%:R" >nul 2>nul

REM ===== Load or ask for EC2 IP =====
if exist "%IP_FILE%" (
    set /p EC2_IP=<"%IP_FILE%"
) else (
    echo.
    echo First time setup. Looking up your EC2 Public IPv4:
    echo   AWS Console -^> EC2 -^> Instances -^> click your instance
    echo   Find "Public IPv4 address" - looks like 54.242.123.45
    echo.
    set /p EC2_IP="Enter your EC2 Public IPv4: "
    echo !EC2_IP!>"%IP_FILE%"
)

:menu
cls
echo.
echo ================================================================
echo   AUREXIS AWS CONTROLLER
echo ================================================================
echo   Server: ubuntu@!EC2_IP!
echo   Key:    %KEY_FILE%
echo ================================================================
echo.
echo   --- SETUP ---
echo   [0] First time on a fresh server: bootstrap (install everything)
echo   [1] Test connection
echo   [2] Push aurexis .py files to server
echo.
echo   --- TRAIN ---
echo   [3] START kitchen sink HEAVY (background, survives disconnect)
echo   [4] Start phone dashboard (open http://!EC2_IP!:8000/ in browser)
echo   [5] Watch live log (Ctrl+C exits watcher; training keeps running)
echo.
echo   --- CHECK ---
echo   [6] Is kitchen sink still running?
echo   [7] Stop kitchen sink
echo   [8] Open full SSH shell
echo.
echo   --- FINISH ---
echo   [9] Pull trained archive back to laptop
echo.
echo   [I] Change server IP (use after stopping/restarting EC2)
echo   [Q] Quit
echo ================================================================
echo.
set /p CHOICE=Choose: 

if /i "!CHOICE!"=="0" goto bootstrap
if /i "!CHOICE!"=="1" goto test
if /i "!CHOICE!"=="2" goto push
if /i "!CHOICE!"=="3" goto start
if /i "!CHOICE!"=="4" goto dashboard
if /i "!CHOICE!"=="5" goto watch
if /i "!CHOICE!"=="6" goto status
if /i "!CHOICE!"=="7" goto kill
if /i "!CHOICE!"=="8" goto shell
if /i "!CHOICE!"=="9" goto pull
if /i "!CHOICE!"=="i" goto changeip
if /i "!CHOICE!"=="q" exit /b 0
echo Invalid choice.
pause
goto menu


:bootstrap
echo.
echo Setting up a fresh server. This installs Python, libraries, etc.
echo Only run this on a brand-new EC2 instance.
echo.
echo Step 1/3: System packages...
ssh -i "%KEY_FILE%" -o StrictHostKeyChecking=no ubuntu@!EC2_IP! "sudo apt-get update -y && sudo apt-get install -y python3-pip python3-venv git unzip"
echo.
echo Step 2/3: Python virtualenv + libraries (3-4 min)...
ssh -i "%KEY_FILE%" -o StrictHostKeyChecking=no ubuntu@!EC2_IP! "python3 -m venv ~/aurexis-env && source ~/aurexis-env/bin/activate && pip install --upgrade pip && pip install requests numpy scipy pillow scikit-image scikit-learn kymatio joblib datasets opencv-python-headless boto3"
echo.
echo Step 3/3: Create project folder...
ssh -i "%KEY_FILE%" -o StrictHostKeyChecking=no ubuntu@!EC2_IP! "mkdir -p ~/Aurexis-scry"
echo.
echo Bootstrap complete. Next step: [2] to push your .py files up.
pause
goto menu


:test
echo.
echo Testing connection to !EC2_IP!...
echo.
ssh -i "%KEY_FILE%" -o StrictHostKeyChecking=no -o ConnectTimeout=15 ubuntu@!EC2_IP! "echo OK - connected; echo; uname -a; echo; echo Files in Aurexis-scry:; ls ~/Aurexis-scry 2>/dev/null | head -20"
if errorlevel 1 (
    echo.
    echo [FAIL] Could not connect. Common causes:
    echo  - EC2 instance is STOPPED. Start it in AWS Console.
    echo  - Stopped/restarted instances get a NEW IP. Use [I] to update.
    echo  - Your laptop's IP isn't allowed in the security group.
)
echo.
pause
goto menu


:push
echo.
echo Uploading .py files to ~/Aurexis-scry on the server...
scp -i "%KEY_FILE%" -o StrictHostKeyChecking=no aurexis_v3.py sources.py cli.py aurexis_fast.py aurexis_train.py aurexis_hf.py aurexis_batch.py aurexis_camera.py aurexis_eval.py aurexis_aws.py kitchen_sink.py status_server.py ubuntu@!EC2_IP!:~/Aurexis-scry/ 2>nul
if errorlevel 1 (
    echo [WARN] Some files may have been missing locally. Continuing.
) else (
    echo Upload complete.
)
echo.
pause
goto menu


:start
echo.
echo Starting kitchen sink HEAVY on the server, in the background...
ssh -n -i "%KEY_FILE%" -o StrictHostKeyChecking=no ubuntu@!EC2_IP! "cd ~/Aurexis-scry && source ~/aurexis-env/bin/activate && nohup setsid python kitchen_sink.py heavy </dev/null >aurexis.log 2>&1 &"
timeout /t 4 /nobreak >nul
echo Verifying it started...
ssh -n -i "%KEY_FILE%" -o StrictHostKeyChecking=no ubuntu@!EC2_IP! "ps aux | grep kitchen_sink | grep -v grep | head -3 && echo --- log preview --- && tail -5 ~/Aurexis-scry/aurexis.log 2>/dev/null"
echo.
echo Kitchen sink is running.
echo It will keep running even if you close this window or your laptop.
echo Use [5] to watch the log, or [4] to start the phone dashboard.
echo.
pause
goto menu


:dashboard
echo.
echo Starting status dashboard on port 8000...
ssh -n -i "%KEY_FILE%" -o StrictHostKeyChecking=no ubuntu@!EC2_IP! "cd ~/Aurexis-scry && source ~/aurexis-env/bin/activate && nohup setsid python status_server.py </dev/null >status.log 2>&1 &"
timeout /t 3 /nobreak >nul
echo Verifying it started...
ssh -n -i "%KEY_FILE%" -o StrictHostKeyChecking=no ubuntu@!EC2_IP! "ps aux | grep status_server | grep -v grep | head -2"
echo.
echo Dashboard URL:
echo.
echo     http://!EC2_IP!:8000/
echo.
echo Open that URL in your phone or laptop browser.
echo.
echo IMPORTANT: If the page doesn't load, port 8000 isn't open yet.
echo Open AWS Console -^> EC2 -^> your instance -^> Security tab
echo -^> click the security group -^> Edit inbound rules
echo -^> Add rule: Custom TCP, port 8000, Source: Anywhere -^> Save.
echo.
pause
goto menu


:watch
echo.
echo Watching aurexis.log live. Press Ctrl+C to stop watching.
echo (Ctrl+C does NOT stop training - just stops the watch.)
echo.
ssh -i "%KEY_FILE%" -o StrictHostKeyChecking=no ubuntu@!EC2_IP! "tail -f ~/Aurexis-scry/aurexis.log"
echo.
goto menu


:status
echo.
echo === Server processes ===
ssh -i "%KEY_FILE%" -o StrictHostKeyChecking=no ubuntu@!EC2_IP! "ps aux | grep -E 'kitchen_sink|status_server' | grep -v grep || echo NO AUREXIS PROCESSES RUNNING"
echo.
echo === Last 8 lines of log ===
ssh -i "%KEY_FILE%" -o StrictHostKeyChecking=no ubuntu@!EC2_IP! "tail -8 ~/Aurexis-scry/aurexis.log 2>/dev/null || echo no log yet"
echo.
echo === Archive size ===
ssh -i "%KEY_FILE%" -o StrictHostKeyChecking=no ubuntu@!EC2_IP! "du -sh ~/Aurexis-scry/aurexis_archive 2>/dev/null || echo no archive yet"
echo.
pause
goto menu


:kill
echo.
ssh -i "%KEY_FILE%" -o StrictHostKeyChecking=no ubuntu@!EC2_IP! "pkill -f kitchen_sink.py && echo Stopped kitchen_sink || echo No kitchen_sink was running"
echo.
pause
goto menu


:shell
echo.
echo Opening interactive SSH shell. Type 'exit' or Ctrl+D to return.
echo.
ssh -i "%KEY_FILE%" -o StrictHostKeyChecking=no ubuntu@!EC2_IP!
echo.
goto menu


:pull
echo.
echo Packaging archive on server...
ssh -i "%KEY_FILE%" -o StrictHostKeyChecking=no ubuntu@!EC2_IP! "cd ~/Aurexis-scry && tar czf aurexis_archive.tar.gz aurexis_archive/ && ls -lh aurexis_archive.tar.gz"
if errorlevel 1 (
    echo [FAIL] Could not package archive.
    pause
    goto menu
)
echo.
echo Downloading to laptop...
scp -i "%KEY_FILE%" -o StrictHostKeyChecking=no ubuntu@!EC2_IP!:~/Aurexis-scry/aurexis_archive.tar.gz .
if exist aurexis_archive.tar.gz (
    echo Extracting...
    tar -xzf aurexis_archive.tar.gz
    echo.
    echo Done. Trained archive is now in: %CD%\aurexis_archive\
    echo Run aurexis.bat (the regular one) to use it locally.
    echo Hit [c] for live camera mode.
) else (
    echo [FAIL] Download failed.
)
echo.
pause
goto menu


:changeip
echo.
set /p EC2_IP="Enter new EC2 Public IPv4: "
echo !EC2_IP!>"%IP_FILE%"
echo Saved.
echo.
pause
goto menu
