@echo off

cd /d D:/DATA_PUSH


start "maindb" cmd /k "python 1DB_main_pusher.py"
