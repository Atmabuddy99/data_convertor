@echo off

cd /d D:/DATA_PUSH


start "seperatedb" cmd /k "python 1push_seperate_indexes_db.py"
