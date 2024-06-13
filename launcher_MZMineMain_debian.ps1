# start powershell {python .\NetworkDriverMain.py --with_while_loop=yes --csvfile=NetUse_OBS.csv}

# ########## Windows ####################
#python3 MZMineMain.py --import_db='E:\Bacterial metabolites database.txt';
#python3 MZMineMain.py --import_db='E:\Emerging pollutants database.txt';
#python3 MZMineMain.py --import_db='E:\Metabolite database.txt';

#python3 MZMineMain.py --import_external_db='E:\data\ToBeInserted\NEG_LC.json' --import_db_set=single --database_origin=LC-MS
#python3 MZMineMain.py --import_external_db='E:\data\ToBeInserted\POS_LC.json' --import_db_set=single --database_origin=LC-MS

# ########## Linux ####################

python3 MZMineMain.py --import_db='/data/frederic/ToBeInserted/Bacterial metabolites database.txt';
python3 MZMineMain.py --import_db='/data/frederic/ToBeInserted/Emerging pollutants database.txt';
python3 MZMineMain.py --import_db='/data/frederic/ToBeInserted/Metabolite database.txt';

# POS
python3 MZMineMain.py --import_external_db='/data/frederic/ToBeInserted/POS_LC.json' --import_db_set=single --database_origin=LC-MS

# NEG
python3 MZMineMain.py --import_external_db='/data/frederic/ToBeInserted/NEG_LC.json' --import_db_set=single --database_origin=LC-MS


#python MZMineMain.py --import_external_db='E:\data\ToBeInserted\GNPS-LIBRARY.mgf' --import_db_set=single --database_origin=GNPS
#python MZMineMain.py --import_external_db='E:\data\ToBeInserted\MoNA-export-LC-MS-MS_Negative_Mode.json' --import_db_set=single --database_origin=MoNA

############## Frag Hub ##############
#python MZMineMain.py --import_external_db='E:\data\Nuage\FragHub\Input\MGF\DEREPLICATOR_IDENTIFIED_LIBRARY.mgf' --import_db_set=singleb --database_origin=FragHub

# start powershell { python MZMineMain.py --machine_learning=yes }

