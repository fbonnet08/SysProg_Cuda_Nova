select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".analysing;
select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".analytics_data;
select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".building;
select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".buildingblocks;
select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".composing;
select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".compound;



select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".data;
select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".databasedetails;
select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".datetable;
select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".experiment;
select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".experimenting;

select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".fragment;
select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".identifying;
select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".ionisation_mode;
select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".ionising;
select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".logintable;

select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".spectral_data;
select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".tool;
select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".tooling;
select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".users;

insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".experiment (experiment_id, scan_id, ionisation_mode) VALUES (1, 1, 1);

insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".experimenting(experiment_id, scan_id, analytic_data_id, date_id) VALUES (1, 1, 1, 1);

insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".logintable(login_id, user_id, firstname, lastname, username, password, role, affiliation, department, researchlab) VALUES (1, 1, 'Frederic', 'Bonnet', 'fbonnet', 'fred1234', 'admin', 'Sorbonne-Banyuls', 'OBS', 'OBS');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".logintable(login_id, user_id, firstname, lastname, username, password, role, affiliation, department, researchlab) VALUES (2, 2, 'Nicolas', 'Desreumaux', 'ndesreumaux', 'nico1234', 'admin', 'Sorbonne-Banyuls', 'OBS', 'OBS');

insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".analysing(tool_id, analytic_data_id, date_id) VALUES (1, 1, 1);
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".analytics_data(analytic_data_id, date_id, sample_name, sample_details, sample_solvent, number_scans, filename) VALUES (2, 2, 'sample name 2', 'sample name 2 details', 'sample 2 solvent', '2885', 'oc1003.mgf');

UPDATE "MolRefAnt_DB".analytics_data
SET sample_details = 'sample name 1 details',
    sample_solvent = 'sample 1 solvent'
WHERE analytic_data_id = 1
  AND date_id = 1;


insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".users(user_id, affiliation, phone, email) VALUES (1, 'Banyuls OBS', '0605040302', 'user@gmail.com');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".tool (tool_id, instrument_source) VALUES (1, 'banyuls');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".tooling(tool_id, user_id) VALUES (1, 1);
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".databasedetails(database_id, database_name, database_affiliation, database_path, library_quality_legend) VALUES (1, 'metabolites', 'banyuls OBS', 'c:\\nuage', 'some library quality_legend 1');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".data(data_id, path_to_data, raw_file, csv_file, xls_file, asc_file, mgf_file, m2s_file, json_file, experiment_id, scan_id) VALUES (1, 'c:\\path_to_data', 'oc1003.raw', 'oc1003.csv', 'oc1003.xls', 'oc1003.asc', 'oc1003.mgf', 'oc1003.m2s', 'oc1003.json', 1, 1 );
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".spectral_data (spectral_id, feature_id, tool_id, pi_name_id, data_collector_id, pepmass, ion_mode, charge, mslevel, scan_number, retention_time, mol_json_file, tool_id_1, user_id, database_id, data_id) VALUES (1, 1001, 1, 1, 1, 739.52 , 'ion mode 1', 1.0, 1, 2885, '13:30:00','', 1, 1, 1, 1);
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".fragment (fragment_id, mz_value, relative, spectral_id, feature_id, tool_id, pi_name_id, data_collector_id) VALUES (1, 130.45, 98.10, 1, 1001, 1, 1, 1);

