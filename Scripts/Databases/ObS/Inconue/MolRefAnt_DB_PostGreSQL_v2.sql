/* ----------------------------------
-- TP-1   MolRefAnt_DB_PostGreSQL Database
----------------------------------
-- Section 1
----------------------------------
-- First cleaning up the previous implementations for fresh start
*/
/* Creating the Schema to begin with */
--drop schema if exists "MolRefAnt_DB" cascade ;
--CREATE SCHEMA "MolRefAnt_DB" AUTHORIZATION frederic;
--COMMENT ON SCHEMA "MolRefAnt_DB" IS 'Creating the schema for the MolRefAnt_DB_PostGreSQL';

/* Dropping all tables */

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".employee_audits;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".employees;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".datetable;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".fragment;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".composing;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".identifying;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".compound;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".experimenting;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".building;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".buildingblocks;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".spectral_data;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".data;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".database;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".level;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".tooling;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".users;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".ionising;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".ionisation_mode;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".experiment;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".analysing;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".tool;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".analytics_data;

/* Creating the test tables */
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".employees cascade ;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".employees (
        employee_id integer primary key,
        name varchar(50) not null
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".employee_audits cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".employee_audits (
        employee_id integer primary key references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".employees(employee_id),
        old_name varchar(50) not null
);

/* Creating the test tables tables */

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".BuildingBlocks cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".BuildingBlocks(
                               bloc_id INT,
                               bloc_name VARCHAR(50),
                               bloc_structure VARCHAR(60),
                               PRIMARY KEY (bloc_id)
);


drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Experiment cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Experiment(
                           experiment_id INT,
                           scan_id VARCHAR(50),
                           ionisation_mode VARCHAR(50),
                           PRIMARY KEY(experiment_id, scan_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Tool cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Tool(
                     tool_id INT,
                     instrument_source VARCHAR(50),
                     ion_mode VARCHAR(50),
                     PRIMARY KEY(tool_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Data cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Data(
                     data_id INT,
                     path_to_data VARCHAR(150),
                     raw_file VARCHAR(60),
                     csv_file VARCHAR(60),
                     xls_file VARCHAR(60),
                     asc_file VARCHAR(60),
                     mgf_file VARCHAR(50),
                     m2s_file VARCHAR(50),
                     experiment_id INT NOT NULL,
                     scan_id VARCHAR(50) NOT NULL,
                     PRIMARY KEY(data_id),
                     FOREIGN KEY(experiment_id, scan_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Experiment(experiment_id, scan_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Users cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Users(
                      user_id INT,
                      PI_name VARCHAR(50),
                      data_colector VARCHAR(50),
                      affiliation VARCHAR(50),
                      phone VARCHAR(50),
                      email VARCHAR(50),
                      PRIMARY KEY(user_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Ionisation_mode cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Ionisation_mode(
                                inonisation_mode_id INT,
                                ionisation_mode VARCHAR(50),
                                experiment_id INT NOT NULL,
                                scan_id VARCHAR(50) NOT NULL,
                                PRIMARY KEY(inonisation_mode_id),
                                UNIQUE(experiment_id, scan_id),
                                FOREIGN KEY(experiment_id, scan_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Experiment(experiment_id, scan_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Analytics_data cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Analytics_data(
                               analytic_data_id INT,
                               date_id INT,
                               sample_name VARCHAR(50),
                               sample_details VARCHAR(100),
                               sample_solvent VARCHAR(50),
                               number_scans VARCHAR(50),
                               filename VARCHAR(50),
                               PRIMARY KEY(analytic_data_id, date_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Database cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Database(
                         database_id INT,
                         database_name VARCHAR(50),
                         database_affiliation VARCHAR(50),
                         database_path VARCHAR(100),
                         PRIMARY KEY(database_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".DateTable cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".DateTable(
                          date_id INT,
                          date_column DATE,
                          time_column TIME,
                          timestamp_with_tz_column VARCHAR(50),
                          analytic_data_id INT NOT NULL,
                          date_id_1 INT NOT NULL,
                          PRIMARY KEY(date_id),
                          UNIQUE(analytic_data_id, date_id_1),
                          FOREIGN KEY(analytic_data_id, date_id_1) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Analytics_data(analytic_data_id, date_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Level cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Level(
                      level_id INT,
                      level_number INT,
                      level_name VARCHAR(50),
                      level_description VARCHAR(200),
                      PRIMARY KEY(level_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Spectral_data cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Spectral_data(
                              spectral_id INT,
                              feature_id INT,
                              pepmass DECIMAL(15,2),
                              charge DECIMAL(15,2),
                              level_id INT NOT NULL,
                              scan_number VARCHAR(50),
                              retention_time TIME,
                              level_id_1 INT NOT NULL,
                              database_id INT NOT NULL,
                              data_id INT NOT NULL,
                              PRIMARY KEY(spectral_id, feature_id),
                              FOREIGN KEY(level_id_1) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Level(level_id),
                              FOREIGN KEY(database_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Database(database_id),
                              FOREIGN KEY(data_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Data(data_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Fragment cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Fragment(
                         fragment_id INT,
                         mz_value DECIMAL(15,2),
                         relative DECIMAL(15,2),
                         spectral_id INT NOT NULL,
                         feature_id INT NOT NULL,
                         PRIMARY KEY(fragment_id),
                         FOREIGN KEY(spectral_id, feature_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Spectral_data(spectral_id, feature_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Compound cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Compound(
                         coumpound_id INT,
                         spectral_id INT,
                         compound_name VARCHAR(50),
                         smiles VARCHAR(50),
                         class VARCHAR(50),
                         pubchem VARCHAR(50),
                         molecular_formula VARCHAR(50),
                         taxonomy VARCHAR(50),
                         library_quality VARCHAR(50),
                         spectral_id_1 INT NOT NULL,
                         feature_id INT NOT NULL,
                         PRIMARY KEY(coumpound_id, spectral_id),
                         UNIQUE(spectral_id_1, feature_id),
                         FOREIGN KEY(spectral_id_1, feature_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Spectral_data(spectral_id, feature_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Composing cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Composing(
                          experiment_id INT,
                          scan_id VARCHAR(50),
                          spectral_id INT,
                          feature_id INT,
                          PRIMARY KEY(experiment_id, scan_id, spectral_id, feature_id),
                          FOREIGN KEY(experiment_id, scan_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Experiment(experiment_id, scan_id),
                          FOREIGN KEY(spectral_id, feature_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Spectral_data(spectral_id, feature_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Identifying cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Identifying(
                            experiment_id INT,
                            scan_id VARCHAR(50),
                            coumpound_id INT,
                            spectral_id INT,
                            PRIMARY KEY(experiment_id, scan_id, coumpound_id, spectral_id),
                            FOREIGN KEY(experiment_id, scan_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Experiment(experiment_id, scan_id),
                            FOREIGN KEY(coumpound_id, spectral_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Compound(coumpound_id, spectral_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Experimenting cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Experimenting(
                              experiment_id INT,
                              scan_id VARCHAR(50),
                              analytic_data_id INT,
                              date_id INT,
                              PRIMARY KEY(experiment_id, scan_id, analytic_data_id, date_id),
                              FOREIGN KEY(experiment_id, scan_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Experiment(experiment_id, scan_id),
                              FOREIGN KEY(analytic_data_id, date_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Analytics_data(analytic_data_id, date_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Building cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Building(
                         bloc_id INT,
                         spectral_id INT,
                         feature_id INT,
                         PRIMARY KEY(bloc_id, spectral_id, feature_id),
                         FOREIGN KEY(bloc_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".BuildingBlocks(bloc_id),
                         FOREIGN KEY(spectral_id, feature_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Spectral_data(spectral_id, feature_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Tooling cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Tooling(
                        tool_id INT,
                        user_id INT,
                        PRIMARY KEY(tool_id, user_id),
                        FOREIGN KEY(tool_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Tool(tool_id),
                        FOREIGN KEY(user_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Users(user_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Ionising cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Ionising(
                         tool_id INT,
                         inonisation_mode_id INT,
                         PRIMARY KEY(tool_id, inonisation_mode_id),
                         FOREIGN KEY(tool_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Tool(tool_id),
                         FOREIGN KEY(inonisation_mode_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Ionisation_mode(inonisation_mode_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Analysing cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Analysing(
                          tool_id INT,
                          analytic_data_id INT,
                          date_id INT,
                          PRIMARY KEY(tool_id, analytic_data_id, date_id),
                          FOREIGN KEY(tool_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Tool(tool_id),
                          FOREIGN KEY(analytic_data_id, date_id) REFERENCES "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Analytics_data(analytic_data_id, date_id)
);


