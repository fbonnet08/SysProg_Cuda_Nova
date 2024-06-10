/* ----------------------------------
-- TP-1   MolRefAnt_DB_PostGreSQL Database
----------------------------------
-- Section 1
----------------------------------
-- First cleaning up the previous implementations for fresh start
*/
/* Creating the Schema to begin with */
/*
--drop schema if exists "MolRefAnt_DB" cascade ;
--CREATE SCHEMA "MolRefAnt_DB" AUTHORIZATION frederic;
--COMMENT ON SCHEMA "MolRefAnt_DB" IS 'Creating the schema for the MolRefAnt_DB_PostGreSQL';
*/
/* Dropping all tables */
/*
drop table if exists BuildingBlocks cascade;
drop table if exists Compound cascade;
drop table if exists Data cascade;
drop table if exists Fragmentation cascade;
drop table if exists Sample cascade;
drop table if exists Tools cascade;
drop table if exists build cascade;
drop table if exists composer cascade;
drop table if exists identifier cascade;
drop table if exists sampling cascade;

drop table if exists Analysing cascade;
drop table if exists Analytics_data cascade;
drop table if exists Building cascade;
drop table if exists BuildingBlocks cascade;
drop table if exists Composing cascade;
drop table if exists Compound cascade;
drop table if exists Data cascade;
drop table if exists DatabaseDetails cascade;
drop table if exists DateTable cascade;
drop table if exists Experiment cascade;
drop table if exists Experimenting cascade;
drop table if exists Fragment cascade;
drop table if exists Identifying cascade;
drop table if exists Ionisation_mode cascade;
drop table if exists Ionising cascade;
drop table if exists LoginTable cascade;
drop table if exists Spectral_data cascade;
drop table if exists Tool cascade;
drop table if exists Tooling cascade;
drop table if exists Users cascade;
drop table if exists employee_audits cascade;
drop table if exists employees cascade;
*/
/* Creating the test tables */
/*
drop table if exists annoter_l_inconnu.employees cascade ;
create table if not exists
    annoter_l_inconnu.employees (
        employee_id integer primary key,
        name varchar(50) not null
);

drop table if exists annoter_l_inconnu.employee_audits cascade;
create table if not exists
    annoter_l_inconnu.employee_audits (
        employee_id integer primary key references annoter_l_inconnu.employees(employee_id),
        old_name varchar(50) not null
);

/* Creating the test tables tables */

create table if not exists
    BuildingBlocks(
                               bloc_id INT,
                               bloc_name VARCHAR(50),
                               bloc_structure VARCHAR(60),
                               PRIMARY KEY(bloc_id)
);

create table if not exists
    Experiment(
                           experiment_id INT,
                           scan_id VARCHAR(50),
                           ionisation_mode VARCHAR(50),
                           PRIMARY KEY(experiment_id, scan_id)
);

create table if not exists
    Tool(
                     tool_id INT,
                     instrument_source VARCHAR(50),
                     PRIMARY KEY(tool_id)
);

create table if not exists
    Data(
                     data_id INT,
                     path_to_data VARCHAR(150),
                     raw_file VARCHAR(60),
                     csv_file VARCHAR(60),
                     xls_file VARCHAR(60),
                     asc_file VARCHAR(60),
                     mgf_file VARCHAR(50),
                     m2s_file VARCHAR(50),
                     json_file VARCHAR(50),
                     experiment_id INT NOT NULL,
                     scan_id VARCHAR(50) NOT NULL,
                     PRIMARY KEY(data_id),
                     FOREIGN KEY(experiment_id, scan_id) references Experiment(experiment_id, scan_id)
);

create table if not exists
    Users(
                      user_id INT,
                      affiliation VARCHAR(50),
                      phone VARCHAR(50),
                      email VARCHAR(50),
                      PRIMARY KEY(user_id)
);

create table if not exists
    Ionisation_mode(
                                inonisation_mode_id INT,
                                ionisation_mode VARCHAR(50),
                                experiment_id INT NOT NULL,
                                scan_id VARCHAR(50) NOT NULL,
                                PRIMARY KEY(inonisation_mode_id),
                                UNIQUE(experiment_id, scan_id),
                                FOREIGN KEY(experiment_id, scan_id) references Experiment(experiment_id, scan_id)
);

create table if not exists
    Analytics_data(
                               analytic_data_id INT,
                               date_id INT,
                               sample_name VARCHAR(50),
                               sample_details VARCHAR(100),
                               sample_solvent VARCHAR(50),
                               number_scans VARCHAR(50),
                               filename VARCHAR(50),
                               PRIMARY KEY(analytic_data_id, date_id)
);
/*
INSERT INTO Analytics_data (
    analytic_data_id,
    date_id,
    sample_name,
    sample_details,
    sample_solvent,
    number_scans,
    filename
)
VALUES
    (1, 1, 'sample 1', 'details 1', 'solvent 1', '4114', 'filename.raw');
*/
create table if not exists
    DatabaseDetails(
                                database_id INT,
                                database_name VARCHAR(50),
                                database_affiliation VARCHAR(50),
                                database_path VARCHAR(100),
                                library_quality_legend VARCHAR(250),
                                PRIMARY KEY(database_id)
);

create table if not exists
    DateTable(
                          date_id INT,
                          date_column DATE,
                          time_column TIME,
                          timestamp_with_tz_column VARCHAR(50),
                          analytic_data_id INT NOT NULL,
                          date_id_1 INT NOT NULL,
                          PRIMARY KEY(date_id),
                          UNIQUE(analytic_data_id, date_id_1),
                          FOREIGN KEY(analytic_data_id, date_id_1) references Analytics_data(analytic_data_id, date_id)
);
/*
INSERT INTO DateTable (date_id, date_column, time_column, timestamp_with_tz_column, analytic_data_id, date_id_1)
VALUES
    (1, '2024-04-19', '13:30:00', '2024-04-19 13:30:00', 1, 1);
*/

create table if not exists
    LoginTable(
                           login_id INT,
                           user_id VARCHAR(50),
                           firstname VARCHAR(50),
                           lastename VARCHAR(50),
                           username VARCHAR(50),
                           password VARCHAR(50),
                           role VARCHAR(50),
                           affiliation VARCHAR(50),
                           department VARCHAR(50),
                           researchlab VARCHAR(50),
                           PRIMARY KEY(login_id, user_id)
);

create table if not exists
    Spectral_data(
                              spectral_id INT,
                              feature_id INT,
                              tool_id INT,
                              pi_name_id INT,
                              data_collector_id INT,
                              pepmass DECIMAL(15,2),
                              ion_mode VARCHAR(50),
                              charge DECIMAL(15,2),
                              MSLevel INT NOT NULL,
                              scan_number VARCHAR(50),
                              retention_time TIME,
                              mol_json_file VARCHAR(50),
                              tool_id_1 INT NOT NULL,
                              user_id INT NOT NULL,
                              database_id INT NOT NULL,
                              data_id INT NOT NULL,
                              PRIMARY KEY(spectral_id, feature_id, tool_id, pi_name_id, data_collector_id),
                              FOREIGN KEY(tool_id_1) references Tool(tool_id),
                              FOREIGN KEY(user_id) references Users(user_id),
                              FOREIGN KEY(database_id) references DatabaseDetails(database_id),
                              FOREIGN KEY(data_id) references Data(data_id)
);

create table if not exists
    Fragment(
                         fragment_id INT,
                         mz_value DECIMAL(15,2),
                         relative DECIMAL(15,2),
                         spectral_id INT NOT NULL,
                         feature_id INT NOT NULL,
                         tool_id INT NOT NULL,
                         pi_name_id INT NOT NULL,
                         data_collector_id INT NOT NULL,
                         PRIMARY KEY(fragment_id),
                         FOREIGN KEY(spectral_id, feature_id, tool_id, pi_name_id, data_collector_id) references Spectral_data(spectral_id, feature_id, tool_id, pi_name_id, data_collector_id)
);

create table if not exists
    Compound(
                         coumpound_id INT,
                         spectral_id INT,
                         compound_name VARCHAR(50),
                         smiles VARCHAR(50),
                         pubchem VARCHAR(50),
                         molecular_formula VARCHAR(50),
                         taxonomy VARCHAR(50),
                         library_quality VARCHAR(50),
                         spectral_id_1 INT NOT NULL,
                         feature_id INT NOT NULL,
                         tool_id INT NOT NULL,
                         pi_name_id INT NOT NULL,
                         data_collector_id INT NOT NULL,
                         PRIMARY KEY(coumpound_id, spectral_id),
                         UNIQUE(spectral_id_1, feature_id, tool_id, pi_name_id, data_collector_id),
                         FOREIGN KEY(spectral_id_1, feature_id, tool_id, pi_name_id, data_collector_id) references Spectral_data(spectral_id, feature_id, tool_id, pi_name_id, data_collector_id)
);

create table if not exists
    Composing(
                          experiment_id INT,
                          scan_id VARCHAR(50),
                          spectral_id INT,
                          feature_id INT,
                          tool_id INT,
                          pi_name_id INT,
                          data_collector_id INT,
                          PRIMARY KEY(experiment_id, scan_id, spectral_id, feature_id, tool_id, pi_name_id, data_collector_id),
                          FOREIGN KEY(experiment_id, scan_id) references Experiment(experiment_id, scan_id),
                          FOREIGN KEY(spectral_id, feature_id, tool_id, pi_name_id, data_collector_id) references Spectral_data(spectral_id, feature_id, tool_id, pi_name_id, data_collector_id)
);

create table if not exists
    Identifying(
                            experiment_id INT,
                            scan_id VARCHAR(50),
                            coumpound_id INT,
                            spectral_id INT,
                            PRIMARY KEY(experiment_id, scan_id, coumpound_id, spectral_id),
                            FOREIGN KEY(experiment_id, scan_id) references Experiment(experiment_id, scan_id),
                            FOREIGN KEY(coumpound_id, spectral_id) references Compound(coumpound_id, spectral_id)
);

create table if not exists
    Experimenting(
                              experiment_id INT,
                              scan_id VARCHAR(50),
                              analytic_data_id INT,
                              date_id INT,
                              PRIMARY KEY(experiment_id, scan_id, analytic_data_id, date_id),
                              FOREIGN KEY(experiment_id, scan_id) references Experiment(experiment_id, scan_id),
                              FOREIGN KEY(analytic_data_id, date_id) references Analytics_data(analytic_data_id, date_id)
);

create table if not exists
    Building(
                         bloc_id INT,
                         spectral_id INT,
                         feature_id INT,
                         tool_id INT,
                         pi_name_id INT,
                         data_collector_id INT,
                         PRIMARY KEY(bloc_id, spectral_id, feature_id, tool_id, pi_name_id, data_collector_id),
                         FOREIGN KEY(bloc_id) references BuildingBlocks(bloc_id),
                         FOREIGN KEY(spectral_id, feature_id, tool_id, pi_name_id, data_collector_id) references Spectral_data(spectral_id, feature_id, tool_id, pi_name_id, data_collector_id)
);

create table if not exists
    Tooling(
                        tool_id INT,
                        user_id INT,
                        PRIMARY KEY(tool_id, user_id),
                        FOREIGN KEY(tool_id) references Tool(tool_id),
                        FOREIGN KEY(user_id) references Users(user_id)
);

create table if not exists
    Ionising(
                         tool_id INT,
                         inonisation_mode_id INT,
                         PRIMARY KEY(tool_id, inonisation_mode_id),
                         FOREIGN KEY(tool_id) references Tool(tool_id),
                         FOREIGN KEY(inonisation_mode_id) references Ionisation_mode(inonisation_mode_id)
);

create table if not exists
    Analysing(
                          tool_id INT,
                          analytic_data_id INT,
                          date_id INT,
                          PRIMARY KEY(tool_id, analytic_data_id, date_id),
                          FOREIGN KEY(tool_id) references Tool(tool_id),
                          FOREIGN KEY(analytic_data_id, date_id) references Analytics_data(analytic_data_id, date_id)
);

