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

drop table if exists MolRefAnt_DB.employee_audits cascade;
drop table if exists MolRefAnt_DB.employees  cascade;
drop table if exists MolRefAnt_DB.datetable  cascade;
drop table if exists MolRefAnt_DB.fragment  cascade;
drop table if exists MolRefAnt_DB.composing  cascade;
drop table if exists MolRefAnt_DB.identifying  cascade;
drop table if exists MolRefAnt_DB.compound  cascade;
drop table if exists MolRefAnt_DB.experimenting  cascade;
drop table if exists MolRefAnt_DB.building  cascade;
drop table if exists MolRefAnt_DB.buildingblocks  cascade;
drop table if exists MolRefAnt_DB.spectral_data  cascade;
drop table if exists MolRefAnt_DB.data cascade ;
drop table if exists MolRefAnt_DB.database  cascade;
drop table if exists MolRefAnt_DB.level  cascade;
drop table if exists MolRefAnt_DB.tooling  cascade;
drop table if exists MolRefAnt_DB.users cascade;
drop table if exists MolRefAnt_DB.ionising  cascade;
drop table if exists MolRefAnt_DB.ionisation_mode  cascade;
drop table if exists MolRefAnt_DB.experiment  cascade;
drop table if exists MolRefAnt_DB.analysing  cascade;
drop table if exists MolRefAnt_DB.tool  cascade;
drop table if exists MolRefAnt_DB.analytics_data  cascade;

drop table if exists MolRefAnt_DB.BuildingBlocks cascade;
drop table if exists MolRefAnt_DB.Compound cascade;
drop table if exists MolRefAnt_DB.Data cascade;
drop table if exists MolRefAnt_DB.Fragmentation cascade;
drop table if exists MolRefAnt_DB.Sample cascade;
drop table if exists MolRefAnt_DB.Tools cascade;
drop table if exists MolRefAnt_DB.build cascade;
drop table if exists MolRefAnt_DB.composer cascade;
drop table if exists MolRefAnt_DB.identifier cascade;
drop table if exists MolRefAnt_DB.sampling cascade;

drop table if exists MolRefAnt_DB.analysing cascade;
drop table if exists MolRefAnt_DB.building cascade;
drop table if exists MolRefAnt_DB.buildingblocks cascade;
drop table if exists MolRefAnt_DB.composing cascade;
drop table if exists MolRefAnt_DB.datetable cascade;
drop table if exists MolRefAnt_DB.employee_audits cascade;
drop table if exists MolRefAnt_DB.employees cascade;
drop table if exists MolRefAnt_DB.experimenting cascade;
drop table if exists MolRefAnt_DB.analytics_data cascade;
drop table if exists MolRefAnt_DB.fragment cascade;
drop table if exists MolRefAnt_DB.identifying cascade;
drop table if exists MolRefAnt_DB.compound cascade;
drop table if exists MolRefAnt_DB.ionising cascade;
drop table if exists MolRefAnt_DB.ionisation_mode cascade;
drop table if exists MolRefAnt_DB.logintable cascade;
drop table if exists MolRefAnt_DB.spectral_data cascade;
drop table if exists MolRefAnt_DB.data cascade;
drop table if exists MolRefAnt_DB.databasedetails cascade;
drop table if exists MolRefAnt_DB.experiment cascade;
drop table if exists MolRefAnt_DB.tooling cascade;
drop table if exists MolRefAnt_DB.tool cascade;
drop table if exists MolRefAnt_DB.users cascade;

drop table if exists MolRefAnt_DB.analysing cascade;
drop table if exists MolRefAnt_DB.building cascade;
drop table if exists MolRefAnt_DB.buildingblock cascade;
drop table if exists MolRefAnt_DB.composing cascade;
drop table if exists MolRefAnt_DB.datetable cascade;
drop table if exists MolRefAnt_DB.employee_audits cascade;
drop table if exists MolRefAnt_DB.employees cascade;
drop table if exists MolRefAnt_DB.experimenting cascade;
drop table if exists MolRefAnt_DB.analytics_data cascade;
drop table if exists MolRefAnt_DB.fragment cascade;
drop table if exists MolRefAnt_DB.identifying cascade;
drop table if exists MolRefAnt_DB.compound cascade;
drop table if exists MolRefAnt_DB.ionising cascade;
drop table if exists MolRefAnt_DB.ionisation_mode cascade;
drop table if exists MolRefAnt_DB.ionmodechem cascade;
drop table if exists MolRefAnt_DB.logintable cascade;
drop table if exists MolRefAnt_DB.spectral_data cascade;
drop table if exists MolRefAnt_DB.data cascade;
drop table if exists MolRefAnt_DB.database_details cascade;
drop table if exists MolRefAnt_DB.experiment cascade;
drop table if exists MolRefAnt_DB.tooling cascade;
drop table if exists MolRefAnt_DB.platform_user cascade;
drop table if exists MolRefAnt_DB.tool cascade;

/* Creating the test tables */
drop table if exists MolRefAnt_DB.employees cascade ;
create table if not exists
    MolRefAnt_DB.employees (
        employee_id integer primary key,
        name varchar(50) not null
);

drop table if exists MolRefAnt_DB.employee_audits cascade;
create table if not exists
    MolRefAnt_DB.employee_audits (
        employee_id integer primary key references MolRefAnt_DB.employees(employee_id),
        old_name varchar(50) not null
);

/* Creating the MolRefAnt_DB database tables */

drop table if exists MolRefAnt_DB.IonModeChem;
create table if not exists
    MolRefAnt_DB.IonModeChem(
                                                            ionmodechem_id serial,
                                                            chemical_composition varchar(50),
                                                            primary key (ionmodechem_id)
);

insert into MolRefAnt_DB.IonModeChem(ionmodechem_id, chemical_composition) values (1, '[M]+');
insert into MolRefAnt_DB.IonModeChem(ionmodechem_id, chemical_composition) values (2, '[M]+');
insert into MolRefAnt_DB.IonModeChem(ionmodechem_id, chemical_composition) values (3, '[M+H]+');
insert into MolRefAnt_DB.IonModeChem(ionmodechem_id, chemical_composition) values (4, '[M+NH4]+');
insert into MolRefAnt_DB.IonModeChem(ionmodechem_id, chemical_composition) values (5, '[M+Na]+');
insert into MolRefAnt_DB.IonModeChem(ionmodechem_id, chemical_composition) values (6, '[M+K]+');
insert into MolRefAnt_DB.IonModeChem(ionmodechem_id, chemical_composition) values (7, '[M+CH3OH+H]+');
insert into MolRefAnt_DB.IonModeChem(ionmodechem_id, chemical_composition) values (8, '[M+ACN+H]+');
insert into MolRefAnt_DB.IonModeChem(ionmodechem_id, chemical_composition) values (9, '[M+ACN+Na]+');
insert into MolRefAnt_DB.IonModeChem(ionmodechem_id, chemical_composition) values (10, '[M+2ACN+H]+');
insert into MolRefAnt_DB.IonModeChem(ionmodechem_id, chemical_composition) values (11, 'Negative');
insert into MolRefAnt_DB.IonModeChem(ionmodechem_id, chemical_composition) values (12, '[M]-');
insert into MolRefAnt_DB.IonModeChem(ionmodechem_id, chemical_composition) values (13, '[M-H]-');
insert into MolRefAnt_DB.IonModeChem(ionmodechem_id, chemical_composition) values (14, '[M+Cl]-');
insert into MolRefAnt_DB.IonModeChem(ionmodechem_id, chemical_composition) values (15, '[M+HCOO]-');
insert into MolRefAnt_DB.IonModeChem(ionmodechem_id, chemical_composition) values (16, '[M+CH3COO]-');
insert into MolRefAnt_DB.IonModeChem(ionmodechem_id, chemical_composition) values (17, '[M-2H]-');
insert into MolRefAnt_DB.IonModeChem(ionmodechem_id, chemical_composition) values (18, '[M-2H+Na]-');
insert into MolRefAnt_DB.IonModeChem(ionmodechem_id, chemical_composition) values (19, '[M-2H+K]-');

/* Creating the test tables tables */

drop table if exists
    MolRefAnt_DB.BuildingBlock cascade;
create table if not exists
    MolRefAnt_DB.BuildingBlock(
                                                              buildingblock_id int,
                                                              buildingblock_name VARCHAR(50),
                                                              buildingblock_structure VARCHAR(60),
                                                              PRIMARY
                                                                  KEY (buildingblock_id)
);

drop table if exists MolRefAnt_DB.Experiment
    cascade;
create table if not exists
    MolRefAnt_DB.Experiment(
                                                           experiment_id INT,
                                                           scan_id VARCHAR(50) NOT NULL,
                                                           ionisation_mode VARCHAR(50),
                                                           PRIMARY KEY(experiment_id)
);

drop table if exists MolRefAnt_DB.Tool cascade;
create table if not exists
    MolRefAnt_DB.Tool(
                                                     tool_id int,
                                                     instrument_source
                                                             VARCHAR(50),
                                                     PRIMARY KEY(tool_id)
);

drop table if exists MolRefAnt_DB.Data cascade;
create table if not exists
    MolRefAnt_DB.Data(
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
                                                     PRIMARY KEY(data_id),
                                                     FOREIGN KEY(experiment_id) REFERENCES
                                                         MolRefAnt_DB.Experiment(experiment_id)
);

drop table if exists
    MolRefAnt_DB.Platform_User cascade;
create table if not exists
    MolRefAnt_DB.Platform_User(
                                                              platform_user_id int,
                                                              firstname VARCHAR(50),
                                                              lastname VARCHAR(50),
                                                              affiliation VARCHAR(50),
                                                              phone VARCHAR(50),
                                                              email VARCHAR(50),
                                                              PRIMARY KEY(platform_user_id)
);

drop table if exists
    MolRefAnt_DB.Ionisation_mode cascade;
create table if not exists
    MolRefAnt_DB.Ionisation_mode(
                                                                inonisation_mode_id INT,
                                                                ionisation_mode VARCHAR(50),
                                                                experiment_id INT NOT NULL,
                                                                PRIMARY KEY(inonisation_mode_id),
                                                                UNIQUE(experiment_id),
                                                                FOREIGN KEY(experiment_id) REFERENCES
                                                                    MolRefAnt_DB.Experiment(experiment_id)
);

drop table if exists
    MolRefAnt_DB.Analytics_data cascade;
create table if not exists
    MolRefAnt_DB.Analytics_data(
                                                               analytics_data_id INT,
                                                               date_id INT,
                                                               sample_name VARCHAR(50),
                                                               sample_details VARCHAR(100),
                                                               sample_solvent VARCHAR(50),
                                                               number_scans VARCHAR(50),
                                                               filename VARCHAR(50),
                                                               PRIMARY KEY(analytics_data_id, date_id),
                                                               unique (analytics_data_id)
);
INSERT INTO MolRefAnt_DB.Analytics_data (
    analytics_data_id,
    date_id,
    sample_name,
    sample_details,
    sample_solvent,
    number_scans,
    filename
)
VALUES
    (1, 1, 'sample 1', 'details 1', 'solvent 1', '4114', 'filename.raw');

drop table if exists
    MolRefAnt_DB.Database_Details cascade;
create table if not exists
    MolRefAnt_DB.Database_Details(
                                                                 database_id INT,
                                                                 database_name VARCHAR(50),
                                                                 database_affiliation
                                                                     VARCHAR(50),
                                                                 database_path VARCHAR(100),
                                                                 library_quality_legend VARCHAR(250),
                                                                 PRIMARY
                                                                     KEY(database_id)
);

drop table if exists MolRefAnt_DB.DateTable
    cascade;
create table if not exists
    MolRefAnt_DB.DateTable(
                                                          date_id INT,
                                                          date_column DATE,
                                                          time_column TIME,
                                                          timestamp_with_tz_column VARCHAR(50),
                                                          analytics_data_id INT NOT NULL,
                                                          date_id_1 INT
                                                                                NOT NULL,
                                                          PRIMARY
                                                              KEY(date_id),
                                                          UNIQUE(analytics_data_id, date_id_1),
                                                          FOREIGN
                                                              KEY(analytics_data_id, date_id_1) references
                                                              MolRefAnt_DB.Analytics_data(analytics_data_id,
                                                                                                                      date_id)
);

INSERT INTO MolRefAnt_DB.DateTable (date_id,
                                                                date_column, time_column, timestamp_with_tz_column, analytics_data_id,
                                                                date_id_1)
VALUES
    (1, '2024-04-19', '13:30:00', '2024-04-19 13:30:00', 1, 1);

drop table if exists MolRefAnt_DB.LoginTable
    cascade;
create table if not exists
    MolRefAnt_DB.LoginTable(
                                                           login_id INT,
                                                           user_id VARCHAR(50) NOT NULL,
                                                           firstname VARCHAR(50),
                                                           lastename VARCHAR(50),
                                                           username VARCHAR(50),
                                                           password VARCHAR(50),
                                                           role VARCHAR(50),
                                                           affiliation VARCHAR(50),
                                                           department VARCHAR(50),
                                                           researchlab VARCHAR(50),
                                                           PRIMARY KEY(login_id)
);

drop table if exists
    MolRefAnt_DB.Spectral_data cascade;
create table if not exists
    MolRefAnt_DB.Spectral_data(
                                                              spectral_data_id INT,
                                                              feature_id INT NOT NULL,
                                                              pepmass DECIMAL(15,2),
                                                              ionmodechem_id INT NOT NULL,
                                                              charge VARCHAR(10),
                                                              MSLevel INT NOT NULL,
                                                              scan_number VARCHAR(50),
                                                              retention_time TIME,
                                                              mol_json_file VARCHAR(50),
                                                              tool_id INT NOT NULL,
                                                              pi_name_id INT NOT NULL,
                                                              data_collector_id INT NOT NULL,
                                                              database_id INT NOT NULL,
                                                              data_id INT NOT NULL,
                                                              num_peaks INT,
                                                              peaks_list TEXT,
                                                              PRIMARY KEY(spectral_data_id),
                                                              FOREIGN KEY(tool_id) REFERENCES
                                                                  MolRefAnt_DB.Tool(tool_id),
                                                              FOREIGN KEY(database_id) REFERENCES
                                                                  MolRefAnt_DB.Database_Details(database_id),
                                                              FOREIGN KEY(data_id) REFERENCES
                                                                  MolRefAnt_DB.Data(data_id),
                                                              FOREIGN KEY(pi_name_id) REFERENCES
                                                                  MolRefAnt_DB.Platform_User(platform_user_id),
                                                              FOREIGN KEY(data_collector_id) REFERENCES
                                                                  MolRefAnt_DB.Platform_User(platform_user_id)

);

drop table if exists MolRefAnt_DB.Fragment
    cascade;
create table if not exists
    MolRefAnt_DB.Fragment(
                                                         fragment_id INT,
                                                         mz_value DECIMAL(15,2),
                                                         relative DECIMAL(15,2),
                                                         spectral_data_id INT NOT NULL,
                                                         PRIMARY KEY(fragment_id),
                                                         FOREIGN KEY(spectral_data_id) REFERENCES
                                                             MolRefAnt_DB.Spectral_data(spectral_data_id)
);

drop table if exists MolRefAnt_DB.Compound
    cascade;
create table if not exists
    MolRefAnt_DB.Compound(
                                                         coumpound_id INT,
                                                         spectral_data_id INT NOT NULL,
                                                         compound_name VARCHAR(50),
                                                         smiles VARCHAR(50),
                                                         pubchem VARCHAR(50),
                                                         molecular_formula VARCHAR(50),
                                                         taxonomy VARCHAR(50),
                                                         library_quality VARCHAR(50),
                                                         spectral_id_1 INT NOT NULL,
                                                         PRIMARY KEY(coumpound_id),

                                                         FOREIGN KEY(spectral_data_id) REFERENCES
                                                             MolRefAnt_DB.Spectral_data(spectral_data_id)
);

drop table if exists MolRefAnt_DB.Composing
    cascade;
create table if not exists
    MolRefAnt_DB.Composing(
                                                          experiment_id INT,
                                                          spectral_data_id INT,
                                                          PRIMARY KEY(experiment_id, spectral_data_id),
                                                          FOREIGN KEY(experiment_id) REFERENCES
                                                              MolRefAnt_DB.Experiment(experiment_id),
                                                          FOREIGN KEY(spectral_data_id) REFERENCES
                                                              MolRefAnt_DB.Spectral_data(spectral_data_id)
);

drop table if exists
    MolRefAnt_DB.Identifying cascade;
create table if not exists
    MolRefAnt_DB.Identifying(
                                                            experiment_id INT,
                                                            coumpound_id INT,
                                                            PRIMARY KEY(experiment_id, coumpound_id),
                                                            FOREIGN KEY(experiment_id) REFERENCES
                                                                MolRefAnt_DB.Experiment(experiment_id),
                                                            FOREIGN KEY(coumpound_id) REFERENCES
                                                                MolRefAnt_DB.Compound(coumpound_id)
);

drop table if exists
    MolRefAnt_DB.Experimenting cascade;
create table if not exists
    MolRefAnt_DB.Experimenting(
                                                              experiment_id INT,
                                                              analytics_data_id INT,
                                                              PRIMARY KEY(experiment_id),
                                                              unique (analytics_data_id),
                                                              FOREIGN KEY(experiment_id) REFERENCES
                                                                  MolRefAnt_DB.Experiment(experiment_id),
                                                              FOREIGN KEY(analytics_data_id) REFERENCES
                                                                  MolRefAnt_DB.Analytics_data(analytics_data_id)
);

drop table if exists MolRefAnt_DB.Building
    cascade;
create table if not exists
    MolRefAnt_DB.Building(
                                                         buildingblock_id INT,
                                                         spectral_data_id INT,
                                                         PRIMARY KEY(buildingblock_id, spectral_data_id),
                                                         FOREIGN KEY(buildingblock_id) REFERENCES
                                                             MolRefAnt_DB.BuildingBlock(buildingblock_id),
                                                         FOREIGN KEY(spectral_data_id) REFERENCES
                                                             MolRefAnt_DB.Spectral_data(spectral_data_id)
);

drop table if exists MolRefAnt_DB.Tooling
    cascade;
create table if not exists
    MolRefAnt_DB.Tooling(
                                                        tool_id INT,
                                                        platform_user_id INT,
                                                        PRIMARY KEY(tool_id, platform_user_id),
                                                        FOREIGN KEY(tool_id) REFERENCES
                                                            MolRefAnt_DB.Tool(tool_id),
                                                        FOREIGN KEY(platform_user_id) REFERENCES
                                                            MolRefAnt_DB.Platform_User(platform_user_id)
);

drop table if exists MolRefAnt_DB.Ionising
    cascade;
create table if not exists
    MolRefAnt_DB.Ionising(
                                                         tool_id INT,
                                                         inonisation_mode_id INT,
                                                         PRIMARY KEY(tool_id, inonisation_mode_id),
                                                         FOREIGN KEY(tool_id) REFERENCES
                                                             MolRefAnt_DB.Tool(tool_id),
                                                         FOREIGN KEY(inonisation_mode_id) REFERENCES
                                                             MolRefAnt_DB.Ionisation_mode(inonisation_mode_id)
);

drop table if exists MolRefAnt_DB.Analysing
    cascade;
create table if not exists
    MolRefAnt_DB.Analysing(
                                                          tool_id INT,
                                                          analytic_data_id INT,
                                                          PRIMARY KEY(tool_id),
                                                          FOREIGN KEY(tool_id) REFERENCES
                                                              MolRefAnt_DB.Tool(tool_id),
                                                          FOREIGN KEY(analytic_data_id) REFERENCES
                                                              MolRefAnt_DB.Analytics_data(analytics_data_id)
);
