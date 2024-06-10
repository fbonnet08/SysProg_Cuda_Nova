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

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".employee_audits cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".employees cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".datetable cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".fragment cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".composing cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".identifying cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".compound cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".experimenting cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".building cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".buildingblocks cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".spectral_data cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".data cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".DatabaseDetails cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".level cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".tooling cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".PlatformUser cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".ionising cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".ionisation_mode cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".experiment cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".analysing cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".tool cascade ;
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".analytics_data cascade ;

/* cleaning upo public */
drop table if exists public_test_table cascade ;
drop table if exists datetable cascade ;
drop table if exists logintable cascade ;
drop table if exists fragment cascade ;
drop table if exists composing cascade ;
drop table if exists identifying cascade ;
drop table if exists compound cascade ;
drop table if exists experimenting cascade ;
drop table if exists building cascade ;
drop table if exists buildingblocks cascade ;
drop table if exists spectral_data cascade ;
drop table if exists data cascade ;
drop table if exists DatabaseDetails cascade ;
drop table if exists tooling cascade ;
drop table if exists PlatformUser cascade ;
drop table if exists ionising cascade ;
drop table if exists ionisation_mode cascade ;
drop table if exists experiment cascade ;
drop table if exists analysing cascade ;
drop table if exists tool cascade ;
drop table if exists analytics_data cascade ;

/* Creating the test tables */
drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".employees cascade ;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".employees (
        employee_id serial primary key,
        name varchar(50) not null
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".employee_audits cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".employee_audits (
        employee_id serial primary key references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".employees(employee_id),
        old_name varchar(50) not null
);

/* Creating the test tables tables */


drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem;
create table if not exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(
    ionmode_id serial,
    chemical_composition varchar(50),
    primary key (ionmode_id)
);

insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(ionmode_id, chemical_composition) values (1, '[M]+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(ionmode_id, chemical_composition) values (2, '[M]+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(ionmode_id, chemical_composition) values (3, '[M+H]+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(ionmode_id, chemical_composition) values (4, '[M+NH4]+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(ionmode_id, chemical_composition) values (5, '[M+Na]+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(ionmode_id, chemical_composition) values (6, '[M+K]+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(ionmode_id, chemical_composition) values (7, '[M+CH3OH+H]+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(ionmode_id, chemical_composition) values (8, '[M+ACN+H]+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(ionmode_id, chemical_composition) values (9, '[M+ACN+Na]+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(ionmode_id, chemical_composition) values (10, '[M+2ACN+H]+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(ionmode_id, chemical_composition) values (11, 'Negative');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(ionmode_id, chemical_composition) values (12, '[M]-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(ionmode_id, chemical_composition) values (13, '[M-H]-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(ionmode_id, chemical_composition) values (14, '[M+Cl]-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(ionmode_id, chemical_composition) values (15, '[M+HCOO]-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(ionmode_id, chemical_composition) values (16, '[M+CH3COO]-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(ionmode_id, chemical_composition) values (17, '[M-2H]-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(ionmode_id, chemical_composition) values (18, '[M-2H+Na]-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".IonModeChem(ionmode_id, chemical_composition) values (19, '[M-2H+K]-');

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".BuildingBlocks cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".BuildingBlocks(
                               bloc_id int,
                               bloc_name VARCHAR(50),
                               bloc_structure VARCHAR(60),
                               PRIMARY KEY(bloc_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Experiment cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Experiment(
                           experiment_id int,
                           scan_id VARCHAR(50),
                           ionisation_mode VARCHAR(50),
                           PRIMARY KEY(experiment_id, scan_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Tool cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Tool(
                     tool_id int,
                     instrument_source VARCHAR(50),
                     PRIMARY KEY(tool_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Data cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Data(
                     data_id int,
                     path_to_data VARCHAR(150),
                     raw_file VARCHAR(60),
                     csv_file VARCHAR(60),
                     xls_file VARCHAR(60),
                     asc_file VARCHAR(60),
                     mgf_file VARCHAR(50),
                     m2s_file VARCHAR(50),
                     json_file VARCHAR(50),
                     experiment_id int NOT NULL,
                     scan_id VARCHAR(50) NOT NULL,
                     PRIMARY KEY(data_id),
                     FOREIGN KEY(experiment_id, scan_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Experiment(experiment_id, scan_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".PlatformUser cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".PlatformUser(
                      platform_user_id int,
                      affiliation VARCHAR(50),
                      phone VARCHAR(50),
                      email VARCHAR(50),
                      PRIMARY KEY(platform_user_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Ionisation_mode cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Ionisation_mode(
                                inonisation_mode_id int,
                                ionisation_mode VARCHAR(50),
                                experiment_id int NOT NULL,
                                scan_id VARCHAR(50) NOT NULL,
                                PRIMARY KEY(inonisation_mode_id),
                                UNIQUE(experiment_id, scan_id),
                                FOREIGN KEY(experiment_id, scan_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Experiment(experiment_id, scan_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Analytics_data cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Analytics_data(
                               analytic_data_id int,
                               date_id INT,
                               sample_name VARCHAR(50),
                               sample_details VARCHAR(100),
                               sample_solvent VARCHAR(50),
                               number_scans VARCHAR(50),
                               filename VARCHAR(50),
                               PRIMARY KEY(analytic_data_id, date_id)
);

INSERT INTO "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Analytics_data (
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

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".DatabaseDetails cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".DatabaseDetails(
                                database_id INT,
                                database_name VARCHAR(50),
                                database_affiliation VARCHAR(50),
                                database_path VARCHAR(100),
                                library_quality_legend VARCHAR(250),
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
                          FOREIGN KEY(analytic_data_id, date_id_1) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Analytics_data(analytic_data_id, date_id)
);

INSERT INTO "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".DateTable (date_id, date_column, time_column, timestamp_with_tz_column, analytic_data_id, date_id_1)
VALUES
    (1, '2024-04-19', '13:30:00', '2024-04-19 13:30:00', 1, 1);


drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".LoginTable cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".LoginTable(
                           login_id INT,
                           platform_user_id VARCHAR(50),
                           firstname VARCHAR(50),
                           lastename VARCHAR(50),
                           username VARCHAR(50),
                           password VARCHAR(50),
                           role VARCHAR(50),
                           affiliation VARCHAR(50),
                           department VARCHAR(50),
                           researchlab VARCHAR(50),
                           PRIMARY KEY(login_id, platform_user_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Spectral_data cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Spectral_data(
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
                              platform_user_id INT NOT NULL,
                              database_id INT NOT NULL,
                              data_id INT NOT NULL,
                              PRIMARY KEY(spectral_id, feature_id, tool_id, pi_name_id, data_collector_id),
                              FOREIGN KEY(platform_user_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".PlatformUser(platform_user_id),
                              FOREIGN KEY(database_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".DatabaseDetails(database_id),
                              FOREIGN KEY(data_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Data(data_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Fragment cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Fragment(
                         fragment_id INT,
                         mz_value DECIMAL(15,2),
                         relative DECIMAL(15,2),
                         spectral_id INT NOT NULL,
                         feature_id INT NOT NULL,
                         tool_id INT NOT NULL,
                         pi_name_id INT NOT NULL,
                         data_collector_id INT NOT NULL,
                         PRIMARY KEY(fragment_id),
                         FOREIGN KEY(spectral_id, feature_id, tool_id, pi_name_id, data_collector_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Spectral_data(spectral_id, feature_id, tool_id, pi_name_id, data_collector_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Compound cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Compound(
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
                         FOREIGN KEY(spectral_id_1, feature_id, tool_id, pi_name_id, data_collector_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Spectral_data(spectral_id, feature_id, tool_id, pi_name_id, data_collector_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Composing cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Composing(
                          experiment_id INT,
                          scan_id VARCHAR(50),
                          spectral_id INT,
                          feature_id INT,
                          tool_id INT,
                          pi_name_id INT,
                          data_collector_id INT,
                          PRIMARY KEY(experiment_id, scan_id, spectral_id, feature_id, tool_id, pi_name_id, data_collector_id),
                          FOREIGN KEY(experiment_id, scan_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Experiment(experiment_id, scan_id),
                          FOREIGN KEY(spectral_id, feature_id, tool_id, pi_name_id, data_collector_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Spectral_data(spectral_id, feature_id, tool_id, pi_name_id, data_collector_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Identifying cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Identifying(
                            experiment_id INT,
                            scan_id VARCHAR(50),
                            coumpound_id INT,
                            spectral_id INT,
                            PRIMARY KEY(experiment_id, scan_id, coumpound_id, spectral_id),
                            FOREIGN KEY(experiment_id, scan_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Experiment(experiment_id, scan_id),
                            FOREIGN KEY(coumpound_id, spectral_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Compound(coumpound_id, spectral_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Experimenting cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Experimenting(
                              experiment_id INT,
                              scan_id VARCHAR(50),
                              analytic_data_id INT,
                              date_id INT,
                              PRIMARY KEY(experiment_id, scan_id, analytic_data_id, date_id),
                              FOREIGN KEY(experiment_id, scan_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Experiment(experiment_id, scan_id),
                              FOREIGN KEY(analytic_data_id, date_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Analytics_data(analytic_data_id, date_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Building cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Building(
                         bloc_id INT,
                         spectral_id INT,
                         feature_id INT,
                         tool_id INT,
                         pi_name_id INT,
                         data_collector_id INT,
                         PRIMARY KEY(bloc_id, spectral_id, feature_id, tool_id, pi_name_id, data_collector_id),
                         FOREIGN KEY(bloc_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".BuildingBlocks(bloc_id),
                         FOREIGN KEY(spectral_id, feature_id, tool_id, pi_name_id, data_collector_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Spectral_data(spectral_id, feature_id, tool_id, pi_name_id, data_collector_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Tooling cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Tooling(
                        tool_id INT,
                        platform_user_id INT,
                        PRIMARY KEY(tool_id, platform_user_id),
                        FOREIGN KEY(tool_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Tool(tool_id),
                        FOREIGN KEY(platform_user_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".PlatformUser(platform_user_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Ionising cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Ionising(
                         tool_id INT,
                         inonisation_mode_id INT,
                         PRIMARY KEY(tool_id, inonisation_mode_id),
                         FOREIGN KEY(tool_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Tool(tool_id),
                         FOREIGN KEY(inonisation_mode_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Ionisation_mode(inonisation_mode_id)
);

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Analysing cascade;
create table if not exists
    "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Analysing(
                          tool_id INT,
                          analytic_data_id INT,
                          date_id INT,
                          PRIMARY KEY(tool_id, analytic_data_id, date_id),
                          FOREIGN KEY(tool_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Tool(tool_id),
                          FOREIGN KEY(analytic_data_id, date_id) references "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Analytics_data(analytic_data_id, date_id)
);

/* Cleaning up public */
drop table if exists public_test_table cascade ;
drop table if exists datetable cascade ;
drop table if exists logintable cascade ;
drop table if exists fragment cascade ;
drop table if exists composing cascade ;
drop table if exists identifying cascade ;
drop table if exists compound cascade ;
drop table if exists experimenting cascade ;
drop table if exists building cascade ;
drop table if exists buildingblocks cascade ;
drop table if exists spectral_data cascade ;
drop table if exists data cascade ;
drop table if exists DatabaseDetails cascade ;
drop table if exists tooling cascade ;
drop table if exists PlatformUser cascade ;
drop table if exists ionising cascade ;
drop table if exists ionisation_mode cascade ;
drop table if exists experiment cascade ;
drop table if exists analysing cascade ;
drop table if exists tool cascade ;
drop table if exists analytics_data cascade ;



