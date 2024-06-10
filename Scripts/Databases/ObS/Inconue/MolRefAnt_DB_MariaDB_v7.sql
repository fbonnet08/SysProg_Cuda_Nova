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

drop table if exists building_block ;
drop table if exists charge ;
drop table if exists database_details ;
drop table if exists employee ;
drop table if exists experiment ;
drop table if exists ionisation_mode ;
drop table if exists ionmodechem ;
drop table if exists tool ;

/* cleanup on v7 */
drop table if exists analysing ;
drop table if exists building ;
drop table if exists buildingblock ;
drop table if exists composing ;
drop table if exists datetable ;
drop table if exists employee_audits ;
drop table if exists employees ;
drop table if exists experimenting ;
drop table if exists analytics_data ;
drop table if exists fragment ;
drop table if exists identifying ;
drop table if exists compound ;
drop table if exists ionising ;
drop table if exists ionisation_mode ;
drop table if exists ionmodechem ;
drop table if exists logintable ;
drop table if exists spectral_data ;
drop table if exists data ;
drop table if exists database_details ;
drop table if exists experiment ;
drop table if exists tooling ;
drop table if exists platform_user ;
drop table if exists tool ;

drop table if exists Analysing;
drop table if exists Building;
drop table if exists Building_Block;
drop table if exists Composing;
drop table if exists DateTable;
drop table if exists Experimenting;
drop table if exists Analytics_data;
drop table if exists Fragment;
drop table if exists Identifying;
drop table if exists Compound;
drop table if exists Ionising;
drop table if exists LoginTable;
drop table if exists Spectral_data;
drop table if exists Charge;
drop table if exists Data;
drop table if exists Database_Details;
drop table if exists Experiment;
drop table if exists Ionisation_mode;
drop table if exists Tooling;
drop table if exists Platform_User;
drop table if exists Tool;
drop table if exists employee_audits;
drop table if exists employee;
drop table if exists ionmodechem;

/* Creating the test tables */
drop table if exists employee  ;
create table if not exists
    employee (
                                                          employee_id serial primary key,
                                                          name varchar(50) not null
);

drop table if exists employee_audits ;
create table if not exists
    employee_audits (
                                                                 employee_id serial primary key references
                                                                     employee(employee_id),
                                                                 old_name varchar(50) not null
);

/* Creating the MolRefAnt_DB database tables */

drop table if exists IonModeChem;
create table if not exists
    IonModeChem(
                                                            ionmodechem_id serial,
                                                            chemical_composition varchar(50),
                                                            primary key (ionmodechem_id)
);

/* Creating the database tables tables */

drop table if exists Building_Block ;
create table if not exists
    Building_Block(
                               building_block_id INT,
                               building_block_name VARCHAR(60),
                               building_block_structure VARCHAR(100),
                               PRIMARY KEY(building_block_id)
);

drop table if exists Tool ;
create table if not exists
    Tool(
                     tool_id INT,
                     instrument_source VARCHAR(50),
                     PRIMARY KEY(tool_id)
);

insert into tool(tool_id, instrument_source) VALUES (1, 'Banyuls_QExactive_Focus');

drop table if exists Platform_User ;
create table if not exists
    Platform_User(
                              platform_user_id INT,
                              firstname VARCHAR(50),
                              lastname VARCHAR(50),
                              name VARCHAR(100),
                              affiliation VARCHAR(50),
                              phone VARCHAR(50),
                              email VARCHAR(50),
                              PRIMARY KEY(platform_user_id)
);

drop table if exists Ionisation_mode ;
create table if not exists
    Ionisation_mode(
                                ionisation_mode_id INT,
                                ionisation_mode VARCHAR(50),
                                signum INT,
                                PRIMARY KEY(ionisation_mode_id)
);

drop table if exists Analytics_data ;
create table if not exists
    Analytics_data(
                                                               analytics_data_id INT,
                                                               date_id INT,
                                                               sample_name VARCHAR(50),
                                                               sample_details VARCHAR(100),
                                                               sample_solvent VARCHAR(50),
                                                               number_scans VARCHAR(50),
                                                               filename VARCHAR(50),
                                                               PRIMARY KEY(analytics_data_id, date_id)
);
INSERT INTO Analytics_data (
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

drop table if exists Database_Details ;
create table if not exists
    Database_Details(
                                                                 database_id INT,
                                                                 database_name VARCHAR(50),
                                                                 database_affiliation VARCHAR(50),
                                                                 database_path VARCHAR(100),
                                                                 library_quality_legend VARCHAR(250),
                                                                 PRIMARY KEY(database_id)
);

drop table if exists DateTable ;
create table if not exists
    DateTable(
                                                          date_id INT,
                                                          date_column DATE,
                                                          time_column TIME,
                                                          timestamp_with_tz_column VARCHAR(50),
                                                          analytics_data_id INT NOT NULL,
                                                          date_id_1 INT NOT NULL,
                                                          PRIMARY KEY(date_id),
                                                          UNIQUE(analytics_data_id, date_id_1),
                                                          FOREIGN KEY(analytics_data_id, date_id_1) REFERENCES Analytics_data(analytics_data_id, date_id)
);

INSERT INTO DateTable (date_id,
                                                                date_column, time_column, timestamp_with_tz_column, analytics_data_id,
                                                                date_id_1)
VALUES
    (1, '2024-04-19', '13:30:00', '2024-04-19 13:30:00', 1, 1);

drop table if exists LoginTable ;
create table if not exists
    LoginTable(
                                                           login_id INT,
                                                           user_id VARCHAR(50) NOT NULL,
                                                           firstname VARCHAR(50),
                                                           lastname VARCHAR(50),
                                                           username VARCHAR(50),
                                                           password VARCHAR(50),
                                                           role VARCHAR(50),
                                                           affiliation VARCHAR(50),
                                                           department VARCHAR(50),
                                                           researchlab VARCHAR(50),
                                                           PRIMARY KEY(login_id)
);

insert into LoginTable(login_id, user_id, firstname, lastname, username, password, role, affiliation, department, researchlab)
values (1, '1000', 'Frederic', 'Bonnet', 'fbonnet', 'Fred1234!', 'admin', 'Banyuls', 'OBS', null);
insert into LoginTable(login_id, user_id, firstname, lastname, username, password, role, affiliation, department, researchlab)
values (2, '1001', 'Nicolas', 'Desmeuriaux', 'ndesmeuriaux', 'Nico1234!', 'admin', 'Banyuls', 'OBS', null);

drop table if exists ionmodechem ;
create table if not exists
    ionmodechem(
                                                            ionmodechem_id INT,
                                                            chemical_composition VARCHAR(80),
                                                            PRIMARY KEY(ionmodechem_id)
);

drop table if exists Charge ;
create table if not exists
    Charge(
                                                       charge_id INT,
                                                       charge VARCHAR(10),
                                                       PRIMARY KEY(charge_id)
);

insert into Charge(charge_id, charge) VALUES (1, '5-');
insert into Charge(charge_id, charge) VALUES (2, '4-');
insert into Charge(charge_id, charge) VALUES (3, '3-');
insert into Charge(charge_id, charge) VALUES (4, '2-');
insert into Charge(charge_id, charge) VALUES (5, '1-');
insert into Charge(charge_id, charge) VALUES (6, '0' );
insert into Charge(charge_id, charge) VALUES (7, '1+');
insert into Charge(charge_id, charge) VALUES (8, '2+');
insert into Charge(charge_id, charge) VALUES (9, '3+');
insert into Charge(charge_id, charge) VALUES (10,'4+');
insert into Charge(charge_id, charge) VALUES (11,'5+');

drop table if exists Experiment ;
create table if not exists
    Experiment(
                                                           experiment_id INT,
                                                           scan_id INT NOT NULL,
                                                           ionisation_mode_id int,
                                                           ionisation_mode_id_1 INT NOT NULL,
                                                           PRIMARY KEY(experiment_id),
                                                           UNIQUE(ionisation_mode_id_1),
                                                           foreign key (ionisation_mode_id_1) references Ionisation_mode(ionisation_mode_id)
);

drop table if exists Data ;
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
                                                     PRIMARY KEY(data_id),
                                                     FOREIGN KEY(experiment_id) REFERENCES Experiment(experiment_id)
);

drop table if exists Spectral_data ;
create table if not exists
    Spectral_data(
                                                              spectral_data_id INT,
                                                              feature_id INT NOT NULL,
                                                              pepmass DECIMAL(15,2),
                                                              ionmodechem_id INT,
                                                              charge_id INT,
                                                              MSLevel INT NOT NULL,
                                                              scan_number VARCHAR(50),
                                                              retention_time TIME,
                                                              mol_json_file VARCHAR(50),
                                                              tool_id INT NOT NULL,
                                                              pi_name_id INT NOT NULL,
                                                              data_collector_id INT NOT NULL,
                                                              num_peaks INT,
                                                              peaks_list VARCHAR(50),
                                                              ionmodechem_id_1 INT NOT NULL,
                                                              charge_id_1 INT NOT NULL,
                                                              tool_id_1 INT NOT NULL,
                                                              database_id INT NOT NULL,
                                                              data_id INT NOT NULL,
                                                              PRIMARY KEY(spectral_data_id),
                                                              FOREIGN KEY(ionmodechem_id_1) REFERENCES ionmodechem(ionmodechem_id),
                                                              FOREIGN KEY(charge_id_1) REFERENCES Charge(charge_id),
                                                              FOREIGN KEY(tool_id_1) REFERENCES Tool(tool_id),
                                                              FOREIGN KEY(database_id) REFERENCES Database_Details(database_id),
                                                              FOREIGN KEY(data_id) REFERENCES Data(data_id),
                                                              FOREIGN KEY(pi_name_id) REFERENCES Platform_User(platform_user_id),
                                                              FOREIGN KEY(data_collector_id) REFERENCES Platform_User(platform_user_id)
);

drop table if exists Fragment ;
create table if not exists
    Fragment(
                                                         fragment_id INT,
                                                         mz_value DECIMAL(15,2),
                                                         relative DECIMAL(15,2),
                                                         spectral_data_id INT NOT NULL,
                                                         PRIMARY KEY(fragment_id),
                                                         FOREIGN KEY(spectral_data_id) REFERENCES Spectral_data(spectral_data_id)
);

drop table if exists Compound ;
create table if not exists
    Compound(
                                                         coumpound_id INT,
                                                         spectral_data_id INT NOT NULL,
                                                         compound_name VARCHAR(50),
                                                         smiles VARCHAR(50),
                                                         pubchem VARCHAR(50),
                                                         molecular_formula VARCHAR(50),
                                                         taxonomy VARCHAR(50),
                                                         library_quality VARCHAR(50),
                                                         spectral_data_id_1 INT NOT NULL,
                                                         PRIMARY KEY(coumpound_id),
                                                         UNIQUE(spectral_data_id_1),
                                                         FOREIGN KEY(spectral_data_id_1) REFERENCES Spectral_data(spectral_data_id)
);

drop table if exists Composing ;
create table if not exists
    Composing(
                                                          experiment_id INT,
                                                          spectral_data_id INT,
                                                          PRIMARY KEY(experiment_id, spectral_data_id),
                                                          FOREIGN KEY(experiment_id) REFERENCES Experiment(experiment_id),
                                                          FOREIGN KEY(spectral_data_id) REFERENCES Spectral_data(spectral_data_id)
);

drop table if exists Identifying ;
create table if not exists
    Identifying(
                            experiment_id INT,
                            coumpound_id INT,
                            PRIMARY KEY(experiment_id, coumpound_id),
                            FOREIGN KEY(experiment_id) REFERENCES Experiment(experiment_id),
                            FOREIGN KEY(coumpound_id) REFERENCES Compound(coumpound_id)
);

drop table if exists Experimenting ;
create table if not exists
    Experimenting(
                              experiment_id INT,
                              analytics_data_id INT,
                              date_id INT,
                              PRIMARY KEY(experiment_id, analytics_data_id, date_id),
                              FOREIGN KEY(experiment_id) REFERENCES Experiment(experiment_id),
                              FOREIGN KEY(analytics_data_id, date_id) REFERENCES Analytics_data(analytics_data_id, date_id)
);

drop table if exists Building ;
create table if not exists
    Building(
                         building_block_id INT,
                         spectral_data_id INT,
                         PRIMARY KEY(building_block_id, spectral_data_id),
                         FOREIGN KEY(building_block_id) REFERENCES Building_Block(building_block_id),
                         FOREIGN KEY(spectral_data_id) REFERENCES Spectral_data(spectral_data_id)
);

drop table if exists Tooling ;
create table if not exists
    Tooling(
                        tool_id INT,
                        platform_user_id INT,
                        PRIMARY KEY(tool_id, platform_user_id),
                        FOREIGN KEY(tool_id) REFERENCES Tool(tool_id),
                        FOREIGN KEY(platform_user_id) REFERENCES Platform_User(platform_user_id)
);

drop table if exists Ionising ;
create table if not exists
    Ionising(
                         tool_id INT,
                         ionisation_mode_id INT,
                         PRIMARY KEY(tool_id, ionisation_mode_id),
                         FOREIGN KEY(tool_id) REFERENCES Tool(tool_id),
                         FOREIGN KEY(ionisation_mode_id) REFERENCES Ionisation_mode(ionisation_mode_id)
);

drop table if exists Analysing ;
create table if not exists
    Analysing(
                          tool_id INT,
                          analytics_data_id INT,
                          date_id INT,
                          PRIMARY KEY(tool_id, analytics_data_id, date_id),
                          FOREIGN KEY(tool_id) REFERENCES Tool(tool_id),
                          FOREIGN KEY(analytics_data_id, date_id) REFERENCES Analytics_data(analytics_data_id, date_id)
);

/* insertion in the main constant table list */

insert into IonModeChem(ionmodechem_id, chemical_composition) values (1, '[M]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (2, '[M]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (3, '[M+H]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (4, '[M+NH4]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (5, '[M+Na]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (6, '[M+K]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (7, '[M+CH3OH+H]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (8, '[M+ACN+H]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (9, '[M+ACN+Na]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (10, '[M+2ACN+H]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (11, 'Negative');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (12, '[M]-');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (13, '[M-H]-');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (14, '[M+Cl]-');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (15, '[M+HCOO]-');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (16, '[M+CH3COO]-');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (17, '[M-2H]-');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (18, '[M-2H+Na]-');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (19, '[M-2H+K]-');
