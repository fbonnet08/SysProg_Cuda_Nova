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

drop table if exists employee_audits  ;
drop table if exists employees  ;
drop table if exists datetable  ;
drop table if exists fragment  ;
drop table if exists Measure  ;
drop table if exists composing  ;
drop table if exists identifying  ;
drop table if exists compound  ;
drop table if exists experimenting  ;
drop table if exists building  ;
drop table if exists buildingblocks  ;
drop table if exists spectral_data  ;
drop table if exists data  ;
drop table if exists DatabaseDetails  ;
drop table if exists level  ;
drop table if exists tooling  ;
drop table if exists Platform_User  ;
drop table if exists ionising  ;
drop table if exists experiment  ;
drop table if exists analysing  ;
drop table if exists tool  ;
drop table if exists analytics_data  ;
drop table if exists ionisation_mode  ;

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

drop table if exists Building_Block  ;
create table if not exists
    Building_Block(
    building_block_id INT,
    building_block_name VARCHAR(60),
    building_block_structure VARCHAR(100),
    PRIMARY KEY(building_block_id)
);

drop table if exists Tool  ;
create table if not exists
    Tool(
    tool_id INT,
    instrument_source VARCHAR(250),
    PRIMARY KEY(tool_id)
);

drop table if exists Platform_User  ;
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

drop table if exists Ionisation_mode  ;
create table if not exists
    Ionisation_mode(
    ionisation_mode_id INT,
    ionisation_mode VARCHAR(50),
    signum INT,
    PRIMARY KEY(ionisation_mode_id)
);

drop table if exists Analytics_data  ;
create table if not exists
    Analytics_data(
    analytics_data_id INT,
    sample_name TEXT,
    sample_details TEXT,
    sample_solvent TEXT,
    number_scans VARCHAR(25),
    filename VARCHAR(250),
    PRIMARY KEY(analytics_data_id)
);

drop table if exists Database_Details  ;
create table if not exists
    Database_Details(
    database_id INT,
    database_name VARCHAR(50),
    database_affiliation VARCHAR(50),
    database_path VARCHAR(100),
    library_quality_legend VARCHAR(250),
    PRIMARY KEY(database_id)
);

drop table if exists DateTable  ;
create table if not exists
    DateTable(
    date_id INT,
    date_column DATE,
    time_column TIME,
    timestamp_with_tz_column VARCHAR(50),
    analytics_data_id INT NOT NULL,
    PRIMARY KEY(date_id),
    UNIQUE(analytics_data_id),
    FOREIGN KEY(analytics_data_id) REFERENCES Analytics_data(analytics_data_id)
);

drop table if exists LoginTable  ;
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

drop table if exists ionmodechem  ;
create table if not exists
    ionmodechem(
    ionmodechem_id INT,
    chemical_composition VARCHAR(80),
    PRIMARY KEY(ionmodechem_id)
);

drop table if exists Charge  ;
create table if not exists
    Charge(
    charge_id INT,
    charge VARCHAR(10),
    PRIMARY KEY(charge_id)
);

drop table if exists User_Role  ;
create table if not exists
    User_Role(
    user_role_id INT,
    user_role VARCHAR(50),
    PRIMARY KEY(user_role_id)
);
/*
    platform_user_id INT NOT NULL,
    FOREIGN KEY(platform_user_id) REFERENCES Platform_User(platform_user_id)
 */

drop table if exists Experiment  ;
create table if not exists
    Experiment(
    experiment_id INT,
    scan_id INT,
    ionisation_mode_id INT NOT NULL,
    PRIMARY KEY(experiment_id),
    FOREIGN KEY(ionisation_mode_id) REFERENCES Ionisation_mode(ionisation_mode_id)
);
/*     UNIQUE(ionisation_mode_id), */

drop table if exists Data  ;
create table if not exists
    Data(
    data_id INT,
    path_to_data VARCHAR(250),
    raw_file VARCHAR(150),
    csv_file VARCHAR(150),
    xls_file VARCHAR(150),
    asc_file VARCHAR(150),
    mgf_file VARCHAR(150),
    m2s_file VARCHAR(150),
    json_file VARCHAR(150),
    experiment_id INT NOT NULL,
    PRIMARY KEY(data_id),
    FOREIGN KEY(experiment_id) REFERENCES Experiment(experiment_id)
);

drop table if exists Spectral_data  ;
create table if not exists
    Spectral_data(
    spectral_data_id INT,
    feature_id VARCHAR(100),
    pepmass DECIMAL(15,4),
    MSLevel VARCHAR(7),
    scan_number VARCHAR(50),
    retention_time TIME,
    mol_json_file VARCHAR(250),
    num_peaks INT,
    peaks_list TEXT,
    ionmodechem_id INT NOT NULL,
    charge_id INT NOT NULL,
    tool_id INT NOT NULL,
    database_id INT NOT NULL,
    data_id INT NOT NULL,
    PRIMARY KEY(spectral_data_id),
    FOREIGN KEY(ionmodechem_id) REFERENCES ionmodechem(ionmodechem_id),
    FOREIGN KEY(charge_id) REFERENCES Charge(charge_id),
    FOREIGN KEY(tool_id) REFERENCES Tool(tool_id),
    FOREIGN KEY(database_id) REFERENCES Database_Details(database_id),
    FOREIGN KEY(data_id) REFERENCES Data(data_id)
);

drop table if exists Measure  ;
create table if not exists
    Measure(
    measure_id INT,
    mz_value DECIMAL(25,12),
    relative DECIMAL(25,12),
    spectral_data_id INT NOT NULL,
    PRIMARY KEY(measure_id),
    FOREIGN KEY(spectral_data_id) REFERENCES Spectral_data(spectral_data_id)
);

drop table if exists Compound  ;
create table if not exists
    Compound(
    compound_id INT,
    compound_name TEXT,
    smiles TEXT,
    pubchem VARCHAR(250),
    molecular_formula VARCHAR(250),
    taxonomy TEXT,
    library_quality VARCHAR(250),
    spectral_data_id INT NOT NULL,
    PRIMARY KEY(compound_id),
    FOREIGN KEY(spectral_data_id) REFERENCES Spectral_data(spectral_data_id)
);

drop table if exists Composing  ;
create table if not exists
    Composing(
    experiment_id INT,
    spectral_data_id INT,
    PRIMARY KEY(experiment_id, spectral_data_id),
    FOREIGN KEY(experiment_id) REFERENCES Experiment(experiment_id),
    FOREIGN KEY(spectral_data_id) REFERENCES Spectral_data(spectral_data_id)
);

drop table if exists Identifying  ;
create table if not exists
    Identifying(
    experiment_id INT,
    compound_id INT,
    PRIMARY KEY(experiment_id, compound_id),
    FOREIGN KEY(experiment_id) REFERENCES Experiment(experiment_id),
    FOREIGN KEY(compound_id) REFERENCES Compound(compound_id)
);

drop table if exists Experimenting  ;
create table if not exists
    Experimenting(
    experiment_id INT,
    analytics_data_id INT,
    PRIMARY KEY(experiment_id, analytics_data_id),
    FOREIGN KEY(experiment_id) REFERENCES Experiment(experiment_id),
    FOREIGN KEY(analytics_data_id) REFERENCES Analytics_data(analytics_data_id)
);

drop table if exists Building  ;
create table if not exists
    Building(
    building_block_id INT,
    spectral_data_id INT,
    PRIMARY KEY(building_block_id, spectral_data_id),
    FOREIGN KEY(building_block_id) REFERENCES Building_Block(building_block_id),
    FOREIGN KEY(spectral_data_id) REFERENCES Spectral_data(spectral_data_id)
);

drop table if exists Tooling  ;
create table if not exists
    Tooling(
    tool_id INT,
    platform_user_id INT,
    PRIMARY KEY(tool_id, platform_user_id),
    FOREIGN KEY(tool_id) REFERENCES Tool(tool_id),
    FOREIGN KEY(platform_user_id) REFERENCES Platform_User(platform_user_id)
);

drop table if exists Ionising  ;
create table if not exists
    Ionising(
    tool_id INT,
    ionisation_mode_id INT,
    PRIMARY KEY(tool_id, ionisation_mode_id),
    FOREIGN KEY(tool_id) REFERENCES Tool(tool_id),
    FOREIGN KEY(ionisation_mode_id) REFERENCES Ionisation_mode(ionisation_mode_id)
);

drop table if exists Analysing  ;
create table if not exists
    Analysing(
    tool_id INT,
    analytics_data_id INT,
    PRIMARY KEY(tool_id, analytics_data_id),
    FOREIGN KEY(tool_id) REFERENCES Tool(tool_id),
    FOREIGN KEY(analytics_data_id) REFERENCES Analytics_data(analytics_data_id)
);

drop table if exists Interpreting  ;
create table if not exists
    Interpreting(
    spectral_data_id INT,
    user_role_id INT,
    platform_user_id int,
    PRIMARY KEY(spectral_data_id, user_role_id, platform_user_id),
    FOREIGN KEY(spectral_data_id) REFERENCES Spectral_data(spectral_data_id),
    FOREIGN KEY(user_role_id) REFERENCES User_Role(user_role_id),
    foreign key (platform_user_id) references platform_user(platform_user_id)
);

/* insertion in the main constant table list */

insert into IonModeChem(ionmodechem_id, chemical_composition) values (1, 'Positive');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (2, '[M]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (3, '[M]*+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (4, '[M+H]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (5, '[M+NH4]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (6, '[M+Na]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (7, '[M+K]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (8, '[M+CH3OH+H]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (9, '[M+ACN+H]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (10, '[M+ACN+Na]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (11, '[M+2ACN+H]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (12, '[M-H2O+H]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (13, '[frag]+');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (14, 'Negative');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (15, '[M]-');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (16, '[M-H]-');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (17, '[M+Cl]-');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (18, '[M+HCOO]-');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (19, '[M+CH3COO]-');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (20, '[M-2H]-');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (21, '[M-2H+Na]-');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (22, '[M-2H+K]-');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (23, '[M+HCOOH-H]-');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (24, 'Neutral');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (25, 'Unknown');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (26, 'N/A');
insert into IonModeChem(ionmodechem_id, chemical_composition) values (27, '');

insert into Charge(charge_id, charge) VALUES (1, '3-');
insert into Charge(charge_id, charge) VALUES (2, '2-');
insert into Charge(charge_id, charge) VALUES (3, '1-');
insert into Charge(charge_id, charge) VALUES (4, '0-');
insert into Charge(charge_id, charge) VALUES (5, '0');
insert into Charge(charge_id, charge) VALUES (6, '0+');
insert into Charge(charge_id, charge) VALUES (7, '1+');
insert into Charge(charge_id, charge) VALUES (8, '2+');
insert into Charge(charge_id, charge) VALUES (9, '3+');

insert into LoginTable(login_id, user_id, firstname, lastname, username, password, role, affiliation, department, researchlab)
values (1, '1000', 'Frederic', 'Bonnet', 'fbonnet', 'Fred1234!', 'admin', 'Banyuls', 'OBS', null);
insert into LoginTable(login_id, user_id, firstname, lastname, username, password, role, affiliation, department, researchlab)
values (2, '1001', 'Nicolas', 'Desmeuriaux', 'ndesmeuriaux', 'Nico1234!', 'admin', 'Banyuls', 'OBS', null);
insert into LoginTable(login_id, user_id, firstname, lastname, username, password, role, affiliation, department, researchlab)
values (3, '2000', 'Didier', 'Stein', 'dstein', 'Didier1234!', 'Principale Investigator', 'Banyuls', 'OBS', null);
insert into LoginTable(login_id, user_id, firstname, lastname, username, password, role, affiliation, department, researchlab)
values (4, '2001', 'Alice', 'Sanchez', 'asanchez', 'Alice1234!', 'Research Engineer', 'Banyuls', 'OBS', null);

insert into User_Role(user_role_id, user_role) VALUES (1, 'PI');
insert into User_Role(user_role_id, user_role) VALUES (2, 'DATACOLLECTOR');

insert into ionisation_mode(ionisation_mode_id, ionisation_mode, signum) VALUES (1, 'Negative', -1);
insert into ionisation_mode(ionisation_mode_id, ionisation_mode, signum) VALUES (2, 'Neutral', 0);
insert into ionisation_mode(ionisation_mode_id, ionisation_mode, signum) VALUES (3, 'Positive', 1);
insert into ionisation_mode(ionisation_mode_id, ionisation_mode, signum) VALUES (4, 'Unknown', 9);
insert into ionisation_mode(ionisation_mode_id, ionisation_mode, signum) VALUES (5, 'N/A', 10);




