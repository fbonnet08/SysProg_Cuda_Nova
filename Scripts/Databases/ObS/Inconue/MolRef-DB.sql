/* ----------------------------------
-- TP-2   Gendarme et Voleur exercise Relational Database
----------------------------------
-- Section 1
----------------------------------
-- First cleaning up the previous implementations for fresh start
*/

show databases ;

drop database if exists MolRefAnt_DB;
create database if not exists MolRefAnt_DB;

drop table if exists MolRefAnt_DB.BuildingBlocks;
create table if not exists MolRefAnt_DB.BuildingBlocks(
    bloc_id INT,
    bloc_name VARCHAR(50),
    bloc_structure VARCHAR(60),
    constraint PRIMARY KEY(bloc_id)
);

drop table if exists MolRefAnt_DB.Sample;
CREATE TABLE if not exists MolRefAnt_DB.Sample(
    sample_id INT,
    sampling_date DATETIME,
    sample_name VARCHAR(50),
    sample_solvent VARCHAR(50),
    sample_details VARCHAR(100),
    constraint PRIMARY KEY(sample_id)
);

drop table if exists MolRefAnt_DB.Compound;
CREATE TABLE if not exists MolRefAnt_DB.Compound(
    coumpound_id INT,
    compound_name VARCHAR(50),
    smiles VARCHAR(50),
    constraint PRIMARY KEY(coumpound_id)
);

drop table if exists MolRefAnt_DB.Tools;
CREATE TABLE if not exists MolRefAnt_DB.Tools(
    tool_id INT,
    collector VARCHAR(50),
    PI VARCHAR(50),
    instrument_source VARCHAR(50),
    MSLevel VARCHAR(50),
    library_quality VARCHAR(50),
    constraint PRIMARY KEY(tool_id)
);

drop table if exists MolRefAnt_DB.Data;
CREATE TABLE if not exists MolRefAnt_DB.Data(
    data_id INT,
    path_to_data VARCHAR(150),
    raw_file VARCHAR(60),
    csv_file VARCHAR(60),
    xls_file VARCHAR(60),
    asc_file VARCHAR(60),
    sample_id INT NOT NULL,
    constraint primary key(data_id),
    constraint foreign key(sample_id) references MolRefAnt_DB.Sample(sample_id)
);

drop table if exists MolRefAnt_DB.Fragmentation;
CREATE TABLE if not exists MolRefAnt_DB.Fragmentation(
    fragment_id INT,
    fragmewnt_name VARCHAR(50),
    pepmass DECIMAL(15,2),
    charge DECIMAL(15,2),
    mz_value VARCHAR(50),
    rention_time TIME,
    ion_mode VARCHAR(50),
    data_id INT NOT NULL,
    constraint PRIMARY KEY(fragment_id),
    constraint FOREIGN KEY(data_id) REFERENCES MolRefAnt_DB.Data(data_id)
);

drop table if exists MolRefAnt_DB.composer;
CREATE TABLE if not exists MolRefAnt_DB.composer(
    sample_id INT,
    fragment_id INT,
    PRIMARY KEY(sample_id, fragment_id),
    constraint FOREIGN KEY(sample_id) REFERENCES MolRefAnt_DB.Sample(sample_id),
    constraint FOREIGN KEY(fragment_id) REFERENCES MolRefAnt_DB.Fragmentation(fragment_id)
);

drop table if exists MolRefAnt_DB.identifier;
CREATE TABLE if not exists MolRefAnt_DB.identifier(
    sample_id INT,
    coumpound_id INT,
    constraint PRIMARY KEY(sample_id, coumpound_id),
    constraint FOREIGN KEY(sample_id) REFERENCES MolRefAnt_DB.Sample(sample_id),
    constraint FOREIGN KEY(coumpound_id) REFERENCES MolRefAnt_DB.Compound(coumpound_id)
);

drop table if exists MolRefAnt_DB.sampling;
CREATE TABLE if not exists MolRefAnt_DB.sampling(
    sample_id INT,
    tool_id INT,
    constraint PRIMARY KEY(sample_id, tool_id),
    constraint FOREIGN KEY(sample_id) REFERENCES MolRefAnt_DB.Sample(sample_id),
    constraint FOREIGN KEY(tool_id) REFERENCES MolRefAnt_DB.Tools(tool_id)
);

drop table if exists MolRefAnt_DB.build;
CREATE TABLE if not exists MolRefAnt_DB.build(
    bloc_id INT,
    fragment_id INT,
    constraint PRIMARY KEY(bloc_id, fragment_id),
    constraint FOREIGN KEY(bloc_id) REFERENCES MolRefAnt_DB.BuildingBlocks(bloc_id),
    constraint FOREIGN KEY(fragment_id) REFERENCES MolRefAnt_DB.Fragmentation(fragment_id)
);
