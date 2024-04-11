/* ----------------------------------
-- TP-2   Gendarme et Voleur exercise Relational Database
----------------------------------
-- Section 1
----------------------------------
-- First cleaning up the previous implementations for fresh start
*/

show databases ;

drop database if exists annoter_l_inconnu;
create database if not exists annoter_l_inconnu;

drop table if exists annoter_l_inconnu.BuildingBlocks;
create table if not exists annoter_l_inconnu.BuildingBlocks(
    bloc_id INT,
    bloc_name VARCHAR(50),
    bloc_structure VARCHAR(60),
    constraint PRIMARY KEY(bloc_id)
);

drop table if exists annoter_l_inconnu.Sample;
CREATE TABLE if not exists annoter_l_inconnu.Sample(
    sample_id INT,
    sampling_date DATETIME,
    sample_name VARCHAR(50),
    sample_solvent VARCHAR(50),
    sample_details VARCHAR(100),
    constraint PRIMARY KEY(sample_id)
);

drop table if exists annoter_l_inconnu.Compound;
CREATE TABLE if not exists Compound(
    coumpound_id INT,
    compound_name VARCHAR(50),
    smiles VARCHAR(50),
    constraint PRIMARY KEY(coumpound_id)
);

drop table if exists annoter_l_inconnu.Tools;
CREATE TABLE if not exists annoter_l_inconnu.Tools(
    tool_id INT,
    collector VARCHAR(50),
    PI VARCHAR(50),
    instrument_source VARCHAR(50),
    MSLevel VARCHAR(50),
    library_quality VARCHAR(50),
    constraint PRIMARY KEY(tool_id)
);

drop table if exists annoter_l_inconnu.Data;
CREATE TABLE if not exists annoter_l_inconnu.Data(
    data_id INT,
    path_to_data VARCHAR(150),
    raw_file VARCHAR(60),
    csv_file VARCHAR(60),
    xls_file VARCHAR(60),
    asc_file VARCHAR(60),
    sample_id INT NOT NULL,
    constraint primary key(data_id),
    constraint foreign key(sample_id) references annoter_l_inconnu.Sample(sample_id)
);

drop table if exists annoter_l_inconnu.Fragmentation;
CREATE TABLE if not exists annoter_l_inconnu.Fragmentation(
    fragment_id INT,
    fragmewnt_name VARCHAR(50),
    pepmass DECIMAL(15,2),
    charge DECIMAL(15,2),
    mz_value VARCHAR(50),
    rention_time TIME,
    ion_mode VARCHAR(50),
    data_id INT NOT NULL,
    constraint PRIMARY KEY(fragment_id),
    constraint FOREIGN KEY(data_id) REFERENCES annoter_l_inconnu.Data(data_id)
);

drop table if exists annoter_l_inconnu.composer;
CREATE TABLE if not exists annoter_l_inconnu.composer(
    sample_id INT,
    fragment_id INT,
    PRIMARY KEY(sample_id, fragment_id),
    constraint FOREIGN KEY(sample_id) REFERENCES annoter_l_inconnu.Sample(sample_id),
    constraint FOREIGN KEY(fragment_id) REFERENCES annoter_l_inconnu.Fragmentation(fragment_id)
);

drop table if exists annoter_l_inconnu.identifier;
CREATE TABLE if not exists annoter_l_inconnu.identifier(
    sample_id INT,
    coumpound_id INT,
    constraint PRIMARY KEY(sample_id, coumpound_id),
    constraint FOREIGN KEY(sample_id) REFERENCES annoter_l_inconnu.Sample(sample_id),
    constraint FOREIGN KEY(coumpound_id) REFERENCES annoter_l_inconnu.Compound(coumpound_id)
);

drop table if exists annoter_l_inconnu.sampling;
CREATE TABLE if not exists annoter_l_inconnu.sampling(
    sample_id INT,
    tool_id INT,
    constraint PRIMARY KEY(sample_id, tool_id),
    constraint FOREIGN KEY(sample_id) REFERENCES annoter_l_inconnu.Sample(sample_id),
    constraint FOREIGN KEY(tool_id) REFERENCES annoter_l_inconnu.Tools(tool_id)
);

drop table if exists annoter_l_inconnu.build;
CREATE TABLE if not exists annoter_l_inconnu.build(
    bloc_id INT,
    fragment_id INT,
    constraint PRIMARY KEY(bloc_id, fragment_id),
    constraint FOREIGN KEY(bloc_id) REFERENCES annoter_l_inconnu.BuildingBlocks(bloc_id),
    constraint FOREIGN KEY(fragment_id) REFERENCES annoter_l_inconnu.Fragmentation(fragment_id)
);
