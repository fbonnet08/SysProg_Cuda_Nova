drop database if exists "MolRefAnt_DB_PostGreSQL";

GRANT ALL ON DATABASE "MolRefAnt_DB_PostGreSQL" TO frederic;

SECURITY LABEL FOR postgres ON DATABASE "MolRefAnt_DB_PostGreSQL" IS 'admin rights to MolRefAntDB_PostgreSQL';

drop database if exists "MolRefAnt_DB_PostGreSQL";


create schema  MolRefAnt_DB;

drop table if exists "MolRefAnt_DB_PostGreSQL".molrefant_db.employees cascade ;
create table if not exists "MolRefAnt_DB_PostGreSQL".molrefant_db.employees (
                                                                                employee_id integer primary key,
                                                                                name varchar(50) not null
);

drop table if exists "MolRefAnt_DB_PostGreSQL".molrefant_db.employee_audits cascade;
create table if not exists "MolRefAnt_DB_PostGreSQL".molrefant_db.employee_audits (
                                                                                      employee_id integer primary key references "MolRefAnt_DB_PostGreSQL".molrefant_db.employees(employee_id),
                                                                                      old_name varchar(50) not null
);


create schema MolRefAnt_DB_PostGreSQL.MolRefAnt_DB;


create schema MolRefAnt_DB;

drop schema MolRefAnt_DB;


create schema MolRefAnt_DB;

comment on schema MolRefAnt_DB is 'schema for MolRefAnt_DB_PostGreSQL';

alter schema MolRefAnt_DB owner to frederic;

grant create, usage on schema MolRefAnt_DB to postgres with grant option;







