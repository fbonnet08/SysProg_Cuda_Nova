/*
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (1, '20-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (2, '19-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (3, '18-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (4, '17-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (5, '16-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (6, '15-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (7, '14-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (8, '13-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (9, '12-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (10, '11-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (11, '10-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (12, '9-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (13, '8-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (14, '7-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (15, '6-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (16, '5-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (17, '4-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (18, '3-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (19, '2-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (20, '1-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (21, '0-');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (22, '0');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (23, '0+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (24, '1+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (25, '2+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (26, '3+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (27, '4+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (28, '5+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (29, '6+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (30, '7+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (31, '8+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (32, '9+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (33, '10+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (34, '11+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (35, '12+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (36, '13+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (37, '14+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (38, '15+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (39, '16+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (40, '17+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (41, '18+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (42, '19+');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (43, '20+');
*/
/*
TODO: implement the loop insert for the charge_id, need to include the insert statement
                    concat()
                    v_charge_string := concat   '%-',abs(i);
                    insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(charge_id, charge) VALUES (v_charge_id, '%-',abs(i));

*/



alter table "MolRefAnt_DB".spectral_data
    alter column mslevel type VARCHAR(7) using mslevel::VARCHAR(7);

alter table "MolRefAnt_DB".spectral_data
    alter column mslevel drop not null;




do
    $$
declare
    v_charge_id integer := 0;
    v_counter integer := 0;
begin
    for i in -20..20
    loop
        v_charge_id := v_charge_id + 1;
        if i <= 0 then
            raise notice 'charge_id % , Current number %-',v_charge_id,abs(i);
        elsif i = 0 then
            raise notice 'charge_id % , Current number %',v_charge_id,abs(i);
        elsif i >= 0 then
            raise notice 'charge_id % , Current number %+',v_charge_id,abs(i);
        end if;
    end loop;
end
$$;



do
    $$
declare
    v_rec RECORD;
    v_tool_id integer := 1;
begin
    for v_rec in select * from "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".ionisation_mode
    loop
        /* raise notice '% %', v_tool_id, v_rec.ionisation_mode_id; */
        insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".ionising(tool_id, ionisation_mode_id)
        VALUES (1, v_rec.ionisation_mode_id);
    end loop;
end
$$;

insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".User_Role(user_role_id, user_role) VALUES (4, 'Student');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".User_Role(user_role_id, user_role) VALUES (2, 'Research Engineer');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".User_Role(user_role_id, user_role) VALUES (5, 'Postdoc');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".User_Role(user_role_id, user_role) VALUES (6, 'Graduate Student');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".User_Role(user_role_id, user_role) VALUES (7, 'Thermofisher');
insert into "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".User_Role(user_role_id, user_role) VALUES (8, 'admin');

INSERT INTO "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Analytics_data (
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

INSERT INTO "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".DateTable (date_id,
                                                                date_column, time_column, timestamp_with_tz_column, analytics_data_id,
                                                                date_id_1)
VALUES
    (1, '2024-04-19', '13:30:00', '2024-04-19 13:30:00', 1, 1);

/* insertion in the main constant table list */

drop table if exists building_block cascade;
drop table if exists charge cascade;
drop table if exists database_details cascade;
drop table if exists employee cascade;
drop table if exists experiment cascade;
drop table if exists ionisation_mode cascade;
drop table if exists ionmodechem cascade;
drop table if exists tool cascade;

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
drop table if exists buildingblock cascade ;
drop table if exists spectral_data cascade ;
drop table if exists data cascade ;
drop table if exists DatabaseDetails cascade ;
drop table if exists tooling cascade ;
drop table if exists Platform_User cascade ;
drop table if exists ionising cascade ;
drop table if exists ionisation_mode cascade ;
drop table if exists experiment cascade ;
drop table if exists analysing cascade ;
drop table if exists tool cascade ;
drop table if exists analytics_data cascade ;
drop table if exists ionmodechem cascade;
drop table if exists charge cascade;


drop table if exists "MolRefAnt_DB".users cascade;
drop table if exists "MolRefAnt_DB".employee_audits cascade;
drop table if exists "MolRefAnt_DB".employees cascade;
drop table if exists "MolRefAnt_DB".ionmodechem cascade;
drop table if exists "MolRefAnt_DB".datetable cascade;
drop table if exists "MolRefAnt_DB".logintable cascade;
drop table if exists "MolRefAnt_DB".fragment cascade;
drop table if exists "MolRefAnt_DB".composing cascade;
drop table if exists "MolRefAnt_DB".identifying cascade;
drop table if exists "MolRefAnt_DB".compound cascade;
drop table if exists "MolRefAnt_DB".experimenting cascade;
drop table if exists "MolRefAnt_DB".building cascade;
drop table if exists "MolRefAnt_DB".buildingblocks cascade;
drop table if exists "MolRefAnt_DB".spectral_data cascade;
drop table if exists "MolRefAnt_DB".data cascade;
drop table if exists "MolRefAnt_DB".databasedetails cascade;
drop table if exists "MolRefAnt_DB".tooling cascade;
drop table if exists "MolRefAnt_DB".platformuser cascade;
drop table if exists "MolRefAnt_DB".ionising cascade;
drop table if exists "MolRefAnt_DB".ionisation_mode cascade;
drop table if exists "MolRefAnt_DB".experiment cascade;
drop table if exists "MolRefAnt_DB".analysing cascade;
drop table if exists "MolRefAnt_DB".tool cascade;
drop table if exists "MolRefAnt_DB".analytics_data cascade;

drop table if exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge;
create table if not exists "MolRefAnt_DB_PostGreSQL"."MolRefAnt_DB".Charge(
                                                                              charge_id serial,
                                                                              charge varchar(10),
                                                                              primary key (charge_id)
);

drop table if exists build;
drop table buildingblocks;
drop table composer;
drop table fragmentation;
drop table data;
drop table identifier;
drop table compound;
drop table sampling;
drop table sample;
drop table tools;


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







