CREATE ROLE frederic WITH
    LOGIN
    SUPERUSER
	CREATEDB
	CREATEROLE
	INHERIT
	REPLICATION
	CONNECTION LIMIT -1
	PASSWORD 'postgre23';
COMMENT ON ROLE frederic IS 'admin user';


drop database if exists "MolRefAnt_DB_PostGreSQL";

CREATE DATABASE "MolRefAnt_DB_PostGreSQL"
    WITH
    OWNER = frederic
    ENCODING = 'UTF8'
    CONNECTION LIMIT = -1
    IS_TEMPLATE = False;

COMMENT ON DATABASE "MolRefAnt_DB_PostGreSQL"
    IS 'Molecular referencing and annotation';

SECURITY LABEL FOR postgres ON DATABASE "MolRefAnt_DB_PostGreSQL" IS 'admin rights to MolRefAntDB_PostgreSQL';

GRANT ALL ON DATABASE "MolRefAnt_DB_PostGreSQL" TO frederic;

drop database "MolRefAnt_DB_PostGreSQL";
