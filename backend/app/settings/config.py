import os
import configparser


cfg = configparser.ConfigParser()
cfg.read('./env-stag.ini')


#=========================================================================
#                          PROJECT INFORMATION 
#=========================================================================
PROJECT = cfg['project']
PROJECT_NAME = PROJECT['name']
ENVIRONMENT = PROJECT['environment']
HOST_NAME = PROJECT['host_name']


BACKEND = cfg['backend']
BE_PORT = BACKEND.getint('port')
FRONTEND = cfg['frontend']
FE_PORT = FRONTEND.getint('port')


#=========================================================================
#                          DATABASE INFORMATION 
#=========================================================================
DATABASE = cfg['database']
SQLALCHEMY_DATABASE_URL = "{type}://{user}:{pw}@{host}:{port}/{db_name}" \
    .format(
        type = DATABASE['type'],
        user = DATABASE['user'],
        pw = DATABASE['pass'],
        host = DATABASE['host'],
        port = DATABASE['port'],
        db_name = DATABASE['database'],
    )
DATABASE_SCHEMA = DATABASE['schema']


#=========================================================================
#                          AUTHENTICATE INFORMATION 
#=========================================================================
AUTHENTICATE = cfg['authenticate']
ENCODE_TYPE = AUTHENTICATE['encode']
DIGEST = AUTHENTICATE['digest']    
ALGORITHM = AUTHENTICATE['algorithm']
ROUNDS = AUTHENTICATE.getint('rounds')
SALT_SIZE = AUTHENTICATE.getint('salt_size')
SALT = bytes(AUTHENTICATE['salt'], "utf-8").decode('unicode_escape')
ACCESS_TOKEN_EXPIRE_MINUTES = AUTHENTICATE.getint('access_expire')
FRESH_TOKEN_EXPIRE_MINUTES = AUTHENTICATE.getint('fresh_expire')
SECRET_KEY = AUTHENTICATE['secret_key']


#=========================================================================
#                          NAS INFORMATION 
#=========================================================================
NAS = cfg['nas']
NAS_HOST = NAS['host']
NAS_USER = NAS['user']
NAS_PASSWORD = NAS['pass']
NAS_CERT_PATH = NAS['cert_path']
NAS_ROOT_PATH = NAS['root_path']


AIRFLOW_API_ENDPOINT = "https://dwh-staging-etl.digi-texx.vn/airflow/api/experimental"  


#=========================================================================
#                          LDAP INFORMATION 
#=========================================================================
LDAP = cfg['ldap']
LDAP_SERVER = LDAP['server']
LDAP_BASE = LDAP['base']
LDAP_ROOT_DN = LDAP['root_dn']
LDAP_ADMIN_USER = LDAP['admin_user']
LDAP_ADMIN_PASSWORD = LDAP['admin_password']