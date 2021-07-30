import fastapi
from ldap3 import Server, Connection, ALL, SUBTREE, ALL_ATTRIBUTES, NTLM
from ldap3.core.exceptions import LDAPSocketOpenError
from settings import config
 

def is_user_ldap(user_name, password):
    try:     
        ldap_server = Server(config.LDAP_SERVER, get_info=ALL, use_ssl=False)
        ldap_connection = Connection(ldap_server, user='cn={admin_user},{root_dn}'.format(root_dn=config.LDAP_ROOT_DN, admin_user=config.LDAP_ADMIN_USER), password=config.LDAP_ADMIN_PASSWORD, raise_exceptions=False)
        if ldap_connection.bind() == True:
            if ldap_connection.search(search_base=config.LDAP_BASE, search_filter=f'(&(objectclass=user)(sAMAccountName={user_name}))',search_scope = SUBTREE, attributes=ALL_ATTRIBUTES) == True:
                entry = ldap_connection.response[0]
                dn = entry['dn']  
                conn2 = Connection(ldap_server, user=dn, password=password, raise_exceptions=False)
                try:
                    if conn2.bind() == True:
                        return True
                    else:
                        return False
                except:
                    return False
            else:
                return False
        else:
            return False
    except LDAPSocketOpenError:
        return False
