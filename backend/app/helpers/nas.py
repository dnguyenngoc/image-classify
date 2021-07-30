from webdav3.client import Client
import io
import pickle
from settings import config


class NasWebDavConnect:
    def __init__(
        self, 
        *kwargs, 
        webdav_hostname: str = '',
        webdav_login: str = '',
        webdav_password: str ='',
        cert_path: str = '',
        key_path: str = '',
        lazy: bool = True
    ):
        self.webdav_hostname = webdav_hostname
        self.webdav_login = webdav_login
        self.webdav_password = webdav_password
        self.cert_path = cert_path
        self.key_path = key_path
        self.client = Client(self.default_options_dis_check())
    
    def default_options_dis_check(self):
        return {
            'webdav_hostname':  self.webdav_hostname,
            'webdav_login':    self.webdav_login,
            'webdav_password': self.webdav_password,
            'cert_path': self.cert_path,
            'key_path': self.key_path,
            'webdav_timeout': 300,
            'disable_check': True
        }

    def read_pickle_file(self, path):
        buffer = io.BytesIO()
        self.client.resource(path).write_to(buffer)
        data = pickle.loads(buffer.getvalue())
        return data

    def upload_file(self, data, path):
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        rs = self.client.resource(path)
        data = rs.read_from(buffer.getvalue())

    def is_existed(self, path):
        try:
            self.client.info(path)
            return True
        except:
            return False

    def create_path(self, path):
        self.client.mkdir(path)

    def get_info(self, path):
        return self.client.info(path)
    
    def remove_file(self, path):
        return self.client.clean(path)
        
    def check_path(self, path):
        return self.client.check(path)