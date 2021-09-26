import configparser


class Config:
    BASE_DIR: str

    def __init__(self):
        self.BASE_DIR = '.'

    def cast(self, value, tp):
        if tp is bool:
            return value.upper() in ['1', 'T', 'TRUE', 'Y', 'YES']
        return tp(value)

    def set_key(self, key: str, value: str):
        key = key.upper()
        if not hasattr(self, key):
            raise ValueError(f"Unknown config key '{key}")
        nt = type(getattr(self, key))
        try:
            setattr(self, key, self.cast(value, nt))
        except:
            raise ValueError(f"Cannot interpret value {value} as {nt}")

    def read_ini(self, inifile):
        cf = configparser.ConfigParser()
        cf.read_file(inifile)
        graph_mapping = {
            'NODES': 'FILE_NODES',
            'EXTERNALS': 'FILE_EXPORT',
            'EDGES': 'FILE_EDGES',
            'LAYERS': 'FILE_LAYERS',
            'NUMS': 'FILE_NUMS',
            'OBJECTS': 'FILE_OBJECTS',
        }
        for (key, value) in cf.items('GRAPH'):
            key = key.upper()
            if key in graph_mapping:
                nk = graph_mapping[key]
                self.set_key(nk, value)
        for (key, value) in cf.items('GRAPH_GEN'):
            self.set_key(key, value)

    @classmethod
    def create_from_inifile(cls, inifile, param_set=None):
        cfg = cls()

        if param_set:
            if isinstance(param_set, dict):
                params = param_set.items()
            else:
                params = param_set
            for (key, value) in params:
                cfg.set_key(key, value)

        if isinstance(inifile, str):
            with open(cfg.get_ini_filename(inifile), 'r') as f:
                cfg.read_ini(f)
        else:
            cfg.read_ini(inifile)
        return cfg

    def get_ini_filename(self, ini):
        return f'{self.BASE_DIR}/config/{ini}.ini'
