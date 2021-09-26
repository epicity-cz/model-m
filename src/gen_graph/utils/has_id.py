class HasId:
    LAST_ID = -1
    _ID_CLASS = None
    id: int

    def gen_id(self):
        cls = self.__class__
        if cls._ID_CLASS:
            cls = cls._ID_CLASS
        self.id = cls.get_id()

    @classmethod
    def get_id(cls):
        cls.LAST_ID += 1
        return cls.LAST_ID
