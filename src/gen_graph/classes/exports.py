from classes.person import Person
from utils.has_id import HasId


class ExportNode(HasId):
    name: str
    code: str
    _ID_CLASS = Person

    def __init__(self, name, code=''):
        self.gen_id()
        self.code = code
        self.name = name
