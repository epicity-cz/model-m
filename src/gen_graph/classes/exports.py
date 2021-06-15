from constants import constants
from lang.mytypes import List
from program.loaders import exports_loader
from utils.has_id import HasId


class ExportNode(HasId):
    code: str
    name: str
    type: constants.ExportType

    def __init__(self, code, name):
        self.gen_id()
        self.code = code
        self.name = name
        self.type = constants.ExportType.PLACE


class Exports:
    nodes: List[ExportNode]

    def __init__(self, filename):
        self.nodes = []
        for (code, name) in exports_loader(filename):
            self.nodes.append(ExportNode(code, name))
