from lang.mytypes import Type
from transport.travel_info import TravelInfo
from transport.travel_info_delaunay import TravelInfoDelaunay, TravelInfoDelaunaySparse
from transport.travel_info_file import TravelInfoFile
from transport.travel_info_tree import TravelInfoTree


def travel_info_by_name(name) -> Type[TravelInfo]:
    all_ti = [TravelInfoDelaunay, TravelInfoDelaunaySparse, TravelInfoFile, TravelInfoTree]
    return {item.__name__: item for item in all_ti}[name]
