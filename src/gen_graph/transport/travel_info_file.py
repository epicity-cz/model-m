from program import loaders
from transport.travel_info import TravelInfo
from utils.log import StdLog


class TravelInfoFile(TravelInfo):
    def generate_connections(self):
        towns_by_zuj = {town.zuj: town for town in self.towns}
        for (frm, to, time) in loaders.transport_loader(self.config.transport()):
            frm_t = towns_by_zuj.get(frm)
            to_t = towns_by_zuj.get(to)
            if frm_t and to_t:
                yield (frm_t, to_t, time)
            else:
                StdLog.log(f"Ignoring transport OOR {frm}->{to}", StdLog.DEBUG)
