import pickle


class Persistent:

    @classmethod
    def dump_attributes(cls):
        return []

    def get_dump_var(self):
        return [getattr(self, item) for item in self.dump_attributes()]

    def set_dump_var(self, data):
        for (dta, item) in zip(data, self.dump_attributes()):
            setattr(self, item, dta)

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.get_dump_var(), f)

    def load(self, file):
        try:
            with open(file, 'rb') as f:
                self.set_dump_var(pickle.load(f))
            return True
        except FileNotFoundError:
            return False
