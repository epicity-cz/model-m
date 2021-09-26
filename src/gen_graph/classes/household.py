from classes.person import Person
from lang.mytypes import List, Any


class Household:
    persons: List[Person]
    apartment: Any

    def __init__(self):
        self.persons = []

    def add_person(self, person: Person):
        self.persons.append(person)
        person.household = self

    def age_average(self):
        return sum([p.age for p in self.persons]) / len(self.persons)
