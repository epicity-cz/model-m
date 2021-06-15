#!/usr/bin/env python3.9

# read all python source files from current directory (or prefix by first argument)
# generate puml diagram from 'classes' showing inheritance and attributes (from typing of class)

import glob
import re
import sys


def main():
    exclude_classes = ['bool', 'str', 'Any', 'int', 'float', 'List', 'EnumZ', 'Enum']
    cls_types = {}
    ext_types = {
        'EnumZ': 'enum',
        'Enum': 'enum',
    }

    r_ignore = re.compile(r'# @puml ignore')
    r_class = re.compile(r'class (\w*)(\((.*)\))?:')
    r_def = re.compile(r'\s*def\s')
    r_attr = re.compile(r'^\s*(\w*):\s*((.*?)(#.*)?)$')
    r_const = re.compile(r'^\s*([A-Z_]*)\s*=\s*(\d*|auto\(\))$')
    r_links = [re.compile(item) for item in [
        r'Dict\[\w*,\s*(\w*)]',
        r'List\[(\w*)]',
        r'@puml link (\w*)',
        r'^([A-Za-z]*)$'
    ]]
    cls = None
    connections = []
    ignore = False

    def end_class():
        nonlocal cls, connections
        if cls:
            print("}")
            for ln in connections:
                print(ln)
            print()
            connections = []
            cls = None

    print("@startuml")
    print("hide empty method")

    if len(sys.argv) > 1:
        prefix = sys.argv[1]
    else:
        prefix = '.'
    # root_dir needs a trailing slash (i.e. /root/dir/)
    for filename in glob.iglob(f'{prefix}/**/*.py', recursive=True):
        with open(filename, "r") as file:
            for line in file.readlines():
                if r_ignore.match(line):
                    ignore = True
                match = r_class.match(line)
                if match:
                    end_class()
                    if ignore:
                        ignore = False
                    else:
                        cls = match[1]
                        if match[3] and "[" not in match[3] and match[3] not in exclude_classes:
                            ext = f" extends {match[3]}"
                        else:
                            ext = ''
                        if cls not in exclude_classes:
                            if cls in cls_types:
                                tp = cls_types[cls]
                            elif match[3] in ext_types:
                                tp = ext_types[match[3]]
                            else:
                                tp = 'Class'
                            print(f"{tp} {cls}{ext} {{")
                        else:
                            cls = None

                elif cls:
                    if r_def.match(line):
                        end_class()
                    else:
                        ma = r_attr.match(line)
                        if ma:
                            print(f"\t{ma[1]}:\t\t{ma[3]}")
                            for r_link in r_links:
                                for mm in r_link.finditer(ma[2]):
                                    if mm[1] not in exclude_classes:
                                        connections.append(f"{cls}::{ma[1]} -> {mm[1]}")

                        ma = r_const.match(line)
                        if ma:
                            num = ma[2]
                            if num == 'auto()':
                                num = '*'
                            print(f"\t{num}\t{ma[1]}")

        end_class()
    print("@enduml")


if __name__ == '__main__':
    main()
