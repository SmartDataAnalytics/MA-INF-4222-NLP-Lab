#!/usr/bin/env python3
from langdetect import detect
import fileinput as fi
import sys

# just a little script to detect languages of lines starting 
# with the json keyword 'text' and writing the language as json value to stdout.
# other lines are just passing through so that this script can be used in a shell pipeline

for line in fi.input():
    if line.startswith('  "text"'):
        try:
            sys.stdout.write('  "lang": "' + detect(line[10:]) + '"\n')
        except Exception:
            sys.stdout.write('  "lang": "NaN"\n')
    sys.stdout.write(line)


