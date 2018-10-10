#!/usr/bin/env python

NONE = "O"

ne_types = dict()
total_ne = 0
total_token = 0
with open('/home/emre/git/blstm-crf-ner/data/conll2003/en/test.txt', encoding='utf-8') as f:
    for c, line in enumerate(f):
        if not line or len(line.split()) == 0:
            continue
        total_token += 1
        tag = line.split()[-1]
        if tag == NONE:
            continue
        parts = str(tag).lower().split('-')
        if parts[0] == 'b' or parts[0] == 's':
            if parts[1] in ne_types:
                ne_types[parts[1]] = int(ne_types[parts[1]]) + 1
            else:
                ne_types[parts[1]] = 1
for ne_type, count in ne_types.items():
    total_ne += int(count)
    print("{} :  {}".format(ne_type, count))
print("Total NEs: {} tokens: {}".format(total_ne, total_token))
