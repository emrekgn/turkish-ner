#!/usr/bin/env python

import optparse
import pickle

# Read parameters from command line
optparser = optparse.OptionParser()
optparser.add_option(
    "-i", "--inp", default="vectors-en.p",
    help="Path to input JSON file"
)
optparser.add_option(
    "-o", "--out", default="en-embeddings-m2v.txt",
    help="Path to output text file"
)
opts = optparser.parse_args()[0]

with open(opts.inp, 'rb') as inp, open(opts.out, 'w+') as out:
    d = pickle.load(inp)
    for key, value in d.items():
        out.write("{} {}\n".format(key.encode('utf-8'), " ".join(map(str, value))))
