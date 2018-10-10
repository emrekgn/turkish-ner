#!/usr/bin/env python

import os
import optparse
import re
from nltk.tokenize import TweetTokenizer

# Read parameters from command line
optparser = optparse.OptionParser()
optparser.add_option(
    "-m", "--muc", default="/home/emre/git/blstm-crf-ner/helpers/WFS7.MUClabeled",
    help="Location of MUC labelled input file"
)
optparser.add_option(
    "-H", "--replace_handles", default="1",
    type='int', help="Replace user handles (default 1)"
)
optparser.add_option(
    "-U", "--replace_url", default="1",
    type='int', help="Replace URLs (default 1)"
)
optparser.add_option(
    "-R", "--reduce_len", default="1",
    type='int', help="Replace repeated character sequences of length 3 or greater with sequences of length 3 "
                     "(default 1)"
)
opts = optparser.parse_args()[0]

# Check parameters validity
assert os.path.isfile(opts.muc)

# Handles will be replaced with their annotations below
tokenizer = TweetTokenizer(strip_handles=False, reduce_len=opts.reduce_len == 1)

with open(opts.muc, "r") as muc, open("/home/emre/git/blstm-crf-ner/helpers/input.txt", "w+") as conll:
    numb_named_entity = 0
    numb_handle = 0
    numb_token = 0
    numb_person_token = 0
    numb_loc_token = 0
    numb_org_token = 0
    numb_date_token = 0
    numb_time_token = 0
    numb_percent_token = 0
    numb_money_token = 0

    for c, line in enumerate(muc):
        # Replace URLs
        if opts.replace_url:
            line = re.sub(r"http\S+", "#URL#", line)

        tokens = line.strip().split()
        numb_token += len(tokens)
        is_label = False
        iobes_tag = None
        prev_label_type = None

        for i, token in enumerate(tokens):
            if 'b_enamex' in token:
                is_label = True
                iobes_tag = None
                prev_label_type = None
                numb_named_entity += 1
                continue
            if is_label:
                if 'e_enamex' in token:
                    search_result = re.search(r'TYPE=\"(PERSON|ORGANIZATION|LOCATION|DATE|TIME|PERCENT|MONEY)\">(.*)<e_enamex>', token)
                    # Singleton label
                    if search_result:
                        groups = search_result.groups()
                        if groups and len(groups) == 2:
                            label_type = groups[0]
                            new_token = str(groups[1]).strip()
                            if label_type == 'PERSON':
                                numb_person_token += 1
                            elif label_type == 'ORGANIZATION':
                                numb_org_token += 1
                            elif label_type == 'LOCATION':
                                numb_loc_token += 1
                            elif label_type == 'DATE':
                                numb_date_token += 1
                            elif label_type == "TIME":
                                numb_time_token += 1
                            elif label_type == "PERCENT":
                                numb_percent_token += 1
                            elif label_type == "MONEY":
                                numb_money_token += 1
                            iobes_tag = "S"
                            if opts.replace_handles == 1 and str.startswith(new_token, '@'):
                                numb_handle += 1
                                new_token = "#HANDLE#"
                                conll.write(new_token + " O\n")
                            else:
                                conll.write("{0} {1}-{2}\n".format(new_token, iobes_tag, label_type))
                    else:
                        # End label
                        if prev_label_type:
                            search_result = re.search(r'(.*)<e_enamex>', token)
                            if search_result:
                                groups = search_result.groups()
                                if groups and len(groups) == 1:
                                    label_type = prev_label_type
                                    new_token = groups[0]
                                    iobes_tag = "E"
                                    conll.write("{0} {1}-{2}\n".format(new_token, iobes_tag, label_type))
                    is_label = False
                elif iobes_tag is None:
                    search_result = re.search(r'TYPE=\"(PERSON|ORGANIZATION|LOCATION|DATE|TIME|PERCENT|MONEY)\">(.*)', token)
                    # Begin label
                    if search_result:
                        groups = search_result.groups()
                        label_type = groups[0]
                        if label_type == 'PERSON':
                            numb_person_token += 1
                        elif label_type == 'ORGANIZATION':
                            numb_org_token += 1
                        elif label_type == 'LOCATION':
                            numb_loc_token += 1
                        elif label_type == 'DATE':
                            numb_date_token += 1
                        elif label_type == "TIME":
                            numb_time_token += 1
                        elif label_type == "PERCENT":
                            numb_percent_token += 1
                        elif label_type == "MONEY":
                            numb_money_token += 1
                        prev_label_type = label_type
                        new_token = groups[1]
                        iobes_tag = "B"
                        conll.write("{0} {1}-{2}\n".format(new_token, iobes_tag, label_type))
                # Inside label
                elif iobes_tag == "B" and prev_label_type:
                    iobes_tag = "I"
                    conll.write("{0} {1}-{2}\n".format(token, iobes_tag, prev_label_type))
            else:
                conll.write(token + " O\n")
        conll.write("\n")
    print("Number of tweets: {0}\nNumber of named entities: {1}\nNumber of handles: {2}\nNumber of tokens: {3}"
          "\nNumber of PERSON tokens: {4}\nNumber of LOC tokens: {5}\nNumber of ORG tokens: {6}"
          .format(c, numb_named_entity, numb_handle, numb_token, numb_person_token, numb_loc_token, numb_org_token))
