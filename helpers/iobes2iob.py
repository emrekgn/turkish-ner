
NONE = "O"

with open('/home/emre/git/blstm-crf-ner/data/celikkaya2013/input.txt', 'r') as inp, open('result-iob.txt', 'w+') as out:
    for c, line in enumerate(inp):
        if not line or len(line.split()) == 0:
            out.write('')
        word, gold_tag, guessed_tag = line.strip().split(None, 2)
        if gold_tag != NONE:
            iobes, entity = str(gold_tag).split('-', 1)
            if str(iobes).upper() == 'S':
                iobes = 'B'
            elif str(iobes).upper() == 'E':
                iobes = 'I'
            gold_tag = "{}-{}".format(str(iobes).upper(), str(entity).lower())
        if guessed_tag != NONE:
            iobes, entity = str(guessed_tag).split('-', 1)
            if str(iobes).upper() == 'S':
                iobes = 'B'
            elif str(iobes).upper() == 'E':
                iobes = 'I'
            guessed_tag = "{}-{}".format(str(iobes).upper(), str(entity).lower())
        out.write("{} {} {}".format(word, gold_tag, guessed_tag))
