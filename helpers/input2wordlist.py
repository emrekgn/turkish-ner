word_list = set()

with open("../data/wnut17/emerging.dev.conll.preproc.url", "r") as inp1, open("../data/wnut17/emerging.test.conll.preproc.url", "r") as inp2, open("../data/wnut17/emerging.train.conll.preproc.url", "r") as inp3, open("input-en.txt", "w+") as out1:
    for line in inp1:
        if "#HANDLE#" in line or not line:
            continue
        if len(line.split()) == 0:
            continue
        word = str(line.strip().split()[0]).lower()
        out1.write("{}\n".format(word))
    for line in inp2:
        if "#HANDLE#" in line or not line:
            continue
        if len(line.split()) == 0:
            continue
        word = str(line.strip().split()[0]).lower()
        out1.write("{}\n".format(word))
    for line in inp3:
        if "#HANDLE#" in line or not line:
            continue
        if len(line.split()) == 0:
            continue
        word = str(line.strip().split()[0]).lower()
        out1.write("{}\n".format(word))


with open("input-en.txt", "r") as inp, open("words.txt", "w+") as out:
    for c, line in enumerate(inp):
        # Skip handles and empty lines
        if "#HANDLE#" in line or not line:
            continue
        if len(line.split()) == 0:
            continue
        word = str(line.strip().split()[0]).lower()
        if not word.isalnum():
            continue
        if word in word_list:
            continue
        if len(word) < 4:
            continue
        word_list.add(word)
        out.write("{}:{}+###+###+###+###+###+###+###+###+{}\n".format(word,
                                                                      word[:-3] + "-" + word[-3:],
                                                                      "-".join([char for char in word])))
