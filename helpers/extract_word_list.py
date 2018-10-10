
count = 1
with open('/home/emre/data/embeddings.vec', encoding='utf-8') as f:
    with open('/home/emre/data/wordlist.txt', 'w+', encoding='utf-8') as f2:
        for line in f:
            if count == 200001:
                break
            word = line.strip().split(' ')[0]
            if count > 1:
                f2.write(word + "\n")
            count+=1
        print(count)
