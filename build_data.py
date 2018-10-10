import os
import subprocess
from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_word_vec_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_word_vectors, get_processing_word


def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant word2vec vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    # get config and processing of words

    config = Config(load=False)
    processing_word = get_processing_word(lowercase=False)

    # Generators
    dev   = CoNLLDataset(config.filename_dev, processing_word)
    test  = CoNLLDataset(config.filename_test, processing_word)
    train = CoNLLDataset(config.filename_train, processing_word)
    train2 = CoNLLDataset(config.filename_train2, processing_word)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test, train2])

    vocab = vocab_words
    if "w2v" in config.use_pretrained:
        vocab_word2vec = get_word_vec_vocab(config.filename_word2vec)
        vocab = vocab_words & vocab_word2vec if config.use_pretrained == "w2v" else vocab_words
    if config.replace_digits:
        vocab.add(NUM)
    vocab.add(UNK)

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # Trim FastText vectors
    if "ft" in config.use_pretrained:
        abs_f_words = os.path.abspath(config.filename_words)
        abs_f_vec = os.path.abspath(config.filename_fasttext)
        cmd = config.get_ft_vectors_cmd.format(abs_f_words, abs_f_vec)
        subprocess.check_call(cmd, shell=True)
        vocab = load_vocab(config.filename_words)
        export_trimmed_word_vectors(vocab, config.filename_fasttext, config.filename_trimmed_ft, config.dim_word)

    if "s2v" in config.use_pretrained:
        abs_s_words = os.path.abspath(config.filename_words)
        abs_s_vec = os.path.abspath(config.filename_fasttext)
        cmd = config.get_sent2vec_vectors_cmd.format(abs_s_words, abs_s_vec)
        subprocess.check_call(cmd, shell=True)
        vocab = load_vocab(config.filename_words)
        export_trimmed_word_vectors(vocab, config.filename_sent2vec, config.filename_trimmed_s2v, config.dim_sent)

    # Trim Morph2Vec vectors
    if "m2v" in config.use_pretrained:
        vocab = load_vocab(config.filename_words)
        export_trimmed_word_vectors(vocab, config.filename_morph2vec,
                                    config.filename_trimmed_m2v, config.dim_morph, partial_match=True)

    # Trim word2vec Vectors
    if "w2v" in config.use_pretrained:
        vocab = load_vocab(config.filename_words)
        export_trimmed_word_vectors(vocab, config.filename_word2vec,
                                    config.filename_trimmed_w2v, config.dim_word)

    # Build and save char vocab
    train = CoNLLDataset(config.filename_train)
    vocab_chars = get_char_vocab(train, config.use_ortho_char)
    write_vocab(vocab_chars, config.filename_chars)


if __name__ == "__main__":
    main()
