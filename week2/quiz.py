from collections import defaultdict

class NgramCounter:

    def __init__(self, n):
        self.n = n
        self.counts = defaultdict(int)

    def add(self, ngram):
        assert len(ngram) == self.n
        self.counts[ngram] += 1

    def get(self, ngram):
        return self.counts[ngram]

def ngramify(words, n):

    for j in range(n, len(words) + 1):
        i = j - (n)
        yield tuple(words[i:j])

def pad_words(words, n):
    return ['<s>'] * (n - 1) + words + ['<e>']

class NgramPTab:

    def __init__(self, n, smooth='laplacian'):
        assert n >= 2
        self.n = n
        self.n_count = NgramCounter(n)
        self.n_1_count = NgramCounter(n-1)
        self.V = 0
        self.smooth = smooth

    def train(self, words: list):

        words = pad_words(words, self.n)
        assert len(words) >= self.n
        for ngram in ngramify(words, self.n): self.n_count.add(ngram)
        for ngram in ngramify(words, self.n-1): self.n_1_count.add(ngram)
        self.V += len(set(words)) - 1

    def calcp(self, ngram):

        if self.smooth=='laplacian':
            return (self.n_count.get(ngram) + 1) / (self.n_1_count.get(ngram[:self.n-1]) + self.V)

        if self.n_count.get(ngram) == 0:
            # print(ngram)
            return 0.
        return self.n_count.get(ngram)/ (self.n_1_count.get(ngram[:self.n - 1]))

    def prob(self, words: list, perp=True):

        N = len(words)
        likelihood = 1.
        for ngram in ngramify(pad_words(words, n), n):
            print(ngram, self.calcp(ngram))
            likelihood *= self.calcp(ngram)

        if perp: return likelihood ** (-1/N)
        return likelihood

# calculating the perplexity of a test sentence with a single input sentence, using a 3-gram model
train_sentence = 'This is the rat that ate the malt that lay in the house that Jack built'.split()
test_sentence = 'This is the house that Jack built'.split()
n = 3

pmf = NgramPTab(n, smooth='laplacian')
pmf.train(train_sentence)
perp = pmf.prob(test_sentence, perp=True)
print(perp)

# calculating the probability of a test bigram with multiple input sentence, using a 3-gram model
n = 2
pmf = NgramPTab(n, smooth=None)

train_sentence = 'This is the house that Jack built'.split()
pmf.train(train_sentence)
train_sentence = 'This is the malt that lay in the house that Jack built'.split()
pmf.train(train_sentence)
train_sentence = 'This is the rat that ate the malt that lay in the house that Jack built'.split()
pmf.train(train_sentence)

print(pmf.calcp(('that','lay')))