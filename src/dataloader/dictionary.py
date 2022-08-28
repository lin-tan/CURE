import codecs
import torch


class Dictionary():
    def __init__(self, vocab_file, min_cnt=0):
        self.vocab_file = vocab_file
        self.pad_word = '<PAD>'
        self.unk_word = '<UNK>'
        self.eos_word = '<EOS>'
        self.ctx_word = '<CTX>'
        self.dictionary = {}
        self.symbols = []
        self.counts = []
        self.min_cnt = min_cnt

        self.pad_index = self.add_symbol(self.pad_word)
        self.eos_index = self.add_symbol(self.eos_word)
        self.unk_index = self.add_symbol(self.unk_word)
        self.ctx_index = self.add_symbol(self.ctx_word)
        self.read_dictionary()

    def add_symbol(self, symbol, n=1):
        if symbol in self.dictionary:
            return self.dictionary[symbol]
        idx = len(self.dictionary)
        self.dictionary[symbol] = idx
        self.symbols.append(symbol)
        self.counts.append(n)
        return idx

    def read_dictionary(self):
        fp = codecs.open(self.vocab_file, 'r', 'utf-8')
        for l in fp.readlines():
            l = l.strip()
            if len(l.split()) != 2:
                continue
            symbol, count = l.split()
            if int(count) < self.min_cnt:
                continue
            self.add_symbol(symbol, int(count))

    def __getitem__(self, item):
        if type(item) != int:
            return self.symbols[int(item)] if int(item) < len(self.symbols) else self.unk
        return self.symbols[item] if item < len(self.symbols) else self.unk

    def __len__(self):
        return len(self.dictionary)

    def index(self, symbol):
        if type(symbol) == list:
            return [self.index(s) for s in symbol]
        if symbol in self.dictionary:
            return self.dictionary[symbol]
        return self.unk_index

    def string(self, tensor, bpe_symbol=None, show_pad=False):
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t) for t in tensor).split('\n')

        hide = [self.eos(), self.pad()] if not show_pad else [self.eos()]

        sent = ' '.join(self[i] for i in tensor if i not in hide)
        if bpe_symbol is not None:
            sent = (sent + ' ').replace(bpe_symbol + ' ', '').rstrip()
        return sent

    def pad(self):
        return self.pad_index

    def eos(self):
        return self.eos_index

    def unk(self):
        return self.unk_index

    def ctx(self):
        return self.ctx_index


if __name__ == "__main__":
    voc = Dictionary('../../../data/vocabulary/vocabulary.txt')
    print(len(voc))
