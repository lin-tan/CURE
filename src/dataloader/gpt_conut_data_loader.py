import codecs
from gpt_conut_dataset import GPTCoNuTDataset


class GPTCoNuTDataLoader():
    def __init__(self, datafile, dictionary, identifier_loader=None):
        self.datafile = datafile
        self.dictionary = dictionary
        self.total_size = 0
        self.get_total_size()

        self.src = []
        self.tgt = []

        self.identifier_loader = identifier_loader
        self.dataset = None

    def get_total_size(self):
        fp = codecs.open(self.datafile, 'r', 'utf-8')
        self.total_size = len(fp.readlines())
        fp.close()

    def reinitialize(self):
        self.src = []
        self.tgt = []
        self.dataset = None

    def load_data(self, start, end):
        self.reinitialize()
        fp = codecs.open(self.datafile, 'r', 'utf-8')
        cnt = -1
        while True:
            line = fp.readline()
            if not line:
                break
            if line.strip() == '':
                continue
            cnt += 1
            if cnt < start:
                continue
            if cnt >= end:
                break
            src, tgt = line.split('\t')
            src = src.strip().split()
            tgt = tgt.strip().split()

            src_tokens = self.dictionary.index(src)
            tgt_tokens = self.dictionary.index(tgt)
            src_tokens = src_tokens + [self.dictionary.eos()]
            tgt_tokens = tgt_tokens + [self.dictionary.eos()]
            self.src.append(src_tokens)
            self.tgt.append(tgt_tokens)
        if self.identifier_loader is not None:
            self.identifier_loader.load_data(start, end)
            self.dataset = GPTCoNuTDataset(self.src, self.tgt, self.dictionary,
                                           identifier=self.identifier_loader.identifier_list)
        else:
            self.dataset = GPTCoNuTDataset(self.src, self.tgt, self.dictionary)
