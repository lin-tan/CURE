import codecs


class IdentifierDataLoader():
    def __init__(self, dictionary, token_file, text_file):
        self.dictionary = dictionary
        self.token_file = token_file
        self.text_file = text_file
        self.identifier_list = []

    def load_data(self, start, end):
        self.identifier_list = []

        def get_prefix(tokens, dictionary):
            prefix = ''
            for token in tokens:
                prefix += dictionary[token]
            return prefix

        fp = codecs.open(self.token_file, 'r', 'utf-8')
        cnt = -1
        while True:
            line = fp.readline()
            if not line:
                break
            cnt += 1
            if cnt < start:
                continue
            if cnt >= end:
                break

            line = line.split('\t')
            identifiers = {
                'id': cnt,
                'tokens': {'': [
                    self.dictionary.index('$NUMBER$'),
                    self.dictionary.index('$STRING$'),
                    self.dictionary.index('_')]},
                'text': []
            }

            for identifier in line:
                token = identifier.strip().split()
                token = [self.dictionary.index(t) for t in token]
                if len(token) == 1:
                    if token[0] not in identifiers['tokens']['']:
                        identifiers['tokens'][''].append(token[0])
                else:
                    for i in range(len(token)):
                        prefix = get_prefix(token[:i], self.dictionary)
                        if prefix not in identifiers['tokens']:
                            identifiers['tokens'][prefix] = []
                        if token[i] not in identifiers['tokens'][prefix]:
                            identifiers['tokens'][prefix].append(token[i])
            self.identifier_list.append(identifiers)

        fp = codecs.open(self.text_file, 'r', 'utf-8')
        cnt = -1
        while True:
            line = fp.readline()
            if not line:
                break
            cnt += 1
            if cnt < start:
                continue
            if cnt >= end:
                break
            self.identifier_list[cnt - start]['text'] = line.strip().split()

