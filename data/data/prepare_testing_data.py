import codecs
import json
import re
import subprocess
import time
import os
import sys

IDENTIFIER_DATA_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
sys.path.append(IDENTIFIER_DATA_DIR + '../../src/dataloader/')

from tokenization import tokenize


def command_with_timeout(cmd, timeout=10):
    p = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    t_beginning = time.time()
    while True:
        if p.poll() is not None:
            break
        seconds_passed = time.time() - t_beginning
        if timeout and seconds_passed > timeout:
            p.terminate()
            return 'TIMEOUT', 'TIMEOUT'
        time.sleep(1)
    out, err = p.communicate()
    print(out, err)
    return out, err


def compile_java_parser():
    os.chdir(IDENTIFIER_DATA_DIR + '../../parser/')
    command_with_timeout([
        'javac', '-cp', '".:lib/*"', '-d', 'target', 'src/main/java/jiang719/*.java'
    ])


def call_java_parser(file_path, start_line, end_line, output_file):
    os.chdir(IDENTIFIER_DATA_DIR + '../../parser/')
    command_with_timeout([
        'java', '-cp', '.:target:lib/*', 'jiang719.CUREInput',
        file_path, output_file, str(start_line), str(end_line)
    ])


def write_buggy_ctx(result_file, output_file):
    result = json.load(open(result_file, 'r'))
    wp = codecs.open(output_file, 'a', 'utf-8')
    buggy_line = ' '.join(tokenize(re.sub('\\s+', ' ', result['buggy line'])))
    context = ' '.join(tokenize(re.sub('\\s+', ' ', result['context'] + '\t\n')))
    wp.write(buggy_line + ' <CTX> ' + context + '\t\n')
    wp.close()


def write_identifiers(java_class_path, java_keyword_path, result_path, identifier_txt_path, identifier_token_path):
    def get_identifiers(java_class_path, java_keyword_path, result_path):
        def get_imported_identifiers(imports, java_class):
            identifier = set()
            for line in imports + ['java.lang']:
                line = line.split('.')[-1]
                if 'A' <= line[0] <= 'Z':
                    identifier.add(line)
                    if line in java_class:
                        for method in java_class[line]:
                            identifier.add(method)
                elif line in java_class:
                    for module in java_class[line]:
                        if 'A' <= module[0] <= 'Z':
                            identifier.add(module)
                            if module in java_class:
                                for method in java_class[module]:
                                    identifier.add(method)
            return identifier

        java_class = json.load(open(java_class_path, 'r'))
        java_keyword = json.load(open(java_keyword_path, 'r'))

        identifiers = set(java_keyword) | {'<<unk>>', '$NUMBER$', '$STRING$', '_', '0xffffffff',
                                        '0x1f', '0x7f', 'T', '0', '1'}
        result = json.load(open(result_path, 'r'))
        identifiers |= set(result['identifiers'])
        identifiers |= set(
            get_imported_identifiers(result['imports'], java_class)
        )
        return identifiers
    identifiers = get_identifiers(java_class_path, java_keyword_path, result_path)

    wp1 = codecs.open(identifier_txt_path, 'a', 'utf-8')
    wp2 = codecs.open(identifier_token_path, 'a', 'utf-8')
    s1, s2 = '', ''
    for identifier in identifiers:
        if identifier in {'<<unk>>', '$NUMBER$', '$STRING$', '_'}:
            tokens = [identifier]
        else:
            tokens = tokenize(identifier)
            if identifier in ['<<', '>>', '==', '!=', '>=', '<=', '&&', '||', '++', '--', '-=', '+=', '*=',
                            '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '->', '<-', '::']:
                tokens = tokens[:-1]
        s2 += ' '.join(tokens) + ' <SEP> '
        s1 += identifier + ' '
    s1 = s1[: -1]
    s2 = s2[: -7]
    wp1.write(s1 + '\n')
    wp2.write(s2 + '\n')


def prepare_cure_input(buggy_file, start_line, end_line, java_class_path, java_keyword_path, tmp_dir, output_dir):
    call_java_parser(buggy_file, start_line, end_line, tmp_dir + '/tmp.json')
    write_buggy_ctx(tmp_dir + '/tmp.json', output_dir + '/input.txt')
    write_identifiers(
        java_class_path, java_keyword_path, tmp_dir + '/tmp.json', 
        output_dir + '/identifier.txt', output_dir + '/identifier.tokens'
    )


def clean_testing_bpe(input_bpe_file, identifier_bpe_file):
    fp = codecs.open(input_bpe_file, 'r', 'utf-8')
    lines = fp.readlines()
    fp.close()
    wp = codecs.open(input_bpe_file, 'w', 'utf-8')
    for l in lines:
        l = re.sub('@@ 	@@ ', '\t', l)
        l = re.sub('<@@ CT@@ X@@ >', ' <CTX> ', l)
        wp.write(l)
    wp.close()

    fp = codecs.open(identifier_bpe_file, 'r', 'utf-8')
    lines = fp.readlines()
    fp.close()
    wp = codecs.open(identifier_bpe_file, 'w', 'utf-8')
    for l in lines:
        wp.write(re.sub(' <@@ SE@@ P@@ > ', '\t', l).strip() + '\n')
    wp.close()


if __name__ == '__main__':
    # compile_java_parser()
    prepare_cure_input(
        buggy_file=IDENTIFIER_DATA_DIR + '../../parser/src/test/java/jiang719/Test.java',
        start_line=11, 
        end_line=12,
        java_class_path=IDENTIFIER_DATA_DIR + 'java_class.json',
        java_keyword_path=IDENTIFIER_DATA_DIR + 'java_keyword.json',
        tmp_dir='/tmp/',
        output_dir=IDENTIFIER_DATA_DIR
    )

    """
    Run subword-nmt to perform subword tokenization
    subword-nmt apply-bpe -c ../vocabulary/subword.txt < input.txt > input_bpe.txt
    subword-nmt apply-bpe -c ../vocabulary/subword.txt < identifier.tokens > identifier_bpe.tokens
    """

    # run clean_testing_bpe() after running the subword-nmt commands above
    # clean_testing_bpe(
    #    IDENTIFIER_DATA_DIR + 'input_bpe.txt',
    #    IDENTIFIER_DATA_DIR + 'identifier_bpe.tokens'
    # )
