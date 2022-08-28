import json
import os
import shutil
import time
import subprocess
import sys

VALIDATE_QUIXBUGS_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
sys.path.append(VALIDATE_QUIXBUGS_DIR + '../dataloader/')

import tokenization


def command_with_timeout(cmd, timeout=5):
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
    return out, err


def compile_fix(filename, tmp_dir):
    FNULL = open(os.devnull, 'w')
    p = subprocess.call(["javac",
                         tmp_dir + "Node.java",
                         tmp_dir + "WeightedEdge.java",
                         filename], stderr=FNULL)
    return False if p else True


def quixbugs_test_suite(algo, quixbugs_dir):
    QUIXBUGS_MAIN_DIR = quixbugs_dir
    CUR_DIR = os.getcwd()
    FNULL = open(os.devnull, 'w')
    try:
        os.chdir(QUIXBUGS_MAIN_DIR)
        p1 = subprocess.Popen(["/usr/bin/javac", "-cp", ".:java_programs:junit4-4.12.jar:hamcrest-all-1.3.jar", 
                                "java_testcases/junit/" + algo.upper() + "_TEST.java"],
                                stdout=subprocess.PIPE, stderr=FNULL, universal_newlines=True)
        out, err = command_with_timeout(
            ["/usr/bin/java", "-cp", ".:java_programs:junit4-4.12.jar:hamcrest-all-1.3.jar",
             "org.junit.runner.JUnitCore", "java_testcases.junit." + algo.upper() + "_TEST"], timeout=5
        )

        os.chdir(CUR_DIR)
        if "FAILURES" in str(out) or "FAILURES" in str(err):
            return 'wrong'
        elif "TIMEOUT" in str(out) or "TIMEOUT" in str(err):
            return 'timeout'
        else:
            return 'plausible'
    except Exception as e:
        print(e)
        os.chdir(CUR_DIR)
        return 'uncompilable'


def insert_fix_quixbugs(file_path, start_loc, end_loc, patch):
    shutil.copyfile(file_path, file_path + '.bak')

    with open(file_path, 'r') as file:
        data = file.readlines()

    patched = False
    with open(file_path, 'w') as file:
        for idx, line in enumerate(data):
            if start_loc - 1 <= idx < end_loc - 1:
                if not patched:
                    file.write(patch)
                    patched = True
            else:
                file.write(line)

    return file_path + '.bak'


def get_strings_numbers(file_path, loc):
    numbers_set = {}
    strings_set = {}
    with open(file_path, 'r') as file:
        data = file.readlines()
        for idx, line in enumerate(data):
            dist = loc - idx - 1
            strings, numbers = tokenization.get_strings_numbers(line)
            for num in numbers:
                if num != '0' and num != '1':
                    if num in numbers_set:
                        numbers_set[num] = min(dist, numbers_set[num])
                    else:
                        numbers_set[num] = dist
            for str in strings:
                if str in strings_set:
                    strings_set[str] = min(dist, strings_set[str])
                else:
                    strings_set[str] = dist
    final_strings = []
    final_numbers = []
    for k, v in numbers_set.items():
        final_numbers.append([k, v])
    for k, v in strings_set.items():
        final_strings.append([k, v])
    final_numbers.sort(key=lambda x: x[1])
    final_strings.sort(key=lambda x: x[1])
    return final_strings, final_numbers


cnt, right = 0, 0


def validate_quixbugs(reranked_result_path, output_path, tmp_dir):
    global cnt, right

    if not os.path.exists(tmp_dir):
        command_with_timeout(['mkdir', tmp_dir])

    reranked_result = json.load(open(reranked_result_path, 'r'))
    validated_result = {}
    for key in reranked_result:
        cnt += 1

        proj, start_loc, end_loc = key.split('-')
        print(right, '/', cnt, proj)

        command_with_timeout(['rm', '-rf', tmp_dir + '/java_programs/'])
        command_with_timeout(['mkdir', tmp_dir + '/java_programs/'])

        shutil.copyfile(tmp_dir + "/java_programs_bak/" + proj + '.java',
                        tmp_dir + "/java_programs/" + proj + '.java')
        shutil.copyfile(tmp_dir + "/java_programs_bak/Node.java", tmp_dir + "/java_programs/Node.java")
        shutil.copyfile(tmp_dir + "/java_programs_bak/WeightedEdge.java", tmp_dir + "/java_programs/WeightedEdge.java")

        validated_result[key] = {'src': reranked_result[key]['src'], 'patches': []}
        bug_start_time = time.time()
        current_is_correct = False
        for tokenized_patch in reranked_result[key]['patches']:
            # validate 5 hours for each bug at most
            if time.time() - bug_start_time > 5 * 3600:
                break
            # validate 5000 patches for each bug at most
            if len(validated_result[key]['patches']) >= 5000:
                break
            filename = tmp_dir + "/java_programs/" + proj + '.java'

            score = tokenized_patch['score']
            tokenized_patch = tokenized_patch['patch']

            strings, numbers = get_strings_numbers(filename, (int(start_loc) + int(end_loc)) // 2)
            strings = [item[0] for item in strings][:5]
            numbers = [item[0] for item in numbers][:5]
            # one tokenized patch may be reconstructed to multiple source-code patches
            reconstructed_patches = tokenization.token2statement(tokenized_patch.split(' '), numbers, strings)
            # validate most 5 source-code patches come from the same tokenized patch
            for patch in reconstructed_patches[:5]:
                patch = patch.strip()
                insert_fix_quixbugs(filename, int(start_loc), int(end_loc), patch)
                compile = compile_fix(filename, tmp_dir + "/java_programs/")
                correctness = 'uncompilable'
                if compile:
                    correctness = quixbugs_test_suite(proj, quixbugs_dir=tmp_dir)
                    if correctness == 'plausible':
                        if not current_is_correct:
                            right += 1
                            current_is_correct = True
                        print(right, '/', cnt, "Plausible patch:", patch)
                        break
                    elif correctness == 'wrong':
                        print(right, '/', cnt, "Wrong patch:", patch)
                    elif correctness == 'timeout':
                        print(right, '/', cnt, "Timeout patch:", patch)
                else:
                    print(right, '/', cnt, 'Uncompilable patch:', patch)
                validated_result[key]['patches'].append({
                    'patch': patch, 'correctness': correctness
                })
                shutil.copyfile(tmp_dir + "/java_programs_bak/" + proj + '.java',
                                tmp_dir + "/java_programs/" + proj + '.java')
            json.dump(validated_result, open(output_path, 'w'), indent=2)
            if current_is_correct:
                break
        json.dump(validated_result, open(output_path, 'w'), indent=2)


if __name__ == '__main__':
    reranked_result_path = VALIDATE_QUIXBUGS_DIR + '../../data/patches/reranked_patches.json'
    output_path = VALIDATE_QUIXBUGS_DIR + '../../data/patches/validated_patches.json'
    tmp_dir = '/tmp/validate_quixbugs/'
    validate_quixbugs(reranked_result_path, output_path, tmp_dir)
