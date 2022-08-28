import json
import os
import shutil
import subprocess
import time
import sys


VALIDATE_DEFECTS4J_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
sys.path.append(VALIDATE_DEFECTS4J_DIR + '../dataloader/')

import tokenization


def clean_tmp_folder(tmp_dir):
    if os.path.isdir(tmp_dir):
        for files in os.listdir(tmp_dir):
            file_p = os.path.join(tmp_dir, files)
            try:
                if os.path.isfile(file_p):
                    os.unlink(file_p)
                elif os.path.isdir(file_p):
                    shutil.rmtree(file_p)
            except Exception as e:
                print(e)
    else:
        os.makedirs(tmp_dir)


def checkout_defects4j_project(project, bug_id, tmp_dir):
    FNULL = open(os.devnull, 'w')
    command = "defects4j checkout " + " -p " + project + " -v " + bug_id + " -w " + tmp_dir
    p = subprocess.Popen([command], shell=True, stdout=FNULL, stderr=FNULL)
    p.wait()


def compile_fix(project_dir):
    os.chdir(project_dir)
    p = subprocess.Popen(["defects4j", "compile"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if "FAIL" in str(err) or "FAIL" in str(out):
        return False
    return True


def command_with_timeout(cmd, timeout=300):
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


def defects4j_test_suite(project_dir, timeout=300):
    os.chdir(project_dir)
    out, err = command_with_timeout(["defects4j", "test", "-r"], timeout)
    return out, err


def defects4j_trigger(project_dir, timeout=300):
    os.chdir(project_dir)
    out, err = command_with_timeout(["defects4j", "export", "-p", "tests.trigger"], timeout)
    return out, err


def defects4j_relevant(project_dir, timeout=300):
    os.chdir(project_dir)
    out, err = command_with_timeout(["defects4j", "export", "-p", "tests.relevant"], timeout)
    return out, err


def defects4j_test_one(project_dir, test_case, timeout=300):
    os.chdir(project_dir)
    out, err = command_with_timeout(["defects4j", "test", "-t", test_case], timeout)
    return out, err


def insert_fix_defects4j(file_path, start_loc, end_loc, patch, project_dir):
    file_path = project_dir + file_path
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


def validate_defects4j(reranked_result_path, output_path, tmp_dir):
    global cnt, right

    if not os.path.exists(tmp_dir):
        command_with_timeout(['mkdir', tmp_dir])

    reranked_result = json.load(open(reranked_result_path, 'r'))
    validated_result = {}
    last_bug = None
    for key in reranked_result:
        cnt += 1
        
        proj, bug_id, path, start_loc, end_loc = key.split('-')
        print(right, '/', cnt, proj, bug_id)

        current_bug = proj + '_' + bug_id
        current_is_correct = False
        if last_bug != current_bug:
            last_bug = current_bug

            # checkout project
            clean_tmp_folder(tmp_dir)
            checkout_defects4j_project(proj, bug_id + 'b', tmp_dir)
            if proj == "Mockito":
                print("Mockito needs separate compilation")
                compile_fix(tmp_dir)

            # check standard test time
            start_time = time.time()
            init_out, init_err = defects4j_test_suite(tmp_dir)
            standard_time = int(time.time() - start_time)

            # check failed test cases
            failed_test_cases = str(init_out).split(' - ')[1:]
            for i, failed_test_case in enumerate(failed_test_cases):
                failed_test_cases[i] = failed_test_case.strip()
            init_fail_num = len(failed_test_cases)
            print(init_fail_num, str(standard_time) + 's')

            trigger, err = defects4j_trigger(tmp_dir)
            triggers = trigger.strip().split('\n')
            for i, trigger in enumerate(triggers):
                triggers[i] = trigger.strip()
            print('trigger number:', len(triggers))

            relevant, err = defects4j_relevant(tmp_dir)
            relevants = relevant.strip().split('\n')
            for i, relevant in enumerate(relevants):
                relevants[i] = relevant.strip()
            print('relevant number:', len(relevants))

        validated_result[key] = {'patches': []}
        bug_start_time = time.time()
        for tokenized_patch in reranked_result[key]['patches']:
            # validate 5 hours for each bug at most
            if time.time() - bug_start_time > 5 * 3600:
                break
            # validate 5000 patches for each bug at most
            if len(validated_result[key]['patches']) >= 5000:
                break

            score = tokenized_patch['score']
            tokenized_patch = tokenized_patch['patch']

            strings, numbers = get_strings_numbers(tmp_dir + path, (int(start_loc) + int(end_loc)) // 2)
            strings = [item[0] for item in strings][:5]
            numbers = [item[0] for item in numbers][:5]
            # one tokenized patch may be reconstructed to multiple source-code patches
            reconstructed_patches = tokenization.token2statement(tokenized_patch.split(' '), numbers, strings)
            # validate most 5 source-code patches come from the same tokenized patch
            for patch in reconstructed_patches[:5]:
                patch = patch.strip()

                patched_file = insert_fix_defects4j(path, int(start_loc), int(end_loc), patch, tmp_dir)
                if proj == 'Mockito':
                    # Mockito needs seperate compile
                    compile_fix(tmp_dir)

                # trigger cases is few and total time is long, we test trigger cases first.
                outs = []
                correctness = None
                start_time = time.time()
                if standard_time >= 10 and len(triggers) <= 5:
                    for trigger in triggers:
                        out, err = defects4j_test_one(tmp_dir, trigger)
                        if 'TIMEOUT' in str(err) or 'TIMEOUT' in str(out):
                            print(right, '/', cnt, current_bug, 'Time out for patch: ', patch,
                                  str(int(time.time() - start_time)) + 's')
                            correctness = 'timeout'
                            break
                        elif 'FAIL' in str(err) or 'FAIL' in str(out):
                            print(right, '/', cnt, current_bug, 'Uncompilable patch:', patch,
                                  str(int(time.time() - start_time)) + 's')
                            correctness = 'uncompilable'
                            break
                        elif "Failing tests: 0" in str(out):
                            continue
                        else:
                            outs += str(out).split(' - ')[1:]
                if len(set(outs)) >= len(triggers):
                    # does not pass any one more
                    print(right, '/', cnt, current_bug, 'Wrong patch:', patch,
                          str(int(time.time() - start_time)) + 's')
                    correctness = 'wrong'

                if correctness is None:
                    # pass at least one more trigger case
                    # have to pass all non-trigger
                    out, err = defects4j_test_suite(tmp_dir)

                    if 'TIMEOUT' in str(err) or 'TIMEOUT' in str(out):
                        print(right, '/', cnt, current_bug, 'Time out for patch: ', patch,
                              str(int(time.time() - start_time)) + 's')
                        correctness = 'timeout'
                    elif 'FAIL' in str(err) or 'FAIL' in str(out):
                        print(right, '/', cnt, current_bug, 'Uncompilable patch:', patch,
                              str(int(time.time() - start_time)) + 's')
                        correctness = 'uncompilable'
                    elif "Failing tests: 0" in str(out):
                        if not current_is_correct:
                            current_is_correct = True
                            right += 1
                        print(right, '/', cnt, current_bug, 'Plausible patch:', patch,
                              str(int(time.time() - start_time)) + 's')
                        correctness = 'plausible'
                    elif len(str(out).split(' - ')[1:]) < init_fail_num:
                        # fail less, could be correct
                        current_failed_test_cases = str(out).split(' - ')[1:]
                        no_new_fail = True
                        for current_failed_test_case in current_failed_test_cases:
                            if current_failed_test_case.strip() not in failed_test_cases:
                                no_new_fail = False
                                break
                        if no_new_fail:
                            # fail less and no new fail cases, could be plausible
                            if not current_is_correct:
                                current_is_correct = True
                                right += 1
                            print(right, '/', cnt, current_bug, 'Plausible patch:', patch,
                                  str(int(time.time() - start_time)) + 's')
                            correctness = 'plausible'
                        else:
                            print(right, '/', cnt, current_bug, 'Wrong patch:', patch,
                                  str(int(time.time() - start_time)) + 's')
                            correctness = 'wrong'
                    else:
                        print(right, '/', cnt, current_bug, 'Wrong patch:', patch,
                              str(int(time.time() - start_time)) + 's')
                        correctness = 'wrong'

                validated_result[key]['patches'].append({
                    'patch': patch, 'score': score, 'correctness': correctness
                })
                shutil.copyfile(patched_file, patched_file.replace('.bak', ''))

        # write after finish validating every bug, to avoid wasting time
        json.dump(validated_result, open(output_path, 'w'), indent=2)

    # write the last time after validating all
    json.dump(validated_result, open(output_path, 'w'), indent=2)


if __name__ == '__main__':
    reranked_result_path = VALIDATE_DEFECTS4J_DIR + '../../data/patches/reranked_patches.json'
    output_path = VALIDATE_DEFECTS4J_DIR + '../../data/patches/validated_patches.json'
    tmp_dir = '/tmp/validate_d4j/'
    validate_defects4j(reranked_result_path, output_path, tmp_dir)
