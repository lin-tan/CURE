import codecs
import json
import os

RERANK_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]


def read_defects4j_meta(meta_path):
    fp = codecs.open(meta_path, 'r', 'utf-8')
    meta = []
    for l in fp.readlines():
        proj, bug_id, path, start_loc, end_loc = l.strip().split('\t')
        meta.append([
            proj, bug_id, path, start_loc, end_loc + 1
        ])
    return meta


def read_quixbugs_meta(meta_path):
    fp = codecs.open(meta_path, 'r', 'utf-8')
    meta = []
    for l in fp.readlines():
        proj, loc = l.strip().split('\t')
        if '-' in loc:
            start_loc, end_loc = loc.split('-')
            start_loc = int(start_loc)
            end_loc = int(end_loc)
        else:
            start_loc, end_loc = int(loc), int(loc) + 1
        meta.append([
            proj, str(start_loc), str(end_loc)
        ])
    return meta


def read_hypo(hypo_path):
    fp = codecs.open(hypo_path, 'r', 'utf-8')
    hypo = {}
    for l in fp.readlines():
        l = l.strip().split()
        if l[0][:2] == 'S-':
            id = int(l[0][2:])
            src = ' '.join(l[1:]).strip()
            src = src.replace('@@ ', '')
            hypo[id] = {'src': src, 'patches': []}
        if l[0][:2] == 'H-':
            id = int(l[0][2:])
            patch = ' '.join(l[2:]).strip()
            patch = patch.replace('@@ ', '')
            score = float(l[1])
            hypo[id]['patches'].append([patch, score])
    return hypo


def cure_rerank(meta, hypo_path_list, output_path):
    # the patch with same rank from different models are grouped together
    group_by_rank = {}
    for hypo_path in hypo_path_list:
        hypo = read_hypo(hypo_path)
        print('finish loading', hypo_path)
        for id in hypo:
            if id not in group_by_rank:
                group_by_rank[id] = {'src': hypo[id]['src'], 'patches': []}
            for rank, (patch, score) in enumerate(hypo[id]['patches']):
                if rank >= len(group_by_rank[id]['patches']):
                    group_by_rank[id]['patches'].append([])
                group_by_rank[id]['patches'][rank].append([patch, score])

    # the patch with same rank are ranked by scores
    reranked_hypo = {}
    print('start ranking')
    for id in group_by_rank:
        key = '-'.join(meta[id])
        reranked_hypo[key] = {'src': group_by_rank[id]['src'], 'patches': []}

        added_patches = set()
        for patches_same_rank in group_by_rank[id]['patches']:
            ranked_by_score = sorted(patches_same_rank, key=lambda e: e[1], reverse=True)
            for patch, score in ranked_by_score:
                if patch not in added_patches:
                    added_patches.add(patch)
                    reranked_hypo[key]['patches'].append({'patch': patch, 'score': score})
            if len(added_patches) >= 5000:
                break

    print('dumping result in json file')
    json.dump(reranked_hypo, open(output_path, 'w'), indent=2)


if __name__ == '__main__':
    meta_path = RERANK_DIR + '../../candidate_patches/QuixBugs/meta.txt'
    quixbugs_meta = read_quixbugs_meta(meta_path)
    hypo_path_list = [RERANK_DIR + '../../data/patches/gpt_conut_1.txt'] + \
                     [RERANK_DIR + '../../data/patches/gpt_fconv_1.txt']
    output_path = RERANK_DIR + '../../data/patches/reranked_patches.json'
    cure_rerank(quixbugs_meta, hypo_path_list, output_path)
