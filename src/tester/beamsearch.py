import re
import os
import math
import torch
import torch.nn as nn
import sys

BEAM_SEARCH_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
sys.path.append(BEAM_SEARCH_DIR + '../models/')
from gpt_conut import GPTCoNuTModel
from gpt_fconv import GPTFConvModel


def get_statement_length(seq):
    s = re.sub('\\s*CaMeL\\s*', 'CaMeL', seq)
    s = re.sub('\\s*_\\s*', '_', s)
    s = re.sub('\\s*\\.\\s*', '.', s)
    s = s.replace('@@ ', '')
    return len(s.strip().split())


def add_token_to_string(string, symbol):
    if symbol in ['CaMeL', '_', '.']:
        return string + symbol
    elif string[-5:] == 'CaMeL' or string[-1:] in ['_', '.'] or string[-2:] == '@@':
        return string + symbol
    else:
        return string + ' ' + symbol


class GPTCoNuTModelCuda(nn.Module):
    def __init__(self, model, beam_size):
        super(GPTCoNuTModelCuda, self).__init__()
        self.model = model.cuda()
        self.beam_size = beam_size
        self.split_size = beam_size
        self.split_size_list = [self.split_size]

    def forward(self):
        pass

    def encoder_out_to_cuda(self, encoder_out):
        return {
            'src_tokens': encoder_out['src_tokens'].cuda(),
            'encoder_out': (
                encoder_out['encoder_out'][0].cuda(),
                encoder_out['encoder_out'][1].cuda(),
            ),
            'encoder_padding_mask': encoder_out['encoder_padding_mask'].cuda(),
        }

    def encoder_out_to_cpu(self, encoder_out):
        return {
            'src_tokens': encoder_out['src_tokens'].to('cpu'),
            'encoder_out': (
                encoder_out['encoder_out'][0].to('cpu'),
                encoder_out['encoder_out'][1].to('cpu'),
            ),
            'encoder_padding_mask': encoder_out['encoder_padding_mask'].to('cpu'),
        }

    def encode(self, src_tokens, src_with_prev_context, ctx_tokens):
        encoder_out = self.model.encoder(
            src_tokens.cuda(),
            src_with_prev_context.cuda(),
            ctx_tokens.cuda(),
            self.model.embed_model,
        )
        return self.encoder_out_to_cpu(encoder_out)

    def decode(self, prev_tokens_index, encoder_out, prev_tokens):
        step = int(torch.sum(prev_tokens_index[0]))
        ctx_len = prev_tokens.size(1)
        if step * ctx_len <= 3000:
            self.split_size = min(self.beam_size, 200)
        elif step * ctx_len <= 5000:
            self.split_size = min(self.beam_size, 100)
        elif step * ctx_len <= 10000:
            self.split_size = min(self.beam_size, 50)
        else:
            self.split_size = min(self.beam_size, 20)
        split_num = self.beam_size // self.split_size
        self.split_size_list = [self.split_size] * split_num
        if self.beam_size % self.split_size != 0:
            self.split_size_list += [self.beam_size % self.split_size]
        if prev_tokens_index.size(0) == 1:
            return self.model.decoder(
                prev_tokens_index.cuda(),
                self.encoder_out_to_cuda(encoder_out),
                prev_tokens.cuda(),
                self.model.embed_model,
            )[0].to('cpu')
        else:
            assert prev_tokens_index.size(0) == sum(self.split_size_list)
            decoder_out = []
            split_encoder_out = {
                'src_tokens': encoder_out['src_tokens'][:self.split_size, ...].cuda(),
                'encoder_out': (
                    encoder_out['encoder_out'][0][:self.split_size, ...].cuda(),
                    encoder_out['encoder_out'][1][:self.split_size, ...].cuda(),
                ),
                'encoder_padding_mask': encoder_out['encoder_padding_mask'][:self.split_size, ...].cuda(),
            }
            for i in range(len(self.split_size_list)):
                if i == len(self.split_size_list) - 1:
                    split_encoder_out = {
                        'src_tokens': split_encoder_out['src_tokens'][:self.split_size_list[-1], ...],
                        'encoder_out': (
                            split_encoder_out['encoder_out'][0][:self.split_size_list[-1], ...],
                            split_encoder_out['encoder_out'][1][:self.split_size_list[-1], ...],
                        ),
                        'encoder_padding_mask': split_encoder_out['encoder_padding_mask'][:self.split_size_list[-1], ...],
                    }
                logits = self.model.decoder(
                    prev_tokens_index[sum(self.split_size_list[: i]) : sum(self.split_size_list[: i + 1]), ...].cuda(),
                    split_encoder_out,
                    prev_tokens[sum(self.split_size_list[: i]) : sum(self.split_size_list[: i + 1]), ...].cuda(),
                    self.model.embed_model,
                )[0]
                logits = logits[:, -1, :]  # beam, L, V -> beam, V
                decoder_out.append(logits.to('cpu'))
        logits = torch.cat(decoder_out, dim=0)
        # logits = logits.to('cpu')
        return logits


class GPTFConvModelCuda(nn.Module):
    def __init__(self, model, beam_size):
        super(GPTFConvModelCuda, self).__init__()
        self.model = model.cuda()
        self.beam_size = beam_size
        self.split_size = beam_size
        self.split_size_list = [self.split_size]

    def forward(self):
        pass

    def encoder_out_to_cuda(self, encoder_out):
        return {
            'src_tokens': encoder_out['src_tokens'].cuda(),
            'encoder_out': (
                encoder_out['encoder_out'][0].cuda(),
                encoder_out['encoder_out'][1].cuda(),
            ),
            'encoder_padding_mask': encoder_out['encoder_padding_mask'].cuda(),
        }

    def encoder_out_to_cpu(self, encoder_out):
        return {
            'src_tokens': encoder_out['src_tokens'].to('cpu'),
            'encoder_out': (
                encoder_out['encoder_out'][0].to('cpu'),
                encoder_out['encoder_out'][1].to('cpu'),
            ),
            'encoder_padding_mask': encoder_out['encoder_padding_mask'].to('cpu'),
        }

    def encode(self, src_tokens, src_with_prev_context):
        encoder_out = self.model.encoder(
            src_tokens.cuda(),
            src_with_prev_context.cuda(),
            self.model.embed_model,
        )
        return self.encoder_out_to_cpu(encoder_out)

    def decode(self, prev_tokens_index, encoder_out, prev_tokens):
        step = int(torch.sum(prev_tokens_index[0]))
        ctx_len = prev_tokens.size(1)
        if step * ctx_len <= 3000:
            self.split_size = min(self.beam_size, 200)
        elif step * ctx_len <= 5000:
            self.split_size = min(self.beam_size, 100)
        elif step * ctx_len <= 10000:
            self.split_size = min(self.beam_size, 50)
        else:
            self.split_size = min(self.beam_size, 20)
        split_num = self.beam_size // self.split_size
        self.split_size_list = [self.split_size] * split_num
        if self.beam_size % self.split_size != 0:
            self.split_size_list += [self.beam_size % self.split_size]
        if prev_tokens_index.size(0) == 1:
            return self.model.decoder(
                prev_tokens_index.cuda(),
                self.encoder_out_to_cuda(encoder_out),
                prev_tokens.cuda(),
                self.model.embed_model,
            )[0].to('cpu')
        else:
            assert prev_tokens_index.size(0) == sum(self.split_size_list)
            decoder_out = []
            split_encoder_out = {
                'src_tokens': encoder_out['src_tokens'][:self.split_size, ...].cuda(),
                'encoder_out': (
                    encoder_out['encoder_out'][0][:self.split_size, ...].cuda(),
                    encoder_out['encoder_out'][1][:self.split_size, ...].cuda(),
                ),
                'encoder_padding_mask': encoder_out['encoder_padding_mask'][:self.split_size, ...].cuda(),
            }
            for i in range(len(self.split_size_list)):
                if i == len(self.split_size_list) - 1:
                    split_encoder_out = {
                        'src_tokens': split_encoder_out['src_tokens'][:self.split_size_list[-1], ...],
                        'encoder_out': (
                            split_encoder_out['encoder_out'][0][:self.split_size_list[-1], ...],
                            split_encoder_out['encoder_out'][1][:self.split_size_list[-1], ...],
                        ),
                        'encoder_padding_mask': split_encoder_out['encoder_padding_mask'][:self.split_size_list[-1], ...],
                    }
                logits = self.model.decoder(
                    prev_tokens_index[sum(self.split_size_list[: i]) : sum(self.split_size_list[: i + 1]), ...].cuda(),
                    split_encoder_out,
                    prev_tokens[sum(self.split_size_list[: i]) : sum(self.split_size_list[: i + 1]), ...].cuda(),
                    self.model.embed_model,
                )[0]
                logits = logits[:, -1, :]  # beam, L, V -> beam, V
                decoder_out.append(logits.to('cpu'))
        logits = torch.cat(decoder_out, dim=0)
        # logits = logits.to('cpu')
        return logits


class BeamSearch():
    def __init__(self, model, dictionary, beam_size=10):
        self.dictionary = dictionary
        if isinstance(model, GPTCoNuTModel):
            self.model = GPTCoNuTModelCuda(model, beam_size)
        elif isinstance(model, GPTFConvModel):
            self.model = GPTFConvModelCuda(model, beam_size)
        self.beam_size = beam_size
        self.max_step = 128

    @staticmethod
    def get_prefix(token, dictionary):
        prefixs, texts = [], []
        prefix, text = '', ''
        stop = False
        for i in range(len(token) - 1, -1, -1):
            cur = dictionary[token[i]]
            if cur not in ['CaMeL', '_', '0', '1', '$NUMBER$']:
                if not stop:
                    stop = True
                    prefix = cur + prefix
                    text = cur + text
                    prefixs.append(prefix)
                    if text[-2:] != '@@':
                        texts.append(text.replace('@@', ''))
                    else:
                        texts.append(text)
                else:
                    if cur[-2:] == '@@':
                        prefix = cur + prefix
                        text = cur + text
                        prefixs.append(prefix)
                        if text[-2:] != '@@':
                            texts.append(text.replace('@@', ''))
                        else:
                            texts.append(text)
                    else:
                        return prefixs, texts
            else:
                stop = False
                prefix = cur + prefix
                if cur != 'CaMeL':
                    text = cur + text
                prefixs.append(prefix)
                if text[-2:] != '@@':
                    texts.append(text.replace('@@', ''))
                else:
                    texts.append(text)
        prefixs.append(prefix)
        if text[-2:] != '@@':
            texts.append(text.replace('@@', ''))
        else:
            texts.append(text)
        return prefixs, texts

    def generate_gpt_fconv(self, sample):
        self.model.eval()
        hypothesis = []
        src_tokens = sample['net_input']['src_tokens']
        src_with_prev_context = sample['net_input']['src_with_prev_context']
        identifiers = sample['identifier']

        self.max_step = max(100, int(torch.sum(src_tokens[0])))
        encoder_out = self.model.encode(
            src_tokens,
            src_with_prev_context,
        )

        prev_tokens_index = sample['target_index']
        prev_tokens_with_context = sample['target_with_prev_context']

        mask = prev_tokens_index.eq(0)
        prev_tokens_index = prev_tokens_index[mask].unsqueeze(0)
        prev_tokens_index = torch.cat([prev_tokens_index, torch.ones(1, 1).long()], dim=-1)
        prev_tokens_with_context = prev_tokens_with_context[:, : prev_tokens_index.size(1)]
        prev_len = prev_tokens_index.size(1)

        self.max_step += prev_tokens_index.size(1)
        bsz = src_tokens.size(0)  # bsz = 1
        tokens = torch.zeros(bsz * self.beam_size, self.max_step - prev_len).long()
        scores = torch.zeros(bsz * self.beam_size, self.max_step - prev_len)
        final_scores = torch.zeros(bsz * self.beam_size)
        tokens_string = ['' for _ in range(tokens.size(0))]

        prev_tokens_index = prev_tokens_index.repeat(bsz * self.beam_size, 1)
        prev_tokens_with_context = prev_tokens_with_context.repeat(bsz * self.beam_size, 1)
        tokens = torch.cat([prev_tokens_with_context, tokens], dim=1)

        if identifiers is not None:
            for k in identifiers[0]['tokens']:
                identifiers[0]['tokens'][k] += [self.dictionary.pad(),
                                                self.dictionary.unk(),
                                                self.dictionary.eos()]
                identifiers[0]['tokens'][k] = torch.LongTensor(identifiers[0]['tokens'][k])

        length_penalty = {-50: -6.5446, -49: -6.7353, -48: -6.5446, -47: -6.4651, -46: -6.5255,
                          -45: -6.5624, -44: -6.2495, -43: -6.4651, -42: -6.1886, -41: -6.4025,
                          -40: -6.2033, -39: -6.2622, -38: -6.1274, -37: -6.2537, -36: -6.0743,
                          -35: -6.0758, -34: -6.031, -33: -5.927, -32: -5.866, -31: -5.9582,
                          -30: -5.64, -29: -5.7394, -28: -5.6764, -27: -5.5039, -26: -5.3938,
                          -25: -5.4417, -24: -5.3806, -23: -5.299, -22: -5.1481, -21: -5.1686,
                          -20: -5.0302, -19: -4.9543, -18: -4.8488, -17: -4.6396, -16: -4.6334,
                          -15: -4.5676, -14: -4.4454, -13: -4.2981, -12: -4.1142, -11: -4.048,
                          -10: -3.7681, -9: -3.5306, -8: -3.834, -7: -3.1647, -6: -3.011,
                          -5: -2.9796, -4: 0.0, -3: 0.0, -2: 0.0, -1: 0.0, 0: 0.0}

        for step in range(0, self.max_step - prev_len):
            if step == 0:
                logits = self.model.decode(
                    prev_tokens_index[:bsz, :],
                    encoder_out,
                    tokens[:bsz, :step + prev_len],
                )
                logits = logits[:, -1, :]  # 1, V
                logits[:, self.dictionary.pad()] = -math.inf
                if identifiers is not None:
                    logits += 100
                lprobs, indices = logits.topk(k=self.beam_size, dim=1)
                tokens[:, prev_len + step: prev_len + step + 1] = indices.transpose(0, 1)
                prev_tokens_index = torch.cat([prev_tokens_index, torch.ones(self.beam_size, 1).long()], dim=-1)
                scores[:, step: step + 1] = lprobs.transpose(0, 1)
                final_scores = scores[:, step]
                for i, string in enumerate(tokens_string):
                    symbol = ''
                    if int(tokens[i, prev_len + step]) != self.dictionary.eos() and \
                            int(tokens[i, prev_len + step]) != self.dictionary.pad():
                        symbol = self.dictionary[int(tokens[i, prev_len + step])]
                    tokens_string[i] = add_token_to_string(string, symbol)
                continue
            if step == 1:
                # bsz * beam, L
                split_size = int(self.model.split_size)
                encoder_out['src_tokens'] = encoder_out['src_tokens']. \
                    repeat(split_size, 1)
                encoder_out['encoder_out'] = (
                    encoder_out['encoder_out'][0].repeat(split_size, 1, 1),
                    encoder_out['encoder_out'][1].repeat(split_size, 1, 1),
                )
                encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask']. \
                    repeat(split_size, 1)

            logits = self.model.decode(
                prev_tokens_index,
                encoder_out,
                tokens[:, :prev_len + step],
            )
            logits[:, self.dictionary.pad()] = -math.inf

            if identifiers is not None:
                for i in range(tokens.size(0)):
                    src_length = int(sample['src_statement_length'][0])
                    cur_length = len(tokens_string[i].strip().split())
                    if cur_length > src_length:
                        gap = max(-50, src_length - cur_length)
                    else:
                        gap = max(-50, cur_length - src_length)
                    if cur_length < src_length:
                        logits[i, self.dictionary.eos()] += length_penalty[gap]
                    else:
                        logits[i, self.dictionary.eos()] -= length_penalty[gap]

                    if identifiers[0]['text'] == []:
                        continue

                    prefixs, texts = BeamSearch.get_prefix(
                        tokens[i][prev_len: prev_len + step],
                        self.dictionary
                    )
                    prefixs.reverse()
                    texts.reverse()
                    tmp = None
                    for prefix, text in zip(prefixs, texts):
                        if prefix in identifiers[0]['tokens']:
                            if prefix != '' and prefix[-5:] != 'CaMeL' and prefix[-1] != '_' and \
                                    text in identifiers[0]['text']:
                                tmp = set(identifiers[0]['tokens'][prefix].tolist()) | \
                                      set(identifiers[0]['tokens'][''].tolist())
                                tmp = torch.LongTensor(list(tmp))
                            else:
                                tmp = identifiers[0]['tokens'][prefix]
                            break
                        if text in identifiers[0]['text']:
                            break
                    if tmp is None:
                        prefix = ''
                        tmp = identifiers[0]['tokens'][prefix]
                    logits[i, tmp] += 100
            lprobs, indices = logits.topk(k=self.beam_size, dim=1)  # beam, beam
            lprobs = lprobs.t().contiguous().view(-1)  # beam x beam
            indices = indices.t().contiguous().view(-1)
            cand_final_scores = lprobs + final_scores.repeat(self.beam_size)
            cand_final_scores, sort_order = cand_final_scores.sort(descending=True)
            lprobs = lprobs.index_select(0, sort_order)
            indices = indices.index_select(0, sort_order)

            # choose finished beam
            eos_mask = indices[: self.beam_size].eq(self.dictionary.eos())
            eos_lprobs = lprobs[: self.beam_size].masked_select(mask=eos_mask)  # N_eos
            eos_cand_final_scores = cand_final_scores[: self.beam_size].masked_select(mask=eos_mask)  # N_eos
            eos_sort_order = sort_order[: self.beam_size].masked_select(mask=eos_mask)  # N_eos
            if eos_cand_final_scores.size(0) > 0:
                eos_beam_ids = eos_sort_order % self.beam_size  # N_eos
                eos_beam_ids = eos_beam_ids.long()
                eos_beam = tokens[eos_beam_ids, prev_len:]  # N_eos, L
                eos_beam[:, step] = self.dictionary.eos()
                eos_beam_scores = scores[eos_beam_ids, :]
                eos_beam_scores[:, step] = eos_lprobs
                for i in range(eos_beam.size(0)):
                    hypothesis.append(
                        {
                            'hypo': eos_beam[i, : step + 1],
                            'score': eos_beam_scores[i, 1: step + 1],
                            'final_score': float(eos_cand_final_scores[i]) / (1 + step),
                        })
                if len(hypothesis) >= self.beam_size:
                    hypothesis = hypothesis[: self.beam_size]
                    break

            # choose next beam
            cand_mask = ~indices.eq(self.dictionary.eos())
            cand_final_scores = cand_final_scores.masked_select(mask=cand_mask)[: self.beam_size]
            sort_order = sort_order.masked_select(mask=cand_mask)[: self.beam_size]
            lprobs = lprobs.masked_select(mask=cand_mask)[: self.beam_size]
            indices = indices.masked_select(mask=cand_mask)[: self.beam_size]
            cand_beam_ids = sort_order % self.beam_size
            cand_beam_ids = cand_beam_ids.long()

            new_tokens_string = []
            for i in range(cand_beam_ids.size(0)):
                symbol = ''
                if int(indices[i]) not in [self.dictionary.eos(), self.dictionary.pad()]:
                    symbol = self.dictionary[int(indices[i])]
                new_tokens_string.append(
                    add_token_to_string(tokens_string[int(cand_beam_ids[i])], symbol)
                )
            tokens_string = new_tokens_string

            tokens = tokens[cand_beam_ids, :]
            tokens[:, prev_len + step] = indices

            scores = scores[cand_beam_ids, :]
            scores[:, step] = lprobs
            final_scores = cand_final_scores
            prev_tokens_index = torch.cat([prev_tokens_index,
                                           torch.ones(self.beam_size, 1).long()], dim=-1)

        if len(hypothesis) < self.beam_size:
            current_num = len(hypothesis)
            for i in range(self.beam_size - current_num):
                tokens[i, -1] = self.dictionary.eos()
                hypothesis.append({
                    'hypo': tokens[i],
                    'score': scores[i, 1:],
                    'final_score': float(final_scores[i]) / (self.max_step - prev_len),
                })

        hypothesis.sort(key=lambda e: e['final_score'], reverse=True)
        return hypothesis

    def generate_gpt_conut(self, sample):
        self.model.eval()
        hypothesis = []
        src_tokens = sample['net_input']['src_tokens']
        ctx_tokens = sample['net_input']['ctx_tokens']
        src_with_prev_context = sample['net_input']['src_with_prev_context']
        identifiers = sample['identifier']

        self.max_step = max(100, int(torch.sum(src_tokens[0])))
        encoder_out = self.model.encode(
            src_tokens,
            src_with_prev_context,
            ctx_tokens,
        )

        prev_tokens_index = sample['target_index']
        prev_tokens_with_context = sample['target_with_prev_context']

        mask = prev_tokens_index.eq(0)
        prev_tokens_index = prev_tokens_index[mask].unsqueeze(0)
        prev_tokens_index = torch.cat([prev_tokens_index, torch.ones(1, 1).long()], dim=-1)
        prev_tokens_with_context = prev_tokens_with_context[:, : prev_tokens_index.size(1)]
        prev_len = prev_tokens_index.size(1)

        self.max_step += prev_tokens_index.size(1)
        bsz = src_tokens.size(0)  # bsz = 1
        tokens = torch.zeros(bsz * self.beam_size, self.max_step - prev_len).long()
        scores = torch.zeros(bsz * self.beam_size, self.max_step - prev_len)
        final_scores = torch.zeros(bsz * self.beam_size)
        tokens_string = ['' for _ in range(tokens.size(0))]

        prev_tokens_index = prev_tokens_index.repeat(bsz * self.beam_size, 1)
        prev_tokens_with_context = prev_tokens_with_context.repeat(bsz * self.beam_size, 1)
        tokens = torch.cat([prev_tokens_with_context, tokens], dim=1)

        if identifiers is not None:
            for k in identifiers[0]['tokens']:
                identifiers[0]['tokens'][k] += [self.dictionary.pad(),
                                                self.dictionary.unk(),
                                                self.dictionary.eos()]
                identifiers[0]['tokens'][k] = torch.LongTensor(identifiers[0]['tokens'][k])

        length_penalty = {-50: -6.5446, -49: -6.7353, -48: -6.5446, -47: -6.4651, -46: -6.5255,
                          -45: -6.5624, -44: -6.2495, -43: -6.4651, -42: -6.1886, -41: -6.4025,
                          -40: -6.2033, -39: -6.2622, -38: -6.1274, -37: -6.2537, -36: -6.0743,
                          -35: -6.0758, -34: -6.031, -33: -5.927, -32: -5.866, -31: -5.9582,
                          -30: -5.64, -29: -5.7394, -28: -5.6764, -27: -5.5039, -26: -5.3938,
                          -25: -5.4417, -24: -5.3806, -23: -5.299, -22: -5.1481, -21: -5.1686,
                          -20: -5.0302, -19: -4.9543, -18: -4.8488, -17: -4.6396, -16: -4.6334,
                          -15: -4.5676, -14: -4.4454, -13: -4.2981, -12: -4.1142, -11: -4.048,
                          -10: -3.7681, -9: -3.5306, -8: -3.834, -7: -3.1647, -6: -3.011,
                          -5: -2.9796, -4: 0.0, -3: 0.0, -2: 0.0, -1: 0.0, 0: 0.0}

        for step in range(0, self.max_step - prev_len):
            if step == 0:
                logits = self.model.decode(
                    prev_tokens_index[:bsz, :],
                    encoder_out,
                    tokens[:bsz, :step + prev_len],
                )
                logits = logits[:, -1, :]  # 1, V
                logits[:, self.dictionary.pad()] = -math.inf
                if identifiers is not None:
                    logits += 100
                lprobs, indices = logits.topk(k=self.beam_size, dim=1)
                tokens[:, prev_len + step: prev_len + step + 1] = indices.transpose(0, 1)
                prev_tokens_index = torch.cat([prev_tokens_index, torch.ones(self.beam_size, 1).long()], dim=-1)
                scores[:, step: step + 1] = lprobs.transpose(0, 1)
                final_scores = scores[:, step]
                for i, string in enumerate(tokens_string):
                    symbol = ''
                    if int(tokens[i, prev_len + step]) != self.dictionary.eos() and \
                            int(tokens[i, prev_len + step]) != self.dictionary.pad():
                        symbol = self.dictionary[int(tokens[i, prev_len + step])]
                    tokens_string[i] = add_token_to_string(string, symbol)
                continue
            if step == 1:
                # bsz * beam, L
                split_size = int(self.model.split_size)
                encoder_out['src_tokens'] = encoder_out['src_tokens']. \
                    repeat(split_size, 1)
                encoder_out['encoder_out'] = (
                    encoder_out['encoder_out'][0].repeat(split_size, 1, 1),
                    encoder_out['encoder_out'][1].repeat(split_size, 1, 1),
                )
                encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask']. \
                    repeat(split_size, 1)

            logits = self.model.decode(
                prev_tokens_index,
                encoder_out,
                tokens[:, :prev_len + step],
            )
            logits[:, self.dictionary.pad()] = -math.inf

            if identifiers is not None:
                for i in range(tokens.size(0)):
                    src_length = int(sample['src_statement_length'][0])
                    cur_length = len(tokens_string[i].strip().split())
                    if cur_length > src_length:
                        gap = max(-50, src_length - cur_length)
                    else:
                        gap = max(-50, cur_length - src_length)
                    if cur_length < src_length:
                        logits[i, self.dictionary.eos()] += length_penalty[gap]
                    else:
                        logits[i, self.dictionary.eos()] -= length_penalty[gap]

                    if not identifiers[0]['text']:
                        continue

                    prefixs, texts = BeamSearch.get_prefix(
                        tokens[i][prev_len: prev_len + step],
                        self.dictionary
                    )
                    prefixs.reverse()
                    texts.reverse()

                    tmp = None
                    for prefix, text in zip(prefixs, texts):
                        if prefix in identifiers[0]['tokens']:
                            if prefix != '' and prefix[-5:] != 'CaMeL' and prefix[-1] != '_' and \
                                    text in identifiers[0]['text']:
                                tmp = set(identifiers[0]['tokens'][prefix].tolist()) | \
                                      set(identifiers[0]['tokens'][''].tolist())
                                tmp = torch.LongTensor(list(tmp))
                            else:
                                tmp = identifiers[0]['tokens'][prefix]
                            break
                        if text in identifiers[0]['text']:
                            break
                    if tmp is None:
                        prefix = ''
                        tmp = identifiers[0]['tokens'][prefix]
                    logits[i, tmp] += 100

            lprobs, indices = logits.topk(k=self.beam_size, dim=1)  # beam, beam
            lprobs = lprobs.t().contiguous().view(-1)  # beam x beam
            indices = indices.t().contiguous().view(-1)
            cand_final_scores = lprobs + final_scores.repeat(self.beam_size)
            cand_final_scores, sort_order = cand_final_scores.sort(descending=True)
            lprobs = lprobs.index_select(0, sort_order)
            indices = indices.index_select(0, sort_order)

            # choose finished beam
            eos_mask = indices[: self.beam_size].eq(self.dictionary.eos())
            eos_lprobs = lprobs[: self.beam_size].masked_select(mask=eos_mask)  # N_eos
            eos_cand_final_scores = cand_final_scores[: self.beam_size].masked_select(mask=eos_mask)  # N_eos
            eos_sort_order = sort_order[: self.beam_size].masked_select(mask=eos_mask)  # N_eos
            if eos_cand_final_scores.size(0) > 0:
                eos_beam_ids = eos_sort_order % self.beam_size  # N_eos
                eos_beam_ids = eos_beam_ids.long()
                eos_beam = tokens[eos_beam_ids, prev_len:]  # N_eos, L
                eos_beam[:, step] = self.dictionary.eos()
                eos_beam_scores = scores[eos_beam_ids, :]
                eos_beam_scores[:, step] = eos_lprobs
                for i in range(eos_beam.size(0)):
                    hypothesis.append(
                        {
                            'hypo': eos_beam[i, : step + 1],
                            'score': eos_beam_scores[i, 1: step + 1],
                            'final_score': float(eos_cand_final_scores[i]) / (1 + step),
                        })
                if len(hypothesis) >= self.beam_size:
                    hypothesis = hypothesis[: self.beam_size]
                    break

            # choose next beam
            cand_mask = ~indices.eq(self.dictionary.eos())
            cand_final_scores = cand_final_scores.masked_select(mask=cand_mask)[: self.beam_size]
            sort_order = sort_order.masked_select(mask=cand_mask)[: self.beam_size]
            lprobs = lprobs.masked_select(mask=cand_mask)[: self.beam_size]
            indices = indices.masked_select(mask=cand_mask)[: self.beam_size]
            cand_beam_ids = sort_order % self.beam_size
            cand_beam_ids = cand_beam_ids.long()

            new_tokens_string = []
            for i in range(cand_beam_ids.size(0)):
                symbol = ''
                if int(indices[i]) not in [self.dictionary.eos(), self.dictionary.pad()]:
                    symbol = self.dictionary[int(indices[i])]
                new_tokens_string.append(
                    add_token_to_string(tokens_string[int(cand_beam_ids[i])], symbol)
                )
            tokens_string = new_tokens_string

            tokens = tokens[cand_beam_ids, :]
            tokens[:, prev_len + step] = indices

            scores = scores[cand_beam_ids, :]
            scores[:, step] = lprobs
            final_scores = cand_final_scores
            prev_tokens_index = torch.cat([prev_tokens_index,
                                           torch.ones(self.beam_size, 1).long()], dim=-1)

        if len(hypothesis) < self.beam_size:
            current_num = len(hypothesis)
            for i in range(self.beam_size - current_num):
                tokens[i, -1] = self.dictionary.eos()
                hypothesis.append({
                    'hypo': tokens[i],
                    'score': scores[i, 1:],
                    'final_score': float(final_scores[i]) / (self.max_step - prev_len),
                })

        hypothesis.sort(key=lambda e: e['final_score'], reverse=True)
        return hypothesis
