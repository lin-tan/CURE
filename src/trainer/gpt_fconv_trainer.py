import json
import os
import sys
import time
import codecs
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import OpenAIGPTLMHeadModel

GPT_FCONV_TRAINER_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
sys.path.append(GPT_FCONV_TRAINER_DIR + '../models/')
sys.path.append(GPT_FCONV_TRAINER_DIR + '../dataloader/')
from gpt_fconv import GPTFConvModel
from dictionary import Dictionary
from gpt_fconv_data_loader import GPTFConvDataLoader


class GPTFConvTrainer():
    def __init__(self, train_loader, valid_loader, dictionary, gpt_file):
        gpt_loaded = torch.load(gpt_file)
        config = gpt_loaded['config']
        gpt_model = OpenAIGPTLMHeadModel(config).cuda()
        gpt_model.load_state_dict(gpt_loaded['model'])

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.dictionary = dictionary

        self.batch_size = 12
        self.load_size = 1200

        self.gpt_model = gpt_model
        self.model = None
        self.hyper_parameter = {}
        self.hyper_parameter_set = {'{}'}
        self.optimizer = None
        self.current_train_step = 0
        self.val_loss = {}

    def shuffle_dataset(self):
        indices = [i for i in range(len(self.train_loader.dataset))]
        random.shuffle(indices)
        return indices

    def train_step(self, samples):
        self.model.train()
        self.current_train_step += 1
        self.optimizer.zero_grad()

        batch = self.train_loader.dataset.collater(samples)
        if torch.cuda.is_available():
            outputs = self.model(
                batch['net_input']['src_tokens'].cuda(),
                batch['net_input']['src_with_prev_context'].cuda(),
                prev_tokens_index=batch['target_index'].cuda(),
                prev_tokens_with_context=batch['target_with_prev_context'].cuda(),
                labels=batch['target'].cuda(),
            )

        logits, avg_attn_scores, apr_loss, lm_loss = outputs[:4]
        loss = apr_loss + 0.3 * lm_loss
        loss.mean().backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5, norm_type=2)
        self.optimizer.step()
        return loss.mean().item(), apr_loss.mean().item(), lm_loss.mean().item()

    def valid_step(self, samples):
        self.model.eval()
        batch = self.valid_loader.dataset.collater(samples)
        outputs = self.model(
            batch['net_input']['src_tokens'].cuda(),
            batch['net_input']['src_with_prev_context'].cuda(),
            prev_tokens_index=batch['target_index'].cuda(),
            prev_tokens_with_context=batch['target_with_prev_context'].cuda(),
            labels=batch['target'].cuda(),
        )
        logits, avg_attn_scores, apr_loss, lm_loss = outputs[:4]
        loss = apr_loss + 0.3 * lm_loss
        return loss.mean().item(), apr_loss.mean().item(), lm_loss.mean().item(), logits

    def validate_and_save(self, model_id, save_dir):
        oom = 0
        with torch.no_grad():
            val_loss, val_fconv_loss, val_lm_loss = [], [], []
            for i in range(0, self.valid_loader.total_size, self.batch_size):
                samples = [self.valid_loader.dataset[j]
                           for j in range(i, min(len(self.valid_loader.dataset), i + self.batch_size))]
                try:
                    loss, fconv_loss, lm_loss, logits = self.valid_step(samples)
                    val_loss.append(float(loss))
                    val_fconv_loss.append(float(fconv_loss))
                    val_lm_loss.append(float(lm_loss))
                except Exception as e:
                    oom += 1

            info = 'val loss:{}, val apr_loss:{}, val lm_loss:{}, val ppl:{}, oom:{}'.format(
                round(float(np.mean(val_loss)), 6),
                round(float(np.mean(val_fconv_loss)), 6),
                round(float(np.mean(val_lm_loss)), 6),
                round(float(np.exp(np.mean(val_loss))), 6),
                oom
            )
            print(info)

            val_loss = np.mean(val_fconv_loss)
            checkpoint = {
                'model': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'current_step': self.current_train_step,
                'config': self.model.module.config(),
                'val_loss': val_loss,
            }
            torch.save(checkpoint, save_dir + 'gpt_fconv_' + str(model_id) + '.pt')
            self.val_loss[model_id] = {
                'val_loss': val_loss,
                'hyper-parameter': str(self.hyper_parameter),
            }
        return val_loss

    def train(self, model_id, epochs, hyper_parameter, save_dir):
        self.hyper_parameter = hyper_parameter
        self.model = GPTFConvModel(
                self.dictionary, embed_dim=384, max_positions=1024,
                encoder_convolutions=self.hyper_parameter['encoder_convolutions'],
                decoder_convolutions=self.hyper_parameter['decoder_convolutions'],
                dropout=self.hyper_parameter['dropout'], embed_model=self.gpt_model,
            ).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=6.25e-5)
        self.model = nn.DataParallel(self.model, device_ids=device_ids)
        
        self.valid_loader.load_data(0, self.valid_loader.total_size)
        for epoch in range(epochs):
            start_time = time.time()
            for i in range(0, self.train_loader.total_size, self.load_size):
                oom = 0
                self.train_loader.load_data(i, i + self.load_size)
                indices = self.shuffle_dataset()
                train_loss, train_apr_loss, train_lm_loss = [], [], []

                start, end = 0, 0
                samples = []
                max_src, max_ctx, max_tgt = 0, 0, 0
                while end < len(self.train_loader.dataset):
                    sample = self.train_loader.dataset[indices[end]]
                    if max_ctx + len(sample['target']) >= 1023 \
                            or max_tgt + len(sample['prev_context']) >= 1023 \
                            or max_ctx + len(sample['source']) >= 1023 \
                            or max_src + len(sample['prev_context']) >= 1023 \
                            or end - start == self.batch_size:
                        try:
                            loss, apr_loss, lm_loss = self.train_step(samples)
                            train_loss.append(loss)
                            train_apr_loss.append(apr_loss)
                            train_lm_loss.append(lm_loss)
                        except Exception as e:
                            oom += 1

                        start = end
                        max_src, max_ctx, max_tgt = 0, 0, 0
                        samples = []
                        continue
                    max_src = max(max_src, len(sample['source']))
                    max_ctx = max(max_ctx, len(sample['prev_context']))
                    max_tgt = max(max_tgt, len(sample['target']))
                    end += 1
                    samples.append(sample)
                if len(samples) > 0:
                    try:
                        loss, apr_loss, lm_loss = self.train_step(samples)
                        train_loss.append(loss)
                        train_apr_loss.append(apr_loss)
                        train_lm_loss.append(lm_loss)
                    except Exception as e:
                        oom += 1

                if (i // self.load_size) % 10 == 0:
                    info = 'epoch:{}, load data:{}, lr:{}, loss:{}, apr_loss:{}, lm_loss:{}, time:{}s, oom:{}'.\
                        format(epoch + 1, i + self.load_size,
                               round(self.optimizer.param_groups[0]['lr'], 10),
                               round(float(np.mean(train_loss)), 6),
                               round(float(np.mean(train_apr_loss)), 6),
                               round(float(np.mean(train_lm_loss)), 6),
                               int(time.time() - start_time), oom
                               )
                    start_time = time.time()
                    print(str(model_id) + ' ' + info)

                if (i // self.load_size) % 100 == 0:
                    self.validate_and_save(model_id, save_dir)
        self.validate_and_save(model_id, save_dir)


if __name__ == '__main__':
    device_ids = [0, 1, 2, 3]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    vocab_file = GPT_FCONV_TRAINER_DIR + '../../data/vocabulary/vocabulary.txt'
    train_file = GPT_FCONV_TRAINER_DIR + '../../data/data/training_bpe.txt'
    valid_file = GPT_FCONV_TRAINER_DIR + '../../data/data/validation_bpe.txt'
    gpt_file = GPT_FCONV_TRAINER_DIR + '../../data/models/code_gpt.pt'

    dictionary = Dictionary(vocab_file, min_cnt=0)
    print('dictionary initialized, vocab size:{}'.format(len(dictionary)))

    train_loader = GPTFConvDataLoader(train_file, dictionary)
    valid_loader = GPTFConvDataLoader(valid_file, dictionary)
    print('data loader initialized, train size:{}, validate size:{}'.
          format(train_loader.total_size, valid_loader.total_size))

    trainer = GPTFConvTrainer(train_loader, valid_loader, dictionary, gpt_file)

    hyper_parameter = {
        'encoder_convolutions': ((192, 5),) * 1,
        'decoder_convolutions': ((192, 5),) * 1,
        'dropout': 0.1,
    }
    trainer.train(1, 2, hyper_parameter, save_dir=GPT_FCONV_TRAINER_DIR + '../../data/models/')
