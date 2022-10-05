import codecs
import torch
import sys
import os
from transformers import OpenAIGPTLMHeadModel

GENERATOR_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
sys.path.append(GENERATOR_DIR + '../models/')
sys.path.append(GENERATOR_DIR + '../dataloader/')
from gpt_conut_data_loader import GPTCoNuTDataLoader
from gpt_fconv_data_loader import GPTFConvDataLoader
from identifier_data_loader import IdentifierDataLoader
from dictionary import Dictionary
from gpt_conut import GPTCoNuTModel
from gpt_fconv import GPTFConvModel
from beamsearch import BeamSearch


class Generator():
    def __init__(self, model, dictionary, data_loader, beam_size=10):
        self.model = model
        self.dictionary = dictionary
        self.data_loader = data_loader
        self.beam_size = beam_size
        self.beamsearch = BeamSearch(model, dictionary, beam_size)
        print(self.model, beam_size)

    def generate(self, output_path):
        wp = codecs.open(output_path, 'w', 'utf-8')
        self.data_loader.load_data(0, self.data_loader.total_size)
        for i in range(self.data_loader.total_size):
            print(i, '/', self.data_loader.total_size)
            data = self.data_loader.dataset[i]
            try:
                self.beamsearch.beam_size = self.beam_size
                sample = self.data_loader.dataset.collater([data])
                with torch.no_grad():
                    if isinstance(self.model, GPTCoNuTModel):
                        hypothesis = self.beamsearch.generate_gpt_conut(sample)
                    elif isinstance(self.model, GPTFConvModel):
                        hypothesis = self.beamsearch.generate_gpt_fconv(sample)
            except Exception as e:
               print(e)
               continue
            id = str(sample['id'].item())
            wp.write('S-{}\t'.format(id))
            wp.write(self.dictionary.string(data['source']) + '\n')
            wp.write('T-{}\t'.format(id))
            wp.write(self.dictionary.string(data['target']) + '\n')
            for h in hypothesis:
                wp.write('H-{}\t{}\t'.format(id, str(h['final_score'])))
                wp.write(self.dictionary.string(h['hypo']) + '\n')
                wp.write('P-{}\t'.format(id))
                wp.write(' '.join(str(round(s.item(), 4)) for s in h['score']) + '\n')
        wp.close()


def generate_gpt_conut(vocab_file, model_file, input_file, identifier_txt_file, identifier_token_file, output_file, beam_size):
    dictionary = Dictionary(vocab_file, min_cnt=0)
    print(len(dictionary))
    loaded = torch.load(
        model_file, map_location='cpu'
    )
    config = loaded['config']
    gpt_config = config['embed_model_config']
    gpt_config.attn_pdrop = 0
    gpt_config.embd_pdrop = 0
    gpt_config.resid_pdrop = 0
    gpt_model = OpenAIGPTLMHeadModel(gpt_config)
    model = GPTCoNuTModel(
        dictionary=dictionary, embed_dim=config['embed_dim'],
        max_positions=config['max_positions'],
        src_encoder_convolutions=config['src_encoder_convolutions'],
        ctx_encoder_convolutions=config['ctx_encoder_convolutions'],
        decoder_convolutions=config['decoder_convolutions'],
        dropout=0, embed_model=gpt_model,
    )

    model.load_state_dict(loaded['model'])
    identifier_loader = IdentifierDataLoader(
        dictionary, identifier_token_file, identifier_txt_file
    )
    data_loader = GPTCoNuTDataLoader(
        input_file, dictionary,
        identifier_loader=identifier_loader
    )
    generator = Generator(model, dictionary, data_loader, beam_size=beam_size)
    print('start generate')
    generator.generate(output_file)


def generate_gpt_fconv(vocab_file, model_file, input_file, identifier_txt_file, identifier_token_file, output_file, beam_size):
    dictionary = Dictionary(vocab_file, min_cnt=0)
    print(len(dictionary))
    loaded = torch.load(
        model_file, map_location='cpu'
    )
    config = loaded['config']
    gpt_config = config['embed_model_config']
    gpt_config.attn_pdrop = 0
    gpt_config.embd_pdrop = 0
    gpt_config.resid_pdrop = 0
    gpt_model = OpenAIGPTLMHeadModel(gpt_config)
    model = GPTFConvModel(
        dictionary=dictionary, embed_dim=config['embed_dim'],
        max_positions=config['max_positions'],
        encoder_convolutions=config['encoder_convolutions'],
        decoder_convolutions=config['decoder_convolutions'],
        dropout=0, embed_model=gpt_model,
    )
    model.load_state_dict(loaded['model'])
    identifier_loader = IdentifierDataLoader(
        dictionary, identifier_token_file, identifier_txt_file
    )
    data_loader = GPTFConvDataLoader(
        input_file, dictionary,
        identifier_loader=identifier_loader
    )
    generator = Generator(model, dictionary, data_loader, beam_size=beam_size)
    print('start generate')
    generator.generate(output_file)


if __name__ == "__main__":
    vocab_file = GENERATOR_DIR + '../../data/vocabulary/vocabulary.txt'
    input_file = GENERATOR_DIR + '../../candidate_patches/QuixBugs/quixbugs_bpe.txt'
    identifier_txt_file = GENERATOR_DIR + '../../candidate_patches/QuixBugs/identifier.txt'
    identifier_token_file = GENERATOR_DIR + '../../candidate_patches/QuixBugs/identifier.tokens'
    beam_size = 1000
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    model_file = GENERATOR_DIR + '../../data/models/gpt_conut_1.pt'
    output_file = GENERATOR_DIR + '../../data/patches/gpt_conut_1.txt'
    generate_gpt_conut(vocab_file, model_file, input_file, identifier_txt_file, identifier_token_file, output_file, beam_size)

    model_file = GENERATOR_DIR + '../../data/models/gpt_fconv_1.pt'
    output_file = GENERATOR_DIR + '../../data/patches/gpt_fconv_1.txt'
    generate_gpt_fconv(vocab_file, model_file, input_file, identifier_txt_file, identifier_token_file, output_file, beam_size)
