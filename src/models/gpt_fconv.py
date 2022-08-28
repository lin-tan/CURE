import torch.nn as nn

from gpt_conut import GPTFConvEncoder, GPTFConvDecoder


class GPTFConvModel(nn.Module):
    def __init__(
            self, dictionary, embed_dim=384, max_positions=1024,
            encoder_convolutions=((192, 5),) * 5,
            decoder_convolutions=((192, 5),) * 5,
            dropout=0.1, embed_model=None,
    ):
        super(GPTFConvModel, self).__init__()
        self.embed_dim = embed_dim
        self.max_positions = max_positions
        self.dictionary = dictionary
        self.encoder_convolutions = encoder_convolutions
        self.decoder_convolutions = decoder_convolutions
        self.embed_model = embed_model

        self.encoder = GPTFConvEncoder(
            dictionary, embed_dim, max_positions,
            encoder_convolutions, dropout,
        )
        self.decoder = GPTFConvDecoder(
            dictionary, embed_dim, max_positions,
            decoder_convolutions, dropout,
        )

    def config(self):
        info = dict()
        info['embed_dim'] = self.embed_dim
        info['max_positions'] = self.max_positions
        info['encoder_convolutions'] = self.encoder_convolutions
        info['decoder_convolutions'] = self.decoder_convolutions
        info['embed_model_config'] = self.embed_model.config
        return info

    def forward(self, src_tokens, src_tokens_with_pre_context,
                prev_tokens_index, prev_tokens_with_context=None, labels=None):
        encoder_out = self.encoder(
            src_tokens, src_tokens_with_pre_context,
            share_embed_model=self.embed_model
        )
        decoder_out = self.decoder(
            prev_tokens_index, encoder_out,
            prev_tokens_with_context,
            share_embed_model=self.embed_model,
            output_lm_logits=True,
        )

        if labels is not None:
            logits, avg_attn_scores, lm_logits = decoder_out
            loss_fct = nn.NLLLoss()
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., :].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            shift_lm_logits = lm_logits[..., :-2, :].contiguous()
            shift_labels = prev_tokens_with_context[:, 1:-1].contiguous()
            lm_loss = loss_fct(shift_lm_logits.view(-1, shift_lm_logits.size(-1)), shift_labels.view(-1))

            decoder_out = (logits, avg_attn_scores, loss, lm_loss)

        return decoder_out
