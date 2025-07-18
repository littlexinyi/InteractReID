import os
import torch
import torch.nn.functional as F
from torch import nn
from .base_transformer import Transformer, LayerNorm
from text_utils.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

class TextTransformer(nn.Module):
    def __init__(self, config,
                 embed_dim: int,
                 context_length: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 positional_embedding_flag: bool,
                 checkpoint: bool,
                 bpe_path=None,
                 ):
        super().__init__()
        self.config = config
        self.context_length = context_length
        self.positional_embedding_flag = positional_embedding_flag
        self.end_id = config.model.vocab_size - 1
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            checkpoint=checkpoint,
            dropout=config.experiment.dropout
        )
        self.token_embedding = nn.Embedding(49408, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.normal(mean=0, std=0.02, size=(self.context_length, transformer_width)))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.initialize_parameters()

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            # nn.init.normal_(self.text_projection.weight, std=self.transformer.width ** -0.5)  # todo
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @property
    def dtype(self):
        return self.positional_embedding.dtype

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, texts, mask_type=None, return_dense=False):
        if mask_type is not None:
            texts, labels = texts
        x = self.token_embedding(texts).type(self.dtype)  # [batch_size, n_ctx, d_model]
        if self.positional_embedding_flag:
            x = x + self.positional_embedding.type(self.dtype)  # Fix!!!
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x @ self.text_projection

        if mask_type is not None or return_dense:
            words_feat = x

        x = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)]

        if mask_type is not None:
            return x, words_feat, labels

        if return_dense:
            return x, words_feat

        return x

    def encode_prompt_token(self, prompt, token):
        b_size = token.shape[0]
        x = self.token_embedding(prompt).type(self.dtype)  # [batch_size, n_ctx, d_model]
        
        collect_ind = prompt == self.end_id               #[bz, 77]       #True False数组
        collect_ind = collect_ind.nonzero()[:, 1]          #返回非0元素的下标，并把[bz, 1] 二维数组取其最后一维，得到[bz]，为bz内每一个样本的最大值下标

        id_split = _tokenizer.encode("*")[0]    #type = int
        ind_insert = prompt[0] == id_split       #找到编码为 *的地方插入img_token
        ind_insert = ind_insert.nonzero()[0]

        token = token.view(b_size, 1, -1)     #[bz, 1, dim]
        x = torch.cat([x[:, :ind_insert], token, x[:, ind_insert+1:]], dim=1)
        
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)    
        x = x[torch.arange(x.size(0)), collect_ind] @ self.text_projection
        return x        


def text_transformers(config):
    model_config = config.model
    kwargs = {
        'context_length': config.experiment.text_length,
        'transformer_width': 512,
        'transformer_heads': 8,
        'transformer_layers': 12,
        'positional_embedding_flag': True,
        'checkpoint': False,
        'embed_dim': model_config.embed_dim,
    }
    model = TextTransformer(config, **kwargs)
    return model
