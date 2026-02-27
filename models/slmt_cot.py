import torch
from torch import nn
from .slmt_layer import Transformer, CrossTransformer, PreNormForward, PreNormAttention, FeedForward
from .bert import BertTextEncoder
from einops import repeat, rearrange
import torch.nn.functional as F


class CoTAttention(nn.Module):
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2*dim, 2*dim//factor, 1, bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor, kernel_size*kernel_size*dim, 1)
        )

    def forward(self, x):
        bs, c, h, w = x.shape

        k1 = self.key_embed(x)  
        v = self.value_embed(x).view(bs, c, -1) 

        y = torch.cat([k1, x], dim=1) 

        att = self.attention_embed(y) 
        att = att.reshape(bs, c, self.kernel_size*self.kernel_size, h, w)  

        att = att.mean(2, keepdim=False).view(bs, c, -1)  
        k2 = F.softmax(att, dim=-1) * v  
        k2 = k2.view(bs, c, h, w)

        return k1 + k2



class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel//reduction, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.shape
        max_result = self.maxpool(x).view(b, c) 
        avg_result = self.avgpool(x).view(b, c) 
        max_out = self.se(max_result).unsqueeze(-1).unsqueeze(-1) 
        avg_out = self.se(avg_result).unsqueeze(-1).unsqueeze(-1) 
        output = self.sigmoid(max_out + avg_out) 
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)  
        avg_result = torch.mean(x, dim=1, keepdim=True)    
        result = torch.cat([max_result, avg_result], 1)   
        output = self.conv(result)                         
        output = self.sigmoid(output)                     
        return output


class cscBlock(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channel=channel, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        residual = x
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out + residual


class cscForTransformer(nn.Module):
   
    def __init__(self, dim, reduction=16, spatial_kernel_size=7, dropout=0.):
        super().__init__()
        self.dim = dim
        
       
        self.csc = cscBlock(channel=dim, reduction=reduction, kernel_size=spatial_kernel_size)
        
        
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        
        b, n, d = x.shape
        
       
        x = x.transpose(1, 2).contiguous()  # [b, d, n]
        
       
        h = int(n**0.5)
        w = n // h
        if h * w < n:
            w += 1
            
            padding = h * w - n
            x = F.pad(x, (0, padding))  # [b, d, n+padding]
        
        x = x.view(b, d, h, w) 
        
       
        x = self.csc(x)  
        
        
        x = x.view(b, d, -1) 
        x = x[:, :, :n]  # 移除填充 [b, d, n]
        x = x.transpose(1, 2)  # [b, n, d]
        
        return self.to_out(x)


class CoTAttentionForTransformer(nn.Module):
  
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., kernel_size=3):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        
      
        self.cot_attention = CoTAttention(dim=dim, kernel_size=kernel_size)
        
      
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, q, k, v):
       
        b, n, d = q.shape
        
       
        x = q.transpose(1, 2).contiguous()  
        
       
        h = int(n**0.5)
        w = n // h
        if h * w < n:
            w += 1
           
            padding = h * w - n
            x = F.pad(x, (0, padding)) 
        
        x = x.view(b, d, h, w)  
        
        
        x = self.cot_attention(x)

    
        x = x.view(b, d, -1)  
        x = x[:, :, :n]  
        x = x.transpose(1, 2)  
        
        return self.to_out(x)


class PreNormCoTAttention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        return self.fn(q, k, v)


class CoTTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormCoTAttention(dim, CoTAttentionForTransformer(dim, heads=heads, dim_head=dim_head, dropout=dropout, kernel_size=kernel_size)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, save_hidden=False):
        if save_hidden:
            hidden_list = []
            hidden_list.append(x)
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
                hidden_list.append(x)
            return hidden_list
        else:
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
            return x


class CoTHhyperLearningLayer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., kernel_size=3):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.kernel_size = kernel_size

      
        self.cot_text = CoTAttentionForTransformer(dim, heads=heads, dim_head=dim_head, dropout=dropout, kernel_size=kernel_size)
        
        
        self.csc_vision = cscForTransformer(dim, reduction=16, spatial_kernel_size=7, dropout=dropout)
        
        
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_ta = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_tv = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_ta = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_tv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=True),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, h_t, h_a, h_v, h_hyper):
        b, n, _, h = *h_t.shape, self.heads

       
        q = self.to_q(h_t)
        
        
        k_ta = self.to_k_ta(h_a)
        v_ta = self.to_v_ta(h_a)

        q, k_ta, v_ta = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k_ta, v_ta))

        
        dots_ta = torch.einsum('b h i d, b h j d -> b h i j', q, k_ta) * self.scale
        attn_ta = self.attend(dots_ta)
        out_ta = torch.einsum('b h i j, b h j d -> b h i d', attn_ta, v_ta)
        out_ta = rearrange(out_ta, 'b h n d -> b n (h d)')
        
        
        out_tv = self.csc_vision(h_v)

        
        h_hyper_shift = self.to_out(out_ta + out_tv)
        h_hyper += h_hyper_shift

        return h_hyper


class PreNormCoTAHL(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, h_t, h_a, h_v, h_hyper):
        h_t = self.norm1(h_t)
        h_a = self.norm2(h_a)
        h_v = self.norm3(h_v)
        h_hyper = self.norm4(h_hyper)

        return self.fn(h_t, h_a, h_v, h_hyper)


class CoTHhyperLearningEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0., kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormCoTAHL(dim, CoTHhyperLearningLayer(dim, heads=heads, dim_head=dim_head, dropout=dropout, kernel_size=kernel_size))
            ]))

    def forward(self, h_t_list, h_a, h_v, h_hyper):
        for i, attn in enumerate(self.layers):
            h_hyper = attn[0](h_t_list[i], h_a, h_v, h_hyper)
        return h_hyper


class slmt_CoT(nn.Module):
    def __init__(self, args):
        super(slmt_CoT, self).__init__()

        args = args.model

        self.h_hyper = nn.Parameter(torch.ones(1, args.token_len, args.token_dim))

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=args.bert_pretrained)

        self.proj_l = nn.Sequential(
            nn.Linear(args.l_input_dim, args.l_proj_dst_dim),
            Transformer(num_frames=args.l_input_length, save_hidden=False, token_len=args.token_length, dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads, mlp_dim=args.proj_mlp_dim)
        )
        self.proj_a = nn.Sequential(
            nn.Linear(args.a_input_dim, args.a_proj_dst_dim),
            Transformer(num_frames=args.a_input_length, save_hidden=False, token_len=args.token_length, dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads, mlp_dim=args.proj_mlp_dim)
        )
        self.proj_v = nn.Sequential(
            nn.Linear(args.v_input_dim, args.v_proj_dst_dim),
            Transformer(num_frames=args.v_input_length, save_hidden=False, token_len=args.token_length, dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads, mlp_dim=args.proj_mlp_dim)
        )

        
        self.l_encoder = CoTTransformerEncoder(
            dim=args.proj_input_dim, 
            depth=args.AHL_depth-1, 
            heads=args.l_enc_heads, 
            dim_head=args.proj_input_dim // args.l_enc_heads, 
            mlp_dim=args.l_enc_mlp_dim,
            kernel_size=3 
        )
        
        
        self.h_hyper_layer = CoTHhyperLearningEncoder(
            dim=args.token_dim, 
            depth=args.AHL_depth, 
            heads=args.ahl_heads, 
            dim_head=args.ahl_dim_head, 
            dropout=args.ahl_droup,
            kernel_size=3 
        )
        
        self.fusion_layer = CrossTransformer(
            source_num_frames=args.token_len, 
            tgt_num_frames=args.token_len, 
            dim=args.proj_input_dim, 
            depth=args.fusion_layer_depth, 
            heads=args.fusion_heads, 
            mlp_dim=args.fusion_mlp_dim
        )

        self.regression_layer = nn.Sequential(
            nn.Linear(args.token_dim, 1)
        )

    def forward(self, x_visual, x_audio, x_text):
        b = x_visual.size(0)

        h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b=b)

        x_text = self.bertmodel(x_text)

        h_v = self.proj_v(x_visual)[:, :self.h_hyper.shape[1]]
        h_a = self.proj_a(x_audio)[:, :self.h_hyper.shape[1]]
        h_l = self.proj_l(x_text)[:, :self.h_hyper.shape[1]]

       
        h_t_list = self.l_encoder(h_l, save_hidden=True)
        
       
        h_hyper = self.h_hyper_layer(h_t_list, h_a, h_v, h_hyper)
        
        feat = self.fusion_layer(h_hyper, h_t_list[-1])[:, 0]

        output = self.regression_layer(feat)

        return output


def build_model(args):
    model = slmt_CoT(args)
    return model 