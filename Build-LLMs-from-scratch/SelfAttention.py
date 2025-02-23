import torch


class SelfAttention_v1(torch.nn.Module):
    def __init__(self, dim_in, dim_out, scale=False):
        super().__init__()
        self.W_q = torch.nn.Parameter(torch.rand(dim_in, dim_out))  # (n, d)
        self.W_k = torch.nn.Parameter(torch.rand(dim_in, dim_out))
        self.W_v = torch.nn.Parameter(torch.rand(dim_in, dim_out))
        self.scale = scale

    def forward(self, X):
        queries = X @ self.W_q  # (m, n) @ (n, d) -> (m, d)
        keys = X @ self.W_k
        values = X @ self.W_v

        attention_scores = queries @ keys.T  # (m, d) @ (d, m) -> (m, m)

        if self.scale:
            d = keys.shape[-1]
            attention_scores /= d**0.5
        attention_weights = torch.nn.functional.softmax(
            attention_scores, dim=-1
        )  # (m, m)

        context_vec = attention_weights @ values  # (m, m) @ (m, d) -> (m, d)
        return context_vec


class SelfAttention_v2(torch.nn.Module):
    """
    This version uses `torch.nn.Linear` to define the weights. This enables optimized random initialization of weights.
    `bias` is set to `False` for Linear layers to effectively perform matrix multiplication.
    """

    def __init__(self, dim_in, dim_out, scale=False, qkv_bias=False):
        super().__init__()
        self.W_q = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_k = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_v = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.scale = scale

    def forward(self, X):
        queries = self.W_q(X)
        keys = self.W_k(X)
        values = self.W_v(X)

        attention_scores = queries @ keys.T

        if self.scale:
            d = keys.shape[-1]
            attention_scores /= d**0.5
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        context_vec = attention_weights @ values
        return context_vec


class MaskedAttention(torch.nn.Module):
    """
    This is a masked version of the self-attention mechanism.
    The mask is applied to the attention scores to prevent the model from attending to future tokens.
    """

    def __init__(
        self, dim_in, dim_out, context_len, dropout, scale=False, qkv_bias=False
    ):
        super().__init__()
        self.W_q = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_k = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_v = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.scale = scale

        # Ensures that the tensors are in the same device (CPU or GPU) as the model. Useful during training.
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )

    def forward(self, X):
        num_batches, num_tokens, dim_in = X.shape
        queries = self.W_q(X)
        keys = self.W_k(X)
        values = self.W_v(X)

        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores = attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )  # trailing `_` means in-place operation.

        if self.scale:
            d = keys.shape[-1]
            attention_scores /= d**0.5

        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context_vec = attention_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        context_len,
        num_heads,
        dropout,
        scale=False,
        qkv_bias=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.heads = torch.nn.ModuleList(
            [
                MaskedAttention(dim_in, dim_out, context_len, dropout, scale, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, X):
        return torch.cat([head(X) for head in self.heads], dim=-1)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout, num_heads, scale=False, qkv_bias=False):
        super().__init__()
        assert dim_out % num_heads == 0, "dim_out must be divisible by num_heads"

        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads  # Determines the split size for input to each head.
        self.W_query = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(dim_out, dim_out)  # A linear layer than combines the outputs of all heads.
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, X):
        batch_size, num_tokens, dim_in = X.shape
        keys = self.W_key(X)  # (batch_size, num_tokens, dim_out)
        queries = self.W_query(X)
        values = self.W_value(X)

        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)  # Implicit split of `dim_out` into `num_heads`. (batch_size, num_tokens, num_heads, head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)  # Swaps `num_tokens` and `num_heads` dimensions for computations using transpose operation.
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)  # Transposes the last two dimensions(num_heads and head_dim) of `keys` for matrix multiplication.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        if self.scale:
            attn_scores /= keys.shape[-1] **0.5
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)  # Transposes back to original shape. (batch_size, num_tokens, num_heads, head_dim)
        # Combines the outputs of all heads.
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.dim_out)
        context_vec = self.out_proj(context_vec)  # Linear layer to combine the outputs of all heads.
        return context_vec
