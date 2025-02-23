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
