import torch
import torch.nn as nn
from timm.layers.config import use_fused_attn
from torch.nn import functional as F


class CrossAttentionSingleInput(nn.Module):
    """
    Cross-Attention with single input that returns both queries and context.
    Queries attend to context (image features).
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        fused_attn=True,
        query_only_residual=False,
        attend_queries=False,
    ):
        """
        Args:
            attend_queries: Draw the keys/values from ``[queries, context]``
                instead of ``context`` alone, so each query attends over the
                other queries *and* the image features in a single softmax.

                This is how ``EoMT_Original`` gets query<->query attention: its
                queries sit in the token stream, so the backbone's own qkv sees
                both. Minus the image->query direction, this reproduces it —
                with no extra parameters, unlike ``LUMA(query_self_attn=True)``,
                which bolts a Mask2Former-style decoder self-attention on top.

                The softmax is shared, so query-query mass has to be won away
                from ~1.7k patch keys; ``EoMT_Original`` lives with the same
                dilution.
            query_only_residual: Zero out the context rows of the returned tensor.

                Call sites consume the output as a *residual delta*
                (``xq = xq + dp(ls(attn(xq)))``), so returning the context
                unchanged adds the image features back onto themselves once per
                EoMT block — ``x <- x + gamma * LN(x)``, compounding with
                ``num_blocks``. One-way cross-attention is supposed to leave the
                image features alone, which means a *zero* delta, not a copy.

                Defaults to ``False``, which reproduces every run trained so far
                bit-for-bit. Set ``True`` (via ``LUMA(query_only_residual=True)``)
                to get the intended behaviour.
        """
        super().__init__()
        self.query_only_residual = query_only_residual
        self.attend_queries = attend_queries
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Separate projections for queries and keys/values
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.fused_attn = fused_attn and use_fused_attn()

    def forward(self, x, num_queries, attn_mask=None):
        """
        Args:
            x: Concatenated tensor of shape (B, N+M, C) where:
                - First N tokens are queries
                - Last M tokens are context (image features)
            num_queries: Number of query tokens (N). If None, splits evenly.
            attn_mask: Optional boolean mask of shape (B, N, M) over the
                query->context attention, following the SDPA convention where
                ``True`` means *may attend*. A row that is entirely ``False``
                would make the softmax collapse to NaN, so callers must leave at
                least one column open per query (``LUMA.build_attn_mask`` keeps
                the prefix-token columns open for exactly this reason).

        Returns:
            Output tensor of shape (B, N+M, C). The first N rows are the attended
            queries; the last M rows are the original context, or zeros when
            ``query_only_residual=True``.
        """
        B, total_tokens, C = x.shape

        if num_queries is None:
            raise ValueError("Num queries missing")

        # Split into queries and context
        queries = x[:, :num_queries, :]  # (B, N, C)
        context = x[:, num_queries:, :]  # (B, M, C)

        N = num_queries
        M = total_tokens - num_queries

        # Project queries
        q = self.q(queries).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Project keys and values from context (plus the queries themselves under
        # attend_queries, which puts query<->query attention in the same softmax)
        kv_src = x if self.attend_queries else context
        L = kv_src.shape[1]
        kv = self.kv(kv_src).reshape(B, L, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # (B, N, M) -> (B, 1, N, M) so it broadcasts over heads.
        if attn_mask is not None:
            # Callers build the mask over the context columns only; the query
            # columns prepended by attend_queries are always open.
            if self.attend_queries:
                attn_mask = torch.cat([attn_mask.new_ones(B, N, N), attn_mask], dim=2)
            attn_mask = attn_mask.unsqueeze(1)

        if self.fused_attn:
            updated_queries = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            # Cross-attention: queries attend to context
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn = attn.masked_fill(~attn_mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # Apply attention to get updated queries
            updated_queries = attn @ v

        updated_queries = updated_queries.transpose(1, 2).reshape(B, N, C)
        updated_queries = self.proj(updated_queries)
        updated_queries = self.proj_drop(updated_queries)

        # Concatenate updated queries with the context part of the residual delta
        if self.query_only_residual:
            context = torch.zeros_like(context)
        output = torch.cat([updated_queries, context], dim=1)  # (B, N+M, C)

        return output
