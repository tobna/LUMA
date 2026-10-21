# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cross_attention import CrossAttentionSingleInput
from models.attention import Attention
from models.layer_scale import LayerScale
from models.drop_path import DropPath
import math

from models.scale_block import ScaleBlock
import warnings


class LUMA(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes,
        num_q,
        num_blocks=4,
        self_a=False,
        masked_attn_enabled=True,
        query_mlp: str = "shared",
        query_mlp_ratio: float = 4.0,
        use_layer_scale=True,
        query_only_residual: bool = False,
        masked_cross_attn: bool = False,
        query_self_attn: bool = False,
        attend_queries: bool = False,
        layer_scale_init: float = 1.0,
        cross_attn_encoder_norm: bool = True,
        query_pos_embed: bool = False,
        thread_queries: bool = False,
        final_query_norm: bool = False,
        predict_q_norm: bool = False,
    ):
        """
        Args:
            query_mlp: How to apply the per-block MLP update to the queries
                inside the EoMT blocks (currently honoured by the ViT backbone
                only — other wrappers use the legacy shared behaviour).

                - ``"shared"`` (default): reuse the backbone block's ``norm2``
                  + ``mlp`` for the query update. **Identical to the previous
                  behaviour.**
                - ``"separate"``: use a freshly initialised LayerNorm + MLP per
                  EoMT block (one module per block in the last ``num_blocks``
                  blocks). Hidden dim is ``embed_dim * query_mlp_ratio``.
                - ``"none"``: skip the query MLP update entirely.
            query_mlp_ratio: MLP hidden-to-embed ratio for ``query_mlp="separate"``.
                Ignored otherwise. Defaults to 4.0 (standard ViT MLP ratio).
            query_only_residual: Stop the cross-attention blocks from writing the
                image features back onto themselves. ``False`` (default)
                reproduces every run trained so far; ``True`` is the intended
                one-way behaviour. See ``CrossAttentionSingleInput``. Ignored when
                ``self_a=True``, where the self-attention legitimately updates
                both streams.
            masked_cross_attn: Restrict each query to attend only to the context
                positions its own mask prediction from the previous block called
                foreground — the mechanism from ``EoMT_Original``, which
                ``models.luma.LUMA`` has never implemented. ``attn_mask_probs``
                anneals the constraint away exactly as it does there, so once
                annealing completes the mask is a no-op and inference is a plain
                ViT again.

                **Not** the same knob as ``masked_attn_enabled`` above, which
                only controls deep-supervision key naming via ``block_postfix``.
                Defaults to ``False``. Honoured by **all 21 wrappers** since
                2026-08-17 — the mask is built in ``predict_and_mask`` from the
                same encoder grid ``_predict`` already reshapes onto, and applied
                inside ``CrossAttentionSingleInput``, so nothing about it is
                architecture-specific. Requires ``self_a=False``, since
                ``Attention`` takes no mask argument.
            query_self_attn: Let the queries attend to *each other* once per EoMT
                block. ``CrossAttentionSingleInput`` is one-way — queries attend
                to image features and nothing else — and the backbone's own
                self-attention runs on the image tokens alone
                (``models/vit.py``), so by default the queries in this
                formulation can never see one another and nothing suppresses
                duplicate detections. ``EoMT_Original`` gets query<->query
                attention for free because its queries sit inside the token
                stream (``models/eomt_original.py:186``). This is the main
                structural thing the decoupled formulation dropped.

                Ordering matches Mask2Former's decoder layer: cross-attention,
                then query self-attention, then the MLP. Defaults to ``False``;
                honoured by the ViT backbone only.
            attend_queries: The parameter-free alternative to
                ``query_self_attn``: keep one attention op, but let its
                keys/values come from ``[queries, image features]`` instead of
                the image features alone. Each query then sees the other
                queries in the same softmax — which is exactly what
                ``EoMT_Original`` gets from having its queries inside the token
                stream, minus the image->query direction.

                Lives inside ``CrossAttentionSingleInput``, so unlike
                ``query_self_attn`` **all 20 backbone wrappers honour it**, and
                it adds no parameters — hence no ``find_unused_parameters``
                exposure at ``num_blocks > 1``. Requires ``self_a=False``
                (``Attention`` already attends over the whole concatenated
                stream). Defaults to ``False``.
            layer_scale_init: ``gamma`` init for the LayerScale on the
                cross-attention residual. Defaults to ``1.0``, which is what
                every run so far trained with (``LayerScale`` used to hardcode
                ones and ignore its own ``init_values``). A small value such as
                ``1e-5`` starts each EoMT block as a near-identity graft onto
                the pretrained stream, which is the usual way to attach a
                randomly initialised module to a pretrained backbone.
            cross_attn_encoder_norm: Whether to run the cross-attention input
                through the encoder's *final* norm. ``True`` (default)
                reproduces every run so far, but it is a double normalisation:
                the wrapper already applied ``block.norm1`` to the concatenated
                stream, and ``backbone.norm``/``fc_norm`` is a norm calibrated
                for the end of the stack being applied mid-stack. 19 of the 20
                backbone wrappers do this; ``models/next_vit.py:183`` is the
                sole one that passes the tokens through raw. Set ``False`` to
                match next_vit. Honoured by the ViT backbone only.
            query_pos_embed: Add a learned per-EoMT-block positional embedding
                to the queries before each cross-attention, the way Mask2Former
                re-injects its query embedding at every decoder layer. Without
                it ``self.q`` is injected once and the blocks have no way to
                tell "which query am I" apart from the content itself, which
                only bites at ``num_blocks > 1``. Zero-initialised, so it starts
                as an exact identity. Defaults to ``False``; honoured by the ViT
                backbone only.
            thread_queries: Feed each EoMT block the *previous* block's updated
                queries instead of the raw ``self.q`` parameter.

                ``forward`` passes ``self.q`` to every block and overwrites the
                returned ``q`` on the next iteration, so with ``num_blocks > 1``
                every block but the last throws its query update away. The
                blocks run in parallel off a shared query init rather than
                refining one set of queries — there is no iterative refinement,
                which is the only reason to want more than one block. It also
                makes ``attn[:-1]`` and ``ls_list[:-1]`` receive exactly-zero
                gradients (autograd still reaches them through the pre-slice
                add, so DDP does not complain) and leaves ``q_self_attn[:-1]``
                with no gradient path at all.

                ``EoMT_Original`` threads its queries for free by keeping them
                in the token stream. Defaults to ``False`` so every checkpoint
                trained before 2026-08-13 still reproduces; honoured by the ViT
                backbone only. No-op at ``num_blocks=1``.
            final_query_norm: Run the queries through ``encoder.norm`` before the
                final ``_predict``, so the shared class/mask heads see the same
                normalisation at every prediction.

                Without it the heads are fed two different distributions. Every
                deep-supervision prediction comes from ``run_block``, which
                slices both queries and image tokens out of ``block.norm1(xq)``
                — normalised. The final prediction comes from ``forward`` after
                ``post_blocks``, and no wrapper normalises ``q`` there (they all
                do ``x = self.norm(x); return x, q``), so ``class_head`` and
                ``mask_head`` — *shared weights* — read a raw residual-stream
                query for the one prediction that is actually used at eval.

                ``EoMT_Original`` has no such split: it calls
                ``_predict(bb.norm(x))`` at every block and at the end, with the
                queries inside ``x``, so both streams are normalised together
                every time (``models/eomt_original.py:161,190``).

                Lives in ``LUMA.forward`` and uses ``encoder.norm``, which every
                wrapper exposes, so **all 20 backbones honour it**. Adds no
                parameters. Defaults to ``False`` so every checkpoint trained
                before 2026-08-13 reproduces bit-for-bit.
            predict_q_norm: Give the prediction heads their *own* LayerNorm over
                the queries, applied inside ``_predict`` — so every prediction,
                deep-supervision and final alike, reads a query normalised by the
                same module.

                This is the third answer to the same split ``final_query_norm``
                addresses, and the only one that fixes the cause rather than one
                side of it. The alternatives both have a defect:

                - ``final_query_norm`` reuses ``encoder.norm``, which is the
                  backbone's *output* norm and, under
                  ``cross_attn_encoder_norm=True``, is simultaneously being
                  trained mid-stack on the cross-attention input. One module,
                  two jobs, conflicting gradients.
                - Feeding the heads a raw query everywhere removes the mismatch
                  but reintroduces scale drift: residual-stream magnitude grows
                  with depth, so per-block logit scales diverge, the
                  deep-supervision terms get implicitly re-weighted by block
                  index, and ``build_attn_mask``'s threshold-at-zero shifts with
                  it.

                A dedicated LayerNorm has neither problem: it is owned by the
                heads, trained only by them, and pins the query scale by
                construction. It is also what Mask2Former does (``decoder_norm``
                before the head) and structurally what ``EoMT_Original`` gets for
                free from normalising queries and image tokens together at every
                prediction site.

                Adds one LayerNorm (``2 * embed_dim`` params) and starts as the
                standard identity-affine init. Mutually exclusive with
                ``final_query_norm`` (that would normalise the final query
                twice). Honoured by the ViT backbone only — the other 19 wrappers
                hand ``_predict`` a ``block.norm1``-normalised query, so the new
                norm would stack on top of an existing one, which is the exact
                double-normalisation this flag exists to remove; the constructor
                raises rather than doing that silently. Defaults to ``False``.
        """
        super().__init__()
        self.encoder = encoder
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled
        self.use_layer_scale = use_layer_scale
        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))
        self.query_only_residual = query_only_residual
        if masked_cross_attn and self_a:
            raise ValueError("masked_cross_attn requires self_a=False; Attention takes no mask")
        self.masked_cross_attn = masked_cross_attn
        if attend_queries and self_a:
            raise ValueError("attend_queries requires self_a=False; Attention already attends over both streams")
        self.attend_queries = attend_queries
        self.layer_scale_init = layer_scale_init
        self.attn = self.get_attention_list(self_a)
        self.q = nn.Parameter(torch.randn(num_q, self.encoder.embed_dim))
        self.ls_list = self.get_layer_scale_list()
        self.dp = DropPath(0.0)

        self.query_self_attn = query_self_attn
        self.q_self_attn = self.get_query_self_attn_list() if query_self_attn else None

        self.cross_attn_encoder_norm = cross_attn_encoder_norm
        self.query_pos_embed = query_pos_embed
        self.thread_queries = thread_queries
        self.final_query_norm = final_query_norm

        if predict_q_norm and final_query_norm:
            raise ValueError(
                "predict_q_norm and final_query_norm both normalise the final query; "
                "pick one (predict_q_norm also covers the deep-supervision predictions)"
            )
        # `query_self_attn`/`thread_queries` were ported to every wrapper on
        # 2026-08-15 and `masked_cross_attn` on 2026-08-17; these three are still
        # read only in `models/vit.py:run_block`. Setting one on another encoder
        # used to be silently ignored — a benchmark config asking for it would
        # train a different model and nothing would say so. Raise instead.
        # (All of them default to the no-op value, so no existing config trips this.)
        vit_only = {
            "query_pos_embed": query_pos_embed,
            "cross_attn_encoder_norm": not cross_attn_encoder_norm,
            "predict_q_norm": predict_q_norm,
        }
        if any(vit_only.values()):
            # Local import: models.vit does not import models.luma, but keeping
            # this out of module scope avoids adding an import cycle if it ever does.
            from models.vit import ViT

            if not isinstance(encoder, ViT):
                requested = sorted(k for k, v in vit_only.items() if v)
                raise ValueError(
                    f"{requested} are honoured by the ViT wrapper only, but the encoder is "
                    f"{type(encoder).__name__}; run_block would ignore them and train a "
                    f"different model than the config asks for. Unset them, or implement "
                    f"them in that wrapper's run_block."
                )
        self.pred_q_norm = (
            nn.LayerNorm(self.encoder.embed_dim) if predict_q_norm else None
        )
        if num_blocks > 1 and not thread_queries:
            # Not fatal — the pre-2026-08-15 benchmark configs run this way and
            # must keep reproducing — but every block except the last throws its
            # query update away, so the run is an expensive num_blocks=1 plus a
            # dead attn module. Every wrapper can thread now; new configs should.
            warnings.warn(
                f"LUMA(num_blocks={num_blocks}, thread_queries=False): blocks run in parallel off the "
                f"raw self.q, so attn[:-1] gets exactly-zero gradients and only the last block's query "
                f"update survives. Set thread_queries=True or num_blocks=1.",
                stacklevel=2,
            )
        if num_blocks > 1 and not thread_queries and (query_self_attn or query_pos_embed):
            # Both only touch the query rows, which every block but the last
            # discards without threading — q_self_attn[:-1] then has no gradient
            # path and DDP aborts. There is no useful reading of this combination.
            raise ValueError(
                "query_self_attn / query_pos_embed with num_blocks > 1 require thread_queries=True; "
                "otherwise every block but the last discards its query update"
            )
        # Zero-init: identity at step 0, same rationale as q_self_attn's proj.
        self.q_pos_embed = (
            nn.Parameter(torch.zeros(num_blocks, num_q, self.encoder.embed_dim))
            if query_pos_embed
            else None
        )

        if query_mlp not in ("shared", "separate", "none"):
            raise ValueError(f"query_mlp must be one of 'shared', 'separate', 'none'; got {query_mlp!r}")
        self.query_mlp = query_mlp
        self.query_mlp_ratio = query_mlp_ratio
        if query_mlp == "separate":
            hidden_dim = int(self.encoder.embed_dim * query_mlp_ratio)
            # One LN+MLP per EoMT block (matches the per-block reuse pattern of
            # the shared variant). The LN is bundled into the Sequential so the
            # call site stays a single `q_mlps[idx](q)` invocation.
            self.q_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(self.encoder.embed_dim),
                        nn.Linear(self.encoder.embed_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, self.encoder.embed_dim),
                    )
                    for _ in range(num_blocks)
                ]
            )

        self.class_head = nn.Linear(self.encoder.embed_dim, num_classes + 1)

        self.mask_head = nn.Sequential(
            nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim),
        )

        patch_size = encoder.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.encoder.embed_dim) for _ in range(num_upscale)],
        )

    def get_layer_scale_list(self):
        ls_list = nn.ModuleList()
        embed_dim = self.encoder.embed_dim
        len_blocks = sum(self.encoder.depths)
        i = 0
        for d in self.encoder.depths:
            for _ in range(d):
                if i >= len_blocks - self.num_blocks:
                    ls_list.append(
                        LayerScale(embed_dim, init_values=self.layer_scale_init)
                        if self.use_layer_scale
                        else nn.Identity()
                    )
                i += 1
            embed_dim = embed_dim * self.encoder.attn_multiplier
        return ls_list

    def block_dims(self):
        """``(embed_dim, num_heads)`` for every block, in forward order.

        Wrappers report ``num_heads`` in one of two shapes: per *stage*, one
        entry per ``depths`` entry (18 of the 20), or per *block*, one entry per
        block (``efficient_net`` and ``next_vit``, whose head count varies
        inside a stage). Normalise here so both are correct — a plain
        ``zip(depths, num_heads)`` silently truncated the per-block form to its
        first ``len(depths)`` entries and built the cross-attention with the
        wrong head counts, and ``strict=True`` later turned that into a crash at
        construction for those two backbones.
        """
        depths, heads = list(self.encoder.depths), list(self.encoder.num_heads)
        dims, embed_dim = [], self.encoder.embed_dim
        if len(heads) == sum(depths):
            per_block = heads
        elif len(heads) == len(depths):
            per_block = [h for d, h in zip(depths, heads, strict=True) for _ in range(d)]
        else:
            raise ValueError(
                f"{type(self.encoder).__name__}.num_heads has {len(heads)} entries; expected "
                f"{len(depths)} (per stage) or {sum(depths)} (per block)"
            )
        i = 0
        for d in depths:
            for _ in range(d):
                dims.append((embed_dim, per_block[i]))
                i += 1
            embed_dim = embed_dim * self.encoder.attn_multiplier
        return dims

    def eomt_block_dims(self):
        """``(embed_dim, num_heads)`` for each of the last ``num_blocks`` blocks."""
        return self.block_dims()[-self.num_blocks :]

    def apply_query_self_attn(self, q: torch.Tensor, block_idx: int) -> torch.Tensor:
        """Let the queries attend to each other once, for one EoMT block.

        Call it from ``run_block`` in Mask2Former order — after the
        cross-attention residual, before the query MLP. A no-op unless
        ``query_self_attn=True``, so every wrapper can call it unconditionally.

        ``block_idx`` may be the 0-based EoMT block index or the negative index
        into the last ``num_blocks`` entries (``i - len(blocks)``); wrappers use
        both conventions and either addresses ``q_self_attn`` correctly.
        """
        if self.q_self_attn is None:
            return q
        return q + self.dp(self.q_self_attn[block_idx](q))

    def get_query_self_attn_list(self):
        """LayerNorm + self-attention over the query tokens, one per EoMT block.

        ``proj`` is zero-initialised so each block starts as an exact identity:
        the graft cannot perturb the pretrained stream at step 0 and has to earn
        its contribution.
        """
        sa_list = nn.ModuleList()
        for dim, heads in self.eomt_block_dims():
            attn = Attention(dim=dim, num_heads=heads)
            nn.init.zeros_(attn.proj.weight)
            nn.init.zeros_(attn.proj.bias)
            sa_list.append(nn.Sequential(nn.LayerNorm(dim), attn))
        return sa_list

    def get_attention_list(self, self_a):
        attn_list = nn.ModuleList()
        for embed_dim, h in self.eomt_block_dims():
            if self_a:
                attn_list.append(Attention(dim=embed_dim, num_heads=h))
            else:
                attn_list.append(
                    CrossAttentionSingleInput(
                        dim=embed_dim,
                        num_heads=h,
                        query_only_residual=self.query_only_residual,
                        attend_queries=self.attend_queries,
                    )
                )
        return attn_list

    def mask_is_active(self, block_idx: int) -> bool:
        """Whether block ``block_idx`` still constrains attention at all.

        ``attn_mask_probs`` anneals to 0, at which point every row is unmasked
        and the whole mechanism is a no-op — so this also gates the extra
        ``_predict`` call, keeping annealed-out inference free.
        """
        return self.masked_cross_attn and bool(self.attn_mask_probs[block_idx] > 0)

    def build_attn_mask(self, mask_logits: torch.Tensor, block_idx: int, grid_size, num_prefix: int):
        """Boolean (B, num_q, num_prefix + H*W) mask over query->context attention.

        Mirrors ``EoMT_Original.forward``: threshold the query's own mask
        prediction at 0 over the patch columns, leave the prefix columns open,
        then randomly release whole rows according to ``attn_mask_probs``
        (``EoMT_Original._disable_attn_mask``).
        """
        interpolated = F.interpolate(mask_logits, grid_size, mode="bilinear")
        mask = interpolated.flatten(2) > 0  # (B, Q, H*W)

        # Prefix (cls/register) columns stay open. This is also what keeps a
        # query whose prediction is empty everywhere from producing a fully
        # masked row, i.e. softmax(-inf, ...) = NaN.
        if num_prefix:
            mask = torch.cat([mask.new_ones(*mask.shape[:2], num_prefix), mask], dim=2)

        prob = self.attn_mask_probs[block_idx]
        if prob < 1:
            released = torch.rand(*mask.shape[:2], device=mask.device) > prob
            mask[released] = True

        return mask

    def predict_and_mask(self, x: torch.Tensor, q, block_idx: int):
        """Deep-supervision predict + the cross-attention mask for one EoMT block.

        Every wrapper's ``run_block`` needs the same four steps in the same order
        — predict, stash the logits while training, build the mask, feed it to
        ``attn`` — so they live here rather than being copy-pasted 21 times.
        Returns the boolean mask, or ``None`` when ``masked_cross_attn`` is off
        or ``attn_mask_probs`` has annealed to zero.

        ``grid_size``/``num_prefix_tokens`` come off the encoder, which is where
        ``_predict`` already reads them: the mask is interpolated onto exactly the
        grid the mask logits were reshaped to, so it cannot disagree with them.

        ``block_idx`` accepts both index conventions the wrappers use — 0-based
        within the EoMT window, or the negative ``i - len(blocks)`` — since both
        select the same entry of a length-``num_blocks`` buffer.
        """
        mask_active = self.mask_is_active(block_idx)
        if not (self.training or mask_active):
            return None

        mask_logits, class_logits = self._predict(x, q)
        if self.training:
            self.mask_logits_per_layer.append(mask_logits)
            self.class_logits_per_layer.append(class_logits)

        if not mask_active:
            return None
        return self.build_attn_mask(
            mask_logits, block_idx, self.encoder.grid_size, self.encoder.num_prefix_tokens
        )

    def _predict(self, x: torch.Tensor, q):
        # The heads' own query norm, so every prediction — deep-supervision and
        # final — reads the same distribution. Callers must pass a *raw* q when
        # this is active (models/vit.py:run_block does); see `predict_q_norm`.
        if self.pred_q_norm is not None:
            q = self.pred_q_norm(q)

        class_logits = self.class_head(q)

        if len(x.shape) < 4:
            x = x[:, self.encoder.num_prefix_tokens :, :]
            x = x.transpose(1, 2).reshape(x.shape[0], -1, *self.encoder.grid_size)
        mask_features = self.mask_head(q)  # (B, Q, C)
        upscaled = self.upscale(x)  # (B, C, H, W)

        B, Q, _ = mask_features.shape
        mask_logits = torch.matmul(mask_features, upscaled.flatten(2)).view(  # (B, C, H*W)
            B, Q, *upscaled.shape[-2:]
        )  # (B, Q, H, W)

        return mask_logits, class_logits

    def forward(self, x: torch.Tensor):
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        x = self.encoder.pre_block(x)
        self.mask_logits_per_layer, self.class_logits_per_layer = [], []
        q = None
        for i, block in enumerate(self.encoder.blocks):
            # Without thread_queries every block re-reads the raw `self.q`
            # parameter and the previous block's query update is discarded.
            q_in = self.q if q is None or not self.thread_queries else q
            x, q = self.encoder.run_block(block, self, x, q_in, i)

        x, q = self.encoder.post_blocks(x, q, self)
        if q is None or (isinstance(q, torch.Tensor) and q.dim() == 2):
            B = x.shape[0]
            q_src = self.q if q is None else q
            q = q_src[None].expand(B, -1, -1)
        # post_blocks normalises x but never q, so without this the shared
        # class/mask heads get a normalised query at every deep-supervision
        # block and a raw one for the prediction that is actually used.
        if self.final_query_norm:
            q = self.encoder.norm(q)
        # Collect Switch load-balancing loss if the backbone supports it
        if hasattr(self.encoder.backbone, "get_internal_loss"):
            self.switch_aux_loss = self.encoder.backbone.get_internal_loss()
        else:
            self.switch_aux_loss = None
        mask_logits, class_logits = self._predict(x, q=q)
        self.mask_logits_per_layer.append(mask_logits)
        self.class_logits_per_layer.append(class_logits)

        return (
            self.mask_logits_per_layer,
            self.class_logits_per_layer,
        )
