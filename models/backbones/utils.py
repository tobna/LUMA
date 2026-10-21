vit_sizes = {
    "na": dict(embed_dim=96, depth=2, num_heads=2),
    "mu": dict(embed_dim=144, depth=6, num_heads=3),
    "Ti": dict(embed_dim=192, depth=12, num_heads=3),
    "S": dict(embed_dim=384, depth=12, num_heads=6),
    "B": dict(embed_dim=768, depth=12, num_heads=12),
    "L": dict(embed_dim=1024, depth=24, num_heads=16),
    "H": dict(embed_dim=1280, depth=32, num_heads=16),
    "H+": dict(embed_dim=1280, depth=32, num_heads=20),
    "G": dict(embed_dim=1536, depth=40, num_heads=24),
    # AIMv2 uses a non-standard width/depth/head pattern (shallow + wide).
    "AIMv2-L": dict(embed_dim=1024, depth=24, num_heads=8),
    "AIMv2-H": dict(embed_dim=1536, depth=24, num_heads=12),
}
