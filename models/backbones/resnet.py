from timm.models import register_model
from timm.models.resnet import Bottleneck
from timm.models.resnet import ResNet as ResNetTimm
from torch import nn


class ResNet(ResNetTimm):
    """The popular ResNet model with a ResizingInterface."""

    def __init__(self, *args, global_pool="avg", **kwargs):
        """Create ResNet model with resizing capabilities.

        Args:
            *args: Arguments for the ResNet model.
            global_pool (str, optional): _description_. Defaults to "avg".
            **kwargs: Keyword arguments for the ResNet model (from Timm).

        Keyword Args:
            block, layers, num_classes, in_chans, output_stride, cardinality, base_width, stem_width, stem_type, replace_stem_pool, block_reduce_first, down_kernel_size, avg_down, act_layer, norm_layer, aa_layer, drop_rate, drop_path_rate, drop_block_rate, zero_init_last, block_args

        """
        admissible_kwargs = [
            "block",
            "layers",
            "num_classes",
            "in_chans",
            "output_stride",
            "cardinality",
            "base_width",
            "stem_width",
            "stem_type",
            "replace_stem_pool",
            "block_reduce_first",
            "down_kernel_size",
            "avg_down",
            "act_layer",
            "norm_layer",
            "aa_layer",
            "drop_rate",
            "drop_path_rate",
            "drop_block_rate",
            "zero_init_last",
            "block_args",
        ]
        for key in list(kwargs.keys()):
            if key not in admissible_kwargs:
                kwargs.pop(key)
        super().__init__(*args, global_pool=global_pool, **kwargs)
        self.global_pool_str = global_pool

    def set_image_res(self, res):
        # resizing not needed for CNNs with pooling
        return

    def set_num_classes(self, n_classes):
        if self.num_classes == n_classes:
            return
        if n_classes > 0:
            self.reset_classifier(num_classes=n_classes, global_pool=self.global_pool_str)
        else:
            self.fc = nn.Identity()


@register_model
def resnet50(pretrained=False, **kwargs):
    """Construct a ResNet-50 model."""
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3])
    return ResNet(**model_args, **kwargs)
