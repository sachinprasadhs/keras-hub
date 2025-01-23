import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.feature_pyramid_backbone import FeaturePyramidBackbone
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.CSPDarkNetBackbone")
class CSPDarkNetBackbone(FeaturePyramidBackbone):
    """This class represents Keras Backbone of CSPNet model.

    This class implements a CSPNet backbone as described in
    [CSPNet: A New Backbone that can Enhance Learning Capability of CNN](
        https://arxiv.org/abs/1911.11929).

    Args:
        stackwise_num_filters:  A list of ints, filter size for each dark
            level in the model.
        stackwise_depth: A list of ints, the depth for each dark level in the
            model.
        block_type: str. One of `"basic_block"` or `"depthwise_block"`.
            Use `"depthwise_block"` for depthwise conv block
            `"basic_block"` for basic conv block.
            Defaults to "basic_block".
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        image_shape: tuple. The input shape without the batch size.
            Defaults to `(None, None, 3)`.

    Examples:
    ```python
    input_data = np.ones(shape=(8, 224, 224, 3))

    # Pretrained backbone
    model = keras_hub.models.CSPDarkNetBackbone.from_preset(
        "csp_darknet_tiny_imagenet"
    )
    model(input_data)

    # Randomly initialized backbone with a custom config
    model = keras_hub.models.CSPNetBackbone(
        stackwise_num_filters=[128, 256, 512, 1024],
        stackwise_depth=[3, 9, 9, 3],
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        stem_filter,
        stem_kernel_size,
        stem_stride,
        stackwise_depth,
        stackwise_num_filters,
        stride,
        bottle_ratio,
        block_ratio,
        stage_type,
        block_type,
        stem_activattion="relu",
        stem_pooling=None,
        avg_down=False,
        image_shape=(None, None, 3),
        data_format=None,
        **kwargs,
    ):
        # === Functional Model ===
        data_format = standardize_data_format(data_format)
        channel_axis = -1 if data_format == "channels_last" else 1

        image_input = layers.Input(shape=image_shape)
        x = image_input  # Intermediate result.
        stem, stem_feat_info = create_csp_stem(
            data_format=data_format,
            channel_axis=channel_axis,
            filters=stem_filter,
            kernel_size=stem_kernel_size,
            stride=stem_stride,
            pooling=stem_pooling,
            padding=None,
            activation="relu",
            name=None,
        )(x)

        super().__init__(inputs=image_input, outputs=stem, **kwargs)

        # === Config ===
        self.stem_filter = (stem_filter,)
        self.stem_kernel_size = (stem_kernel_size,)
        self.stem_stride = (stem_stride,)
        self.stackwise_depth = (stackwise_depth,)
        self.stackwise_num_filters = (stackwise_num_filters,)
        self.image_shape = image_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_num_filters": self.stackwise_num_filters,
                "stackwise_depth": self.stackwise_depth,
                "image_shape": self.image_shape,
                "stem_stride": self.stem_stride,
                "stem_kernel_size": self.stem_kernel_size,
                "stem_filter": self.stem_filter,
            }
        )
        return config


def bottleneck_block(
    filters,
    channel_axis,
    data_format,
    dilation=1,
    bottle_ratio=0.25,
    groups=1,
    activation="relu",
    name=None,
):
    """
    Spatial pyramid pooling layer used in YOLOv3-SPP

    Args:
        filters: Integer, the dimensionality of the output spaces (i.e. the
            number of output filters in used the blocks).
        hidden_filters: Integer, the dimensionality of the intermediate
            bottleneck space (i.e. the number of output filters in the
            bottleneck convolution). If None, it will be equal to filters.
            Defaults to None.
        kernel_sizes: A list or tuple representing all the pool sizes used for
            the pooling layers, defaults to (5, 9, 13).
        activation: Activation for the conv layers, defaults to "silu".
        name: the prefix for the layer names used in the block.

    Returns:
        a function that takes an input Tensor representing an
        SpatialPyramidPoolingBottleneck.
    """
    if name is None:
        name = f"bottleneck{keras.backend.get_uid('bottleneck')}"

    hidden_filters = int(round(filters * bottle_ratio))

    def apply(x):
        shortcut = x
        x = layers.Conv2D(
            hidden_filters,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            name=f"{name}_bottleneck_conv_1",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=f"{name}_bottleneck_bn_1"
        )(x)
        x = layers.Activation(activation)(x)

        x = layers.Conv2D(
            hidden_filters,
            kernel_size=3,
            dilation=dilation,
            groups=groups,
            padding="same",
            use_bias=False,
            data_format=data_format,
            name=f"{name}_bottleneck_conv_2",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=f"{name}_bottleneck_bn_2"
        )(x)
        x = layers.Activation(activation)(x)

        x = layers.Conv2D(
            filters,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            name=f"{name}_bottleneck_conv_3",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=f"{name}_bottleneck_bn_3"
        )(x)
        x = layers.Activation(activation)(x)

        x = layers.add([x, shortcut])
        x = layers.Activation(activation)(x)
        return x

    return apply


def dark_block(
    filters,
    data_format,
    channel_axis,
    dilation=1,
    bottle_ratio=0.5,
    groups=1,
    activation="relu",
    name=None,
):
    if name is None:
        name = f"dark{keras.backend.get_uid('dark')}"

    hidden_filters = int(round(filters * bottle_ratio))

    def apply(x):
        shortcut = x
        x = layers.Conv2D(
            hidden_filters,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            name=f"{name}_dark_conv_1",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=f"{name}_dark_bn_1"
        )(x)
        x = layers.Activation(activation)(x)

        x = layers.Conv2D(
            filters,
            kernel_size=3,
            dilation=dilation,
            groups=groups,
            padding="same",
            use_bias=False,
            data_format=data_format,
            name=f"{name}_dark_conv_2",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=f"{name}_dark_bn_2"
        )(x)
        x = layers.Activation(activation)(x)

        x = layers.add([x, shortcut])
        return x

    return apply


def edge_block(
    filters,
    data_format,
    channel_axis,
    dilation=1,
    bottle_ratio=0.5,
    groups=1,
    activation="relu",
    name=None,
):
    if name is None:
        name = f"edge{keras.backend.get_uid('edge')}"

    hidden_filters = int(round(filters * bottle_ratio))

    def apply(x):
        shortcut = x
        x = layers.Conv2D(
            hidden_filters,
            kernel_size=3,
            use_bias=False,
            dilation=dilation,
            groups=groups,
            padding="same",
            data_format=data_format,
            name=f"{name}_edge_conv_1",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=f"{name}_edge_bn_1"
        )(x)
        x = layers.Activation(activation)(x)

        x = layers.Conv2D(
            filters,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            name=f"{name}_edge_conv_2",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name=f"{name}_edge_bn_2"
        )(x)
        x = layers.Activation(activation)(x)

        x = layers.add([x, shortcut])
        return x

    return apply


def cross_stage(
    filters,
    stride,
    # prev_channels,
    dilation,
    depth,
    data_format,
    channel_axis,
    block_ratio=1.0,
    bottle_ratio=1.0,
    expand_ratio=1.0,
    groups=1,
    first_dilation=None,
    avg_down=False,
    activation="silu",
    down_growth=False,
    cross_linear=False,
    block_dpr=None,
    block_fn=bottleneck_block,
    name=None,
):
    if name is None:
        name = f"cross_stage_{keras.backend.get_uid('cross_stage')}"

    first_dilation = first_dilation or dilation

    def apply(x):
        down_chs = filters if down_growth else prev_channels
        expand_chs = int(round(filters * expand_ratio))
        block_channels = int(round(filters * block_ratio))

        if stride != 1 or first_dilation != dilation:
            if avg_down:
                if stride == 2:
                    x = layers.AveragePooling2D(2, name=f"{name}_avg_pool")(x)
                x = layers.Conv2D(
                    filters,
                    kernel_size=1,
                    strides=1,
                    use_bias=False,
                    groups=groups,
                    data_format=data_format,
                    name=f"{name}_conv_down",
                )(x)
                x = layers.BatchNormalization(
                    axis=channel_axis, name=f"{name}_bn"
                )(x)
                x = layers.Activation(activation)(x)
            else:
                x = layers.Conv2D(
                    down_chs,
                    kernel_size=3,
                    strides=stride,
                    dilation=first_dilation,
                    use_bias=False,
                    groups=groups,
                    data_format=data_format,
                    name=f"{name}_conv_down",
                )(x)
                x = layers.BatchNormalization(
                    axis=channel_axis, name=f"{name}_bn"
                )(x)
                x = layers.Activation(activation)(x)

        x = layers.Conv2D(
            expand_chs,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            name=f"{name}_conv_exp",
        )(x)
        x = layers.BatchNormalization(axis=channel_axis, name=f"{name}_bn")(x)
        if not cross_linear:
            x = layers.Activation(activation)(x)

        xs, xb = ops.split(
            x, indices_or_sections=expand_chs // 2, axis=channel_axis
        )

        for i in range(depth):
            xb = block_fn(
                filters=block_channels,
                dilation=dilation,
                bottle_ratio=bottle_ratio,
                groups=groups,
                drop_path=block_dpr[i] if block_dpr is not None else 0.0,
                activation="relu",
                data_format=data_format,
                channel_axis=channel_axis,
            )(xb)

        xb = layers.Conv2D(
            expand_chs // 2,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            name=f"{name}_conv_transition_b",
        )(xb)
        xb = layers.BatchNormalization(
            axis=channel_axis, name=f"{name}_transition_b_bn"
        )(xb)
        xb = layers.Activation(activation)(xb)

        out = layers.Concatenate(
            axis=channel_axis, name=f"{name}_conv_transition"
        )([xs, xb])
        out = layers.Conv2D(
            filters,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            name=f"{name}_conv_transition",
        )(out)
        out = layers.BatchNormalization(
            axis=channel_axis, name=f"{name}_transition_bn"
        )(out)
        out = layers.Activation(activation)(out)
        return out

    return apply


def cross_stage3(
    filters,
    stride,
    prev_channels,
    dilation,
    depth,
    data_format,
    channel_axis,
    block_ratio=1.0,
    bottle_ratio=1.0,
    expand_ratio=1.0,
    groups=1,
    first_dilation=None,
    avg_down=False,
    activation="silu",
    down_growth=False,
    cross_linear=False,
    block_dpr=None,
    block_fn=bottleneck_block,
    name=None,
):
    if name is None:
        name = f"cross_stage3_{keras.backend.get_uid('cross_stage3')}"

    first_dilation = first_dilation or dilation

    def apply(x):
        down_chs = filters if down_growth else prev_channels
        expand_chs = int(round(filters * expand_ratio))
        block_channels = int(round(filters * block_ratio))

        if stride != 1 or first_dilation != dilation:
            if avg_down:
                if stride == 2:
                    x = layers.AveragePooling2D(2, name=f"{name}_avg_pool")(x)
                x = layers.Conv2D(
                    filters,
                    kernel_size=1,
                    strides=1,
                    use_bias=False,
                    groups=groups,
                    data_format=data_format,
                    name=f"{name}_conv_down",
                )(x)
                x = layers.BatchNormalization(
                    axis=channel_axis, name=f"{name}_bn"
                )(x)
                x = layers.Activation(activation)(x)
            else:
                x = layers.Conv2D(
                    down_chs,
                    kernel_size=3,
                    strides=stride,
                    dilation=first_dilation,
                    use_bias=False,
                    groups=groups,
                    data_format=data_format,
                    name=f"{name}_conv_down",
                )(x)
                x = layers.BatchNormalization(
                    axis=channel_axis, name=f"{name}_bn"
                )(x)
                x = layers.Activation(activation)(x)

        x = layers.Conv2D(
            expand_chs,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            name=f"{name}_conv_exp",
        )(x)
        x = layers.BatchNormalization(axis=channel_axis, name=f"{name}_bn")(x)
        if not cross_linear:
            x = layers.Activation(activation)(x)

        x1, x2 = ops.split(
            x, indices_or_sections=expand_chs // 2, axis=channel_axis
        )

        for i in range(depth):
            x1 = block_fn(
                filters=block_channels,
                dilation=dilation,
                bottle_ratio=bottle_ratio,
                groups=groups,
                drop_path=block_dpr[i] if block_dpr is not None else 0.0,
                activation="relu",
                data_format=data_format,
                channel_axis=channel_axis,
            )(x1)

        out = layers.Concatenate(
            axis=channel_axis, name=f"{name}_conv_transition"
        )([x1, x2])
        out = layers.Conv2D(
            expand_chs // 2,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            name=f"{name}_conv_transition",
        )(out)
        out = layers.BatchNormalization(
            axis=channel_axis, name=f"{name}_transition_bn"
        )(out)
        out = layers.Activation(activation)(out)
        return out

    return apply


def dark_stage():
    pass


def create_csp_stem(
    data_format,
    channel_axis,
    filters=32,
    kernel_size=3,
    stride=2,
    pooling=None,
    padding="valid",
    activation="relu",
    name=None,
):
    if not isinstance(filters, (tuple, list)):
        filters = [filters]
    stem_depth = len(filters)
    assert stem_depth
    assert stride in (1, 2, 4)
    last_idx = stem_depth - 1

    def apply(x):
        stem_stride = 1
        for i, chs in enumerate(filters):
            conv_stride = (
                2
                if (i == 0 and stride > 1)
                or (i == last_idx and stride > 2 and not pooling)
                else 1
            )
            x = layers.Conv2D(
                chs,
                kernel_size=kernel_size,
                strides=conv_stride,
                # padding=padding if i == 0 else "",
                use_bias=False,
                data_format=data_format,
                name=f"csp_stem_conv_{i}",
            )(x)
            x = layers.BatchNormalization(axis=channel_axis, name=f"csp_stem_bn_{i}")(
                x
            )
            x = layers.Activation(activation)(x)
            stem_stride *= conv_stride

        if pooling == "max":
            assert stride > 2
            x = layers.MaxPooling2D(
                pool_size=3,
                strides=2,
                padding="valid",
                data_format=None,
                name="pool",
            )(x)
            stem_stride *= 2
        return x, stem_stride

    return apply


def create_csp_stages(
    inputs,
    filters,
    data_format,
    channel_axis,
    depth,
    reduction,
    drop_path_rate,
    block_dpr,
    stride,
    output_stride,
    stage_type=None,
    block_type=None,
    name=None,
):
    num_stages = len(depth)
    block_dpr = (
        [None] * num_stages
        if not drop_path_rate
        else [
            x.tolist()
            for x in ops.linspace(0, drop_path_rate, sum(depth)).split(depth)
        ]
    )
    dilation = 1
    net_stride = reduction
    if stage_type == "dark":
        stage_fn = dark_stage
    elif stage_type == "csp":
        stage_fn = cross_stage
    else:
        stage_fn = cross_stage3

    if block_type == "dark":
        block_fn = dark_block
    elif block_type == "edge":
        block_fn = edge_block
    else:
        block_fn = bottleneck_block

    stages = inputs
    for block_idx in enumerate(depth):
        if net_stride >= output_stride and stride > 1:
            dilation *= stride
            stride = 1
        net_stride *= stride
        first_dilation = 1 if dilation in (1, 2) else 2
        stages = stage_fn(
            filters,
            depth,
            data_format,
            channel_axis,
            stride=stride,
            dilation=dilation,
            block_ratio=1.0,
            bottle_ratio=1.0,
            expand_ratio=1.0,
            groups=1,
            first_dilation=first_dilation,
            avg_down=False,
            activation="silu",
            down_growth=False,
            cross_linear=False,
            block_dpr=None,
            block_fn=block_fn,
            name=None,
        )(stages)
    return stages
