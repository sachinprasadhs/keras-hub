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
        stackwise_depth: A list of ints, the depth for each block in the
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
        stackwise_stride,
        stackwise_num_filters,
        block_type,
        groups=1,
        stage_type=None,
        activation="leaky_relu",
        output_stride=32,
        first_dilation=None,
        bottle_ratio=(1.0),
        block_ratio=(1.0),
        expand_ratio=(0.5),
        stem_padding=None,
        stem_pooling=None,
        avg_down=False,
        down_growth=False,
        cross_linear=False,
        image_shape=(None, None, 3),
        data_format=None,
        dtype=None,
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
            padding=stem_padding,
            activation=activation,
            dtype=dtype,
        )(x)

        stages, pyramid_outputs = create_csp_stages(
            inputs=stem,
            filters=stackwise_num_filters,
            data_format=data_format,
            channel_axis=channel_axis,
            stackwise_depth=stackwise_depth,
            reduction=stem_feat_info,
            groups=groups,
            block_ratio=block_ratio,
            bottle_ratio=bottle_ratio,
            expand_ratio=expand_ratio,
            stride=stackwise_stride,
            avg_down=avg_down,
            first_dilation=first_dilation,
            down_growth=down_growth,
            cross_linear=cross_linear,
            activation=activation,
            output_stride=output_stride,
            stage_type=stage_type,
            block_type=block_type,
            dtype=dtype,
            name="csp_stage",
        )

        super().__init__(inputs=image_input, outputs=stages, dtype=dtype, **kwargs)

        # === Config ===
        self.stem_filter = (stem_filter,)
        self.stem_kernel_size = (stem_filter,)
        self.stem_stride = (stem_stride,)
        self.stackwise_depth = (stackwise_depth,)
        self.stackwise_stride = (stackwise_stride,)
        self.stackwise_num_filters = (stackwise_num_filters,)
        self.stage_type = (stage_type,)
        self.block_type = (block_type,)
        self.output_stride = (output_stride,)
        self.groups = (groups,)
        self.activation = (activation,)
        self.first_dilation = (first_dilation,)
        self.bottle_ratio = (bottle_ratio,)
        self.block_ratio = (block_ratio,)
        self.expand_ratio = (expand_ratio,)
        self.stem_padding = (stem_padding,)
        self.stem_pooling = (stem_pooling,)
        self.avg_down = (avg_down,)
        self.down_growth = (down_growth,)
        self.cross_linear = (cross_linear,)
        self.image_shape = (image_shape,)
        self.data_format = (data_format,)
        self.image_shape = (image_shape,)
        self.pyramid_outputs = pyramid_outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stem_filter": self.stem_filter,
                "stem_kernel_size": self.stem_kernel_size,
                "stem_stride": self.stem_stride,
                "stackwise_depth": self.stackwise_depth,
                "stackwise_stride": self.stackwise_stride,
                "stackwise_num_filters": self.stackwise_num_filters,
                "stage_type": self.stage_type,
                "block_type": self.block_type,
                "output_stride": self.output_stride,
                "groups": self.groups,
                "activation": self.activation,
                "first_dilation": self.first_dilation,
                "bottle_ratio": self.bottle_ratio,
                "block_ratio": self.block_ratio,
                "expand_ratio": self.expand_ratio,
                "stem_padding": self.stem_padding,
                "stem_pooling": self.stem_pooling,
                "avg_down": self.avg_down,
                "down_growth": self.down_growth,
                "cross_linear": self.cross_linear,
                "image_shape": self.image_shape,
                "data_format": self.data_format,
                "pyramid_outputs": self.pyramid_outputs,
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
    dtype=None,
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
        activation: Activation for the conv layers, defaults to "relu".
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
            filters=hidden_filters,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_bottleneck_conv_1",
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_bottleneck_bn_1",
        )(x)
        if activation == "leaky_relu":
            x = layers.LeakyReLU(
                negative_slope=0.1,
                dtype=dtype,
                name=f"{name}_bottleneck_block_activation_1",
            )(x)
        else:
            x = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_bottleneck_block_activation_1",
            )(x)

        x = layers.Conv2D(
            filters=hidden_filters,
            kernel_size=3,
            dilation_rate=dilation,
            groups=groups,
            padding="same",
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_bottleneck_conv_2",
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_bottleneck_bn_2",
        )(x)
        if activation == "leaky_relu":
            x = layers.LeakyReLU(
                negative_slope=0.1,
                dtype=dtype,
                name=f"{name}_bottleneck_block_activation_2",
            )(x)
        else:
            x = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_bottleneck_block_activation_2",
            )(x)

        x = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_bottleneck_conv_3",
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_bottleneck_bn_3",
        )(x)
        if activation == "leaky_relu":
            x = layers.LeakyReLU(
                negative_slope=0.1,
                dtype=dtype,
                name=f"{name}_bottleneck_block_activation_3",
            )(x)
        else:
            x = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_bottleneck_block_activation_3",
            )(x)

        x = layers.add([x, shortcut])
        if activation == "leaky_relu":
            x = layers.LeakyReLU(
                negative_slope=0.1,
                dtype=dtype,
                name=f"{name}_bottleneck_block_activation_4",
            )(x)
        else:
            x = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_bottleneck_block_activation_4",
            )(x)
        return x

    return apply


def dark_block(
    filters,
    data_format,
    channel_axis,
    dilation,
    bottle_ratio,
    groups,
    activation,
    dtype=None,
    name=None,
):
    if name is None:
        name = f"dark{keras.backend.get_uid('dark')}"

    hidden_filters = int(round(filters * bottle_ratio))

    def apply(x):
        shortcut = x
        x = layers.Conv2D(
            filters=hidden_filters,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_dark_conv_1",
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_dark_bn_1",
        )(x)
        if activation == "leaky_relu":
            x = layers.LeakyReLU(
                negative_slope=0.1,
                dtype=dtype,
                name=f"{name}_dark_block_activation_1",
            )(x)
        else:
            x = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_dark_block_activation_1",
            )(x)

        x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            dilation_rate=dilation,
            groups=groups,
            padding="same",
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_dark_conv_2",
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_dark_bn_2",
        )(x)
        if activation == "leaky_relu":
            x = layers.LeakyReLU(
                negative_slope=0.1,
                dtype=dtype,
                name=f"{name}_dark_block_activation_2",
            )(x)
        else:
            x = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_dark_block_activation_2",
            )(x)

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
    dtype=None,
    name=None,
):
    if name is None:
        name = f"edge{keras.backend.get_uid('edge')}"

    hidden_filters = int(round(filters * bottle_ratio))

    def apply(x):
        shortcut = x
        x = layers.Conv2D(
            filters=hidden_filters,
            kernel_size=3,
            use_bias=False,
            dilation_rate=dilation,
            groups=groups,
            padding="same",
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_edge_conv_1",
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_edge_bn_1",
        )(x)
        if activation == "leaky_relu":
            x = layers.LeakyReLU(
                negative_slope=0.1,
                dtype=dtype,
                name=f"{name}_edge_block_activation_1",
            )(x)
        else:
            x = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_edge_block_activation_1",
            )(x)

        x = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_edge_conv_2",
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_edge_bn_2",
        )(x)
        if activation == "leaky_relu":
            x = layers.LeakyReLU(
                negative_slope=0.1,
                dtype=dtype,
                name=f"{name}_bottleneck_block_activation_2",
            )(x)
        else:
            x = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_bottleneck_block_activation_2",
            )(x)

        x = layers.add([x, shortcut])
        return x

    return apply


def cross_stage(
    filters,
    stride,
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
    activation="relu",
    down_growth=False,
    cross_linear=False,
    block_fn=bottleneck_block,
    dtype=None,
    name=None,
):
    if name is None:
        name = f"cross_stage_{keras.backend.get_uid('cross_stage')}"

    first_dilation = first_dilation or dilation

    def apply(x):
        prev_filters = keras.ops.shape(x)[channel_axis]
        down_chs = filters if down_growth else prev_filters
        expand_chs = int(round(filters * expand_ratio))
        block_channels = int(round(filters * block_ratio))

        if stride != 1 or first_dilation != dilation:
            if avg_down:
                if stride == 2:
                    x = layers.AveragePooling2D(
                        2, dtype=dtype, name=f"{name}_avg_pool"
                    )(x)
                x = layers.Conv2D(
                    filters=filters,
                    kernel_size=1,
                    strides=1,
                    use_bias=False,
                    groups=groups,
                    data_format=data_format,
                    dtype=dtype,
                    name=f"{name}_cross_stage_conv_down_1",
                )(x)
                x = layers.BatchNormalization(
                    epsilon=1e-05,
                    axis=channel_axis,
                    dtype=dtype,
                    name=f"{name}_bn_1",
                )(x)
                if activation == "leaky_relu":
                    x = layers.LeakyReLU(
                        negative_slope=0.1,
                        dtype=dtype,
                        name=f"{name}_cross_stage_activation_1",
                    )(x)
                else:
                    x = layers.Activation(
                        activation,
                        dtype=dtype,
                        name=f"{name}_cross_stage_activation_1",
                    )(x)
            else:
                x = layers.Conv2D(
                    filters=down_chs,
                    kernel_size=3,
                    strides=stride,
                    dilation_rate=first_dilation,
                    use_bias=False,
                    groups=groups,
                    data_format=data_format,
                    dtype=dtype,
                    name=f"{name}_cross_stage3_conv_down_1",
                )(x)
                x = layers.BatchNormalization(
                    epsilon=1e-05,
                    axis=channel_axis,
                    dtype=dtype,
                    name=f"{name}_bn_1",
                )(x)
                if activation == "leaky_relu":
                    x = layers.LeakyReLU(
                        negative_slope=0.1,
                        dtype=dtype,
                        name=f"{name}_cross_stage_activation_1",
                    )(x)
                else:
                    x = layers.Activation(
                        activation,
                        dtype=dtype,
                        name=f"{name}_cross_stage_activation_1",
                    )(x)

        x = layers.Conv2D(
            filters=expand_chs,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_conv_exp",
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-05, axis=channel_axis, dtype=dtype, name=f"{name}_bn_2"
        )(x)
        if not cross_linear:
            if activation == "leaky_relu":
                x = layers.LeakyReLU(
                    negative_slope=0.1,
                    dtype=dtype,
                    name=f"{name}_cross_stage_activation_2",
                )(x)
            else:
                x = layers.Activation(
                    activation,
                    dtype=dtype,
                    name=f"{name}_cross_stage_activation_2",
                )(x)
        prev_filters = keras.ops.shape(x)[channel_axis]
        xs, xb = ops.split(
            x, indices_or_sections=prev_filters//(expand_chs // 2), axis=channel_axis
        )

        for i in range(depth):
            xb = block_fn(
                filters=block_channels,
                dilation=dilation,
                bottle_ratio=bottle_ratio,
                groups=groups,
                activation="relu",
                data_format=data_format,
                channel_axis=channel_axis,
                dtype=dtype,
                name=f"block_{i}_{name}",
            )(xb)

        xb = layers.Conv2D(
            filters=expand_chs // 2,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_conv_transition_b",
        )(xb)
        xb = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_transition_b_bn",
        )(xb)
        if activation == "leaky_relu":
            xb = layers.LeakyReLU(
                negative_slope=0.1,
                dtype=dtype,
                name=f"{name}_cross_stage_activation_3",
            )(xb)
        else:
            xb = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_cross_stage_activation_3",
            )(xb)

        out = layers.Concatenate(
            axis=channel_axis, name=f"{name}_conv_transition_concat"
        )([xs, xb])
        out = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_conv_transition",
        )(out)
        out = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_transition_bn",
        )(out)
        if activation == "leaky_relu":
            out = layers.LeakyReLU(
                negative_slope=0.1,
                dtype=dtype,
                name=f"{name}_cross_stage_activation_4",
            )(out)
        else:
            out = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_cross_stage_activation_4",
            )(out)
        return out

    return apply


def cross_stage3(
    data_format,
    channel_axis,
    filters,
    stride,
    dilation,
    depth,
    block_ratio,
    bottle_ratio,
    expand_ratio,
    avg_down,
    activation,
    first_dilation,
    down_growth,
    cross_linear,
    block_fn,
    groups,
    name=None,
    dtype=None,
):
    if name is None:
        name = f"cross_stage3_{keras.backend.get_uid('cross_stage3')}"

    first_dilation = first_dilation or dilation

    def apply(x):
        prev_filters = keras.ops.shape(x)[channel_axis]
        down_chs = filters if down_growth else prev_filters
        expand_chs = int(round(filters * expand_ratio))
        block_filters = int(round(filters * block_ratio))

        if stride != 1 or first_dilation != dilation:
            if avg_down:
                if stride == 2:
                    x = layers.AveragePooling2D(
                        2, dtype=dtype, name=f"{name}_cross_stage3_avg_pool"
                    )(x)
                x = layers.Conv2D(
                    filters=filters,
                    kernel_size=1,
                    strides=1,
                    use_bias=False,
                    groups=groups,
                    data_format=data_format,
                    dtype=dtype,
                    name=f"{name}_cross_stage3_conv_down_1",
                )(x)
                x = layers.BatchNormalization(
                    epsilon=1e-05,
                    axis=channel_axis,
                    dtype=dtype,
                    name=f"{name}_bn_1",
                )(x)
                if activation == "leaky_relu":
                    x = layers.LeakyReLU(
                        negative_slope=0.1,
                        dtype=dtype,
                        name=f"{name}_cross_stage3_activation_1",
                    )(x)
                else:
                    x = layers.Activation(
                        activation,
                        dtype=dtype,
                        name=f"{name}_cross_stage3_activation_1",
                    )(x)
            else:
                x = layers.Conv2D(
                    filters=down_chs,
                    kernel_size=3,
                    strides=stride,
                    dilation_rate=first_dilation,
                    use_bias=False,
                    groups=groups,
                    data_format=data_format,
                    dtype=dtype,
                    name=f"{name}_cross_stage3_conv_down_2",
                )(x)
                x = layers.BatchNormalization(
                    epsilon=1e-05,
                    axis=channel_axis,
                    dtype=dtype,
                    name=f"{name}_bn_2",
                )(x)
                if activation == "leaky_relu":
                    x = layers.LeakyReLU(
                        negative_slope=0.1,
                        dtype=dtype,
                        name=f"{name}_cross_stage3_activation_1",
                    )(x)
                else:
                    x = layers.Activation(
                        activation,
                        dtype=dtype,
                        name=f"{name}_cross_stage3_activation_1",
                    )(x)

        x = layers.Conv2D(
            filters=expand_chs,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_conv_exp",
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-05, axis=channel_axis, dtype=dtype, name=f"{name}_bn"
        )(x)
        if not cross_linear:
            if activation == "leaky_relu":
                x = layers.LeakyReLU(
                    negative_slope=0.1,
                    dtype=dtype,
                    name=f"{name}_cross_stage3_activation_2",
                )(x)
            else:
                x = layers.Activation(
                    activation,
                    dtype=dtype,
                    name=f"{name}_cross_stage3_activation_2",
                )(x)

        prev_filters = keras.ops.shape(x)[channel_axis]
        x1, x2 = ops.split(
            x, indices_or_sections=prev_filters//(expand_chs // 2), axis=channel_axis
        )

        for i in range(depth):
            x1 = block_fn(
                filters=block_filters,
                dilation=dilation,
                bottle_ratio=bottle_ratio,
                groups=groups,
                activation=activation,
                data_format=data_format,
                channel_axis=channel_axis,
                dtype=dtype,
                name=f"block_{i}_{name}",
            )(x1)

        out = layers.Concatenate(
            axis=channel_axis, dtype=dtype, name=f"{name}_conv_transition"
        )([x1, x2])
        out = layers.Conv2D(
            filters=expand_chs // 2,
            kernel_size=1,
            use_bias=False,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_conv_transition",
        )(out)
        out = layers.BatchNormalization(
            epsilon=1e-05,
            axis=channel_axis,
            dtype=dtype,
            name=f"{name}_transition_bn",
        )(out)
        if activation == "leaky_relu":
            out = layers.LeakyReLU(
                negative_slope=0.1,
                dtype=dtype,
                name=f"{name}_cross_stage3_activation_3",
            )(out)
        else:
            out = layers.Activation(
                activation,
                dtype=dtype,
                name=f"{name}_cross_stage3_activation_3",
            )(out)
        return out

    return apply


def dark_stage():
    # TODO
    pass


def create_csp_stem(
    data_format,
    channel_axis,
    activation,
    padding,
    filters=32,
    kernel_size=3,
    stride=2,
    pooling=None,
    dtype=None,
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
                filters=chs,
                kernel_size=kernel_size,
                strides=conv_stride,
                padding=padding if i == 0 else "valid",
                use_bias=False,
                data_format=data_format,
                dtype=dtype,
                name=f"csp_stem_conv_{i}",
            )(x)
            x = layers.BatchNormalization(
                epsilon=1e-05,
                axis=channel_axis,
                dtype=dtype,
                name=f"csp_stem_bn_{i}",
            )(x)
            if activation == "leaky_relu":
                x = layers.LeakyReLU(
                    negative_slope=0.1,
                    dtype=dtype,
                    name=f"csp_stem_activation_{i}",
                )(x)
            else:
                x = layers.Activation(
                    activation,
                    dtype=dtype,
                    name=f"csp_stem_activation_{i}",
                )(x)
            stem_stride *= conv_stride

        if pooling == "max":
            assert stride > 2
            x = layers.MaxPooling2D(
                pool_size=3,
                strides=2,
                padding="same",
                data_format=data_format,
                dtype=dtype,
                name="csp_stem_pool",
            )(x)
            stem_stride *= 2
        return x, stem_stride

    return apply


def create_csp_stages(
    inputs,
    filters,
    data_format,
    channel_axis,
    stackwise_depth,
    reduction,
    first_dilation,
    block_ratio,
    bottle_ratio,
    expand_ratio,
    stride,
    groups,
    avg_down,
    down_growth,
    cross_linear,
    activation,
    output_stride,
    stage_type,
    block_type,
    dtype,
    name,
):
    if name is None:
        name = f"csp_stage_{keras.backend.get_uid('csp_stage')}"

    num_stages = len(stackwise_depth)
    dilation = 1
    net_stride = reduction
    stride = _pad_arg(stride, num_stages)
    expand_ratio = _pad_arg(expand_ratio, num_stages)
    bottle_ratio = _pad_arg(bottle_ratio, num_stages)
    block_ratio = _pad_arg(block_ratio, num_stages)

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
    pyramid_outputs = {}
    for stage_idx, _ in enumerate(stackwise_depth):
        if net_stride >= output_stride and stride[stage_idx] > 1:
            dilation *= stride[stage_idx]
            stride = 1
        net_stride *= stride[stage_idx]
        first_dilation = 1 if dilation in (1, 2) else 2
        stages = stage_fn(
            data_format,
            channel_axis,
            filters=filters[stage_idx],
            depth=stackwise_depth[stage_idx],
            stride=stride[stage_idx],
            dilation=dilation,
            block_ratio=block_ratio[stage_idx],
            bottle_ratio=bottle_ratio[stage_idx],
            expand_ratio=expand_ratio[stage_idx],
            groups=groups,
            first_dilation=first_dilation,
            avg_down=avg_down,
            activation=activation,
            down_growth=down_growth,
            cross_linear=cross_linear,
            block_fn=block_fn,
            dtype=dtype,
            name=f"stage_{stage_idx}",
        )(stages)
        pyramid_outputs[f"P{stage_idx} + 2"] = stages
    return stages, pyramid_outputs


def _pad_arg(x, n):
    # pads an argument tuple to specified n by padding with last value
    if not isinstance(x, (tuple, list)):
        x = (x,)
    curr_n = len(x)
    pad_n = n - curr_n
    if pad_n <= 0:
        return x[:n]
    return tuple(x + (x[-1],) * pad_n)
