import keras
from keras import ops
from keras_hub.src.layers.preprocessing.preprocessing_layer import PreprocessingLayer
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.api_export import keras_hub_export

@keras_hub_export("keras_hub.layers.VideoConverter")
class VideoConverter(PreprocessingLayer):
    """Base class for video preprocessing layers.

    `VideoConverter` tasks handle resizing and normalizing video inputs.
    It delegates to `ImageConverter` for frame-level processing.

    Args:
        image_size: The target size of the frames.
        scale: The scale factor to apply to pixels.
        offset: The offset to apply to pixels.
    """

    def __init__(
        self,
        image_size=None,
        scale=None,
        offset=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_converter = ImageConverter(
            image_size=image_size, scale=scale, offset=offset, **kwargs
        )
        self.image_size = image_size
        self.scale = scale
        self.offset = offset

    def call(self, inputs):
        if isinstance(inputs, list):
            # List of videos (each video is a 4D tensor or list of frames)
            outputs = []
            for video in inputs:
                out = self.image_converter(video)
                outputs.append(out)
            return outputs
        else:
            # Assume 5D tensor (batch, frames, height, width, channels)
            shape = ops.shape(inputs)
            batch_size = shape[0]
            num_frames = shape[1]
            h, w, c = shape[2], shape[3], shape[4]

            # Flatten batch and temporal dims
            flat_inputs = ops.reshape(inputs, (batch_size * num_frames, h, w, c))

            # Process frames as images
            flat_outputs = self.image_converter(flat_inputs)

            # Reshape back to 5D
            output_shape = ops.shape(flat_outputs)
            outputs = ops.reshape(
                flat_outputs,
                (batch_size, num_frames, output_shape[1], output_shape[2], output_shape[3]),
            )
            return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "scale": self.scale,
                "offset": self.offset,
            }
        )
        return config
