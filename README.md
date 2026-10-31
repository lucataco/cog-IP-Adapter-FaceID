# h94/IP-Adapter-FaceID Cog model

This is an implementation of [h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own model to [Replicate](https://replicate.com).

## Basic Usage

Run a prediction

    cog predict -i face_image=@demo.png

## Output

![red-dress](output.0.png)

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE).

The model weights are not covered by this license. [IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID) is released for non-commercial research use only, because the InsightFace face models it depends on are non-commercial.
