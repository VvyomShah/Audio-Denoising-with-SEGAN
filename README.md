# Audio-Denoising-with-SEGAN
Audio de-noiser for 8KHz audio files using SEGAN.

## Pre-requisites

- Python 3.x
- PyTorch 1.6.0
- Libraries in requirements.txt

run `pip install -r requirements.txt`

## Inference

- Create a folder `checkpoints` and place downloaded model from [here](https://drive.google.com/file/d/1-JCbBGEDCvrkBoSIsvfqiIvvivGuoRy_/view?usp=sharing)
- Create an empty folder `output`.
- Create an enpty folder `input` to store all noisy audio.
- Run `python test.py`.
- Clean audio will be stored in `output`.

## Credits

- https://github.com/santi-pdp/segan_pytorch
- https://github.com/dansuh17/segan-pytorch
- https://github.com/leftthomas/SEGAN
