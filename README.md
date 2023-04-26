# PianoGenie
A transformer based model that generates Music Sequences. Our goal for the music generation system is to enable musicians, composers, and music enthusiasts to easily generate original music pieces, while also pushing the boundaries of what is possible with artificial intelligence and music composition.

### About:
A modified implementation of Pno-Ai's version of Google Magenta's [Music Transformer](https://magenta.tensorflow.org/music-transformer) in Python/Pytorch. This library is designed to train a neural network on Piano MIDI data to generate musical samples. MIDIs are encoded into "Event Sequences", a dense array of musical instructions (note on, note off, dynamic change, time shift) encoded as numerical tokens. A custom transformer model learns to predict instructions on training sequences, and in `generate.py` a trained model can randomly sample from its learned distribution. (It is recommended to 'prime' the model's internal state with a MIDI input.)

[Get the report ->](https://github.com/Zedx07/PianoGenie/files/11328476/Report_gp_09_Shubham_Krutik_Vatsal.pdf)

### Training Data:
The initial dataset comes from several years of recordings from the International Piano-e-Competition: over 1,000 performances played by professional pianists on a Yamaha Disklavier. Obtainable [here](https://magenta.tensorflow.org/datasets/maestro). A sufficiently large dataset (order of 50 MB) of piano MIDIs should be sufficient to train a model.

### Deploying Model:
Navigate to the unzipped folder and create an environment with suitable dependencies

`conda env create -n new-env`

`conda activate new-env`

Move files into the `data` folder for training. Create the folder if it doesn't already exist.

Train a model with

`python run.py`

You can also train the model on previously trainde model by using parameter:
  `--checkpoint`: Optional path to saved model, if none provided, the model is trained from scratch.

Find a model in the saved_models directory, copy its name

Generate new samples with

`python generate.py --model <model name>`

  - Additional parameters:
  
    `--input_live` : use a file `twinkle.midi` from the main directory when generating new samples
    
    `--temps <float>` : control the temperature value of the selection for token generation
    
    `--n_trials <int>` : control the number of samples produced
    
    `--stuck_note_duration <float>` : control when to end notes that do note come with a NOTE_OFF
