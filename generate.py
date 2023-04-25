import argparse, uuid, subprocess
import torch
from model import MusicTransformer
from preprocess import SequenceEncoder
from preprocess import PreprocessingPipeline
from helpers import sample, write_midi
from helpers import vectorize
import pretty_midi
import six

class GeneratorError(Exception):
    pass

def main():
    parser = argparse.ArgumentParser("Script to generate MIDI tracks by sampling from a trained model.")

    parser.add_argument("--model", type=str, 
            help="Key in saved_models/model.yaml, helps look up model arguments and path to saved checkpoint.")
    parser.add_argument("--sample_length", type=int, default=512,
            help="number of events to generate")
    parser.add_argument("--temps", nargs="+", type=float, 
            default=[1.0],
            help="space-separated list of temperatures to use when sampling")
    parser.add_argument("--n_trials", type=int, default=3,
            help="number of MIDI samples to generate per experiment")
    parser.add_argument("--live_input", action='store_true', default = False,
            help="if true, take in a seed from a MIDI input controller")

    parser.add_argument("--play_live", action='store_true', default=False,
            help="play sample(s) at end of script if true")
    parser.add_argument("--keep_ghosts", action='store_true', default=False)
    parser.add_argument("--stuck_note_duration", type=int, default=0)

    args=parser.parse_args()

    model = args.model
    
    '''
    try:
        model_dict = yaml.safe_load(open('saved_models/model.yaml'))[model_key]
    except:
        raise GeneratorError(f"could not find yaml information for key {model_key}")
    '''
    
    model_path = 'saved_models/'+model
    # model_args = model_dict["args"]
    try:
        state = torch.load(model_path)
    except RuntimeError:
        state = torch.load(model_path, map_location="cpu")
    
    n_velocity_events = 32
    n_time_shift_events = 125

    decoder = SequenceEncoder(n_time_shift_events, n_velocity_events,
           min_events=0)

    if args.live_input:
        print("Expecting a midi input...")
        pretty_midis = []
        m = 'twinkle.mid'
        with open(m, "rb") as f:
                try:
                    midi_str = six.BytesIO(f.read())
                    pretty_midis.append(pretty_midi.PrettyMIDI(midi_str))
                    #print("Successfully parsed {}".format(m))
                except:
                    print("Could not parse {}".format(m))
        pipeline = PreprocessingPipeline(input_dir="data")
        note_sequence = pipeline.get_note_sequences(pretty_midis)
        note_sequence = [vectorize(ns) for ns in note_sequence]
        prime_sequence = decoder.encode_sequences(note_sequence)
        prime_sequence = prime_sequence[1:6]

    else:
        prime_sequence = []

    # model = MusicTransformer(**model_args)
    model = MusicTransformer(256+125+32, 1024, 
            d_model = 64, n_heads = 8, d_feedforward=256, 
            depth = 4, positional_encoding=True, relative_pos=True)
    model.load_state_dict(state, strict=False)

    temps = args.temps

    trial_key = str(uuid.uuid4())[:6]
    n_trials = args.n_trials

    keep_ghosts = args.keep_ghosts
    stuck_note_duration = None if args.stuck_note_duration == 0 else args.stuck_note_duration

    for temp in temps:
        print(f"sampling temp={temp}")
        note_sequence = []
        for i in range(n_trials):
            print("generating sequence")
            output_sequence = sample(model, prime_sequence = prime_sequence, sample_length=args.sample_length, temperature=temp)
            note_sequence = decoder.decode_sequence(output_sequence, 
                verbose=True, stuck_note_duration=1, keep_ghosts=True)

            output_dir = f"output/midis/{trial_key}/"
            file_name = f"sample{i+1}_{temp}"
            write_midi(note_sequence, output_dir, file_name)
    '''
    for temp in temps:      
        try:
            subprocess.run(['timidity', f"output/{model_key}/{trial_key}/sample{i+1}_{temp}.midi"])
        except KeyboardInterrupt:
            continue
    '''
if __name__ == "__main__":
    main()
