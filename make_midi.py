import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig
import mido

tokenizer = None
bert_model = None
class SonificationModel(nn.Module):
    def __init__(self, bert_model):
        super(SonificationModel, self).__init__()
        self.bert_model = bert_model
        
    def forward(self, X):
        x = self.bert_model(**X, output_hidden_states=True)[1]
        return [torch.tanh(h) for h in x]

def get_embeddings(text):
    global bert_model, tokenizer
    
    batch = tokenizer(text, return_tensors="pt")

    model = SonificationModel(bert_model)
    with torch.no_grad():    
        outs = model(batch) 
    
    data = torch.zeros(7,768)
    for idx, hidden_x in enumerate(outs):
        data[idx, :] = hidden_x[0,0,:]
    data = data.numpy()
    return data

def convert_note(note):
    notes = "C,Db,D,Eb,E,F,Gb,G,Ab,A,Bb,B".split(",")
    numbers = list(range(60,72))  
    try:
        octave = int(note[-1])
        note_idx = notes.index(note[:-1])
    except:
        octave = 4
        note_idx = notes.index(note)
    offset = (octave-4)*12    
    return numbers[note_idx] + offset

def map_note(value, root="C4", scale_type="major"):
    """
    value: a numeric value ranged from 0 to 1
    root: the root note used as the reference
    """
    root_number = convert_note(root)
    disct = (value//0.1).astype(np.int32)
    if scale_type == "major":
        scale_interval = [2,2,1,2,2,2,1]    
    elif scale_type == "natural":
        scale_interval = [2,1,2,2,1,2,2]    
    else:
        raise ValueError("only support major and natural")
    scale = np.cumsum([*(scale_interval*2)])
    
    return scale[disct] + root_number

def map_chord(value, ref):
    value0 = (value-np.mean(value))/np.std(value)
    value = np.maximum(np.minimum(value0, 3), -3) + 3
    disct = value.astype(np.int32)
    intervals = np.array([5, 7, 4, 4, 7, 7, 5])
    
    return ref-intervals[disct]

def map_duration(value):
    value0 = (value-np.mean(value))/np.std(value)
    value = np.maximum(np.minimum(value0, 3), -3) + 3
    duration = np.array(
        [0.5]*3 + [1]*2 + [1.5]*2
    )
    disct = value.astype(np.int32)
    return duration[disct], disct

def map_track1_data(melody_data, chord_data, duration_data, root, scale_type):
    notes = map_note(melody_data, root, scale_type)
    chord = map_chord(chord_data, notes)
    duration, value0 = map_duration(duration_data)
    track = [notes, chord, duration]
    return track

def map_track2_data(melody_data, chord_data, duration_data, ref_notes, base_offset=-12):
    notes = map_chord(melody_data, ref_notes+base_offset)
    chord = map_chord(chord_data, notes)
    duration, value0 = map_duration(duration_data)
    track = [notes, chord, duration]
    return track

def make_piano_track(track_data, ticks_per_beat, bpm=120, velocity=80, program=1):    
    track = mido.MidiTrack()
    track.append(mido.Message("program_change", program=program))
    for note1, note2, beat in zip(track_data[0], track_data[1], track_data[2]):
        ticks = int(mido.second2tick((60/bpm)*beat, ticks_per_beat, mido.bpm2tempo(120)))
        msg11 = mido.Message('note_on', note=note1, velocity=velocity, time=0)
        msg12 = mido.Message('note_on', note=note2, velocity=velocity, time=0)
        msg21 = mido.Message('note_off', note=note1, velocity=velocity, time=ticks)
        msg22 = mido.Message('note_off', note=note2, velocity=velocity, time=0)            
        track.append(msg11)        
        track.append(msg12)
        track.append(msg21)        
        track.append(msg22)        
    return track

def make_string_track(track_data, ticks_per_beat, bpm=120, velocity=80, program=1):    
    track = mido.MidiTrack()
    track.append(mido.Message("program_change", program=program, channel=2))
    mask_gate = 0
    acc_ticks = 0
    playing_note = 0
    for note1, beat in zip(track_data[0], track_data[2]):
        ticks = int(mido.second2tick((60/bpm)*beat, ticks_per_beat, mido.bpm2tempo(120)))
        acc_ticks += ticks        
        if mask_gate == 0:
            msg11 = mido.Message('note_on', note=note1, velocity=velocity, time=0, channel=2)
            track.append(msg11)
            playing_note = note1 
            mask_gate += 1
        elif mask_gate == 2:
            msg21 = mido.Message('note_off', note=playing_note, velocity=velocity, time=acc_ticks, channel=2)
            track.append(msg21)
            mask_gate = 0
            acc_ticks = 0        
        else:
            mask_gate += 1
    track.append(mido.Message('note_off', note=playing_note, velocity=velocity, time=acc_ticks, channel=2))
    return track

def make_tune(text, scale="G", scale_type="major", outname="out", clip_length=128):    

    data = get_embeddings(text)
    track1 = map_track1_data(data[6,:], data[5,:], data[4,:], root=scale, scale_type=scale_type)
    track2 = map_track2_data(data[3,:], data[2,:], data[1,:], ref_notes=track1[0])
    def make_clip(length, track):
        return [x[:length] for x in track]    
    track1 = make_clip(clip_length, track1)
    track2 = make_clip(clip_length, track2)
    mid = mido.MidiFile()
    bpm = 112
    tpb = 64
    mid.ticks_per_beat=tpb
    mid.tracks.append(make_piano_track(track1, ticks_per_beat=tpb, bpm=bpm, program=1))
    mid.tracks.append(make_piano_track(track2, ticks_per_beat=tpb, bpm=bpm, velocity=64, program=1))
    mid.tracks.append(make_string_track(track2, ticks_per_beat=tpb, bpm=bpm, velocity=64, program=9))

    outpath = f"{outname}.{scale}.{scale_type}.mid"
    mid.save(outpath)
    print("Create tune of %s to %s" % (text, outpath))    

def init_model():
    global tokenizer, bert_model
    MODEL_NAME = "distilbert-base-multilingual-cased"    
    print("Loading Bert...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    bert_model = DistilBertModel.from_pretrained(MODEL_NAME)
    print("Done")

if __name__ == "__main__":    
    init_model()
    make_tune("白日AI唱夢", scale="G", outname="seq1", clip_length=768)
    make_tune("LOPE-Dream-pop", scale="G", outname="seq2", clip_length=768)