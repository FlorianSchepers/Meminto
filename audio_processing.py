import torchaudio

from decorators import log_time


SAMPLING_RATE = 16000

def batch(iterable,n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx+n, l)] 

def load_audio(audio_path):
  audio, sr = torchaudio.load(audio_path)
  number_of_channels = audio.size()[0]
  if number_of_channels > 1:
     audio = audio[0]
  resampler = torchaudio.transforms.Resample(sr, SAMPLING_RATE)
  audio_resampled = resampler(audio)
  return audio_resampled.squeeze()

@log_time
def split_audio(audio_source, diarization):
  audio = load_audio(audio_source.resolve()) 
  
  audio_sections = []
  for turn, _, speaker in diarization.itertracks(yield_label=True):
    audio_sections.append({"speaker": speaker, "turn": turn, "audio": audio[int(SAMPLING_RATE*round(turn.start, 1)):int(SAMPLING_RATE*round(turn.end, 1))]})
  return audio_sections