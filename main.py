from audio_processing import split_audio
from diarization import diarize_audio, load_diarization, save_diarization
from transscript_to_meeting_minutes import transscript_to_meeting_minutes
from transscription import load_transscript, transscript_audio, save_transscript


def create_meeting_minutes(audio_source, language):
    diarization = diarize_audio(audio_source)
    save_diarization(diarization, 'diarization.pkl')
    
    diarization = load_diarization('diarization.pkl')
    audio_sections = split_audio(audio_source, diarization)
    transscript_sections = transscript_audio(audio_sections, language)
    save_transscript(audio_sections, transscript_sections, 'transscript.txt')
    
    transscript = load_transscript('transscript.txt')
    meeting_minutes = transscript_to_meeting_minutes(transscript, language)
    print(meeting_minutes)

def main()->None:
    audio_source = r"examples\Scoreboard.wav"
    language = "english"
    create_meeting_minutes(audio_source, language)

if __name__ == "__main__":
    main()