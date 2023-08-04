from audio_processing import split_audio
from diarization import diarize_audio, load_diarization, save_diarization


def create_meeting_minutes(audio_source, language):
    diarization = diarize_audio(audio_source)
    save_diarization(diarization, 'diarization.pkl')
    diarization = load_diarization('diarization.pkl')

    audio_sections = split_audio(audio_source, diarization)
    print(f"Number of audio sections: {len(audio_sections)}")

def main()->None:
    audio_source = r"examples\Scoreboard.wav"
    language = "english"
    create_meeting_minutes(audio_source, language)

if __name__ == "__main__":
    main()