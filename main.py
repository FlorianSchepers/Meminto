from diarization import diarize_audio

def create_meeting_minutes(audio_source, language):
    diarization = diarize_audio(audio_source)
    print(diarization)

def main()->None:
    audio_source = r"examples\Scoreboard.wav"
    language = "english"
    create_meeting_minutes(audio_source, language)

if __name__ == "__main__":
    main()