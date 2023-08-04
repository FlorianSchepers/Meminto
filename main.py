from audio_processing import split_audio
from diarization import diarize_audio, load_diarization, save_diarization
from transscription import transscript_audio


def create_meeting_minutes(audio_source, language):
    diarization = diarize_audio(audio_source)
    save_diarization(diarization, 'diarization.pkl')
    
    diarization = load_diarization('diarization.pkl')
    audio_sections = split_audio(audio_source, diarization)

    transscript_sections = transscript_audio(audio_sections, language)

    for idx, section in enumerate(audio_sections):
        print(f"start={section['turn'].start:.1f}s stop={section['turn'].end:.1f}s speaker_{section['speaker']}:\n")
        for batch in transscript_sections[idx]:
            print(batch.strip())
        print("\n")

def main()->None:
    audio_source = r"examples\Scoreboard.wav"
    language = "english"
    create_meeting_minutes(audio_source, language)

if __name__ == "__main__":
    main()