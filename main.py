import click
from audio_processing import split_audio
from decorators import log_time
from diarization import diarize_audio
from helpers import load_pkl, parse_input_file_path, save_as_pkl, select_language, write_text_to_file
from transcript_to_meeting_minutes import transcript_to_meeting_minutes
from transcription import load_transcript, transcript_audio, save_transcript


@log_time
def create_meeting_minutes(audio_source, language, openai):
    diarization = diarize_audio(audio_source)
    save_as_pkl(diarization, "diarization.pkl")

    diarization = load_pkl("diarization.pkl")
    audio_sections = split_audio(audio_source, diarization)
    transcript_sections = transcript_audio(audio_sections, language)
    save_as_pkl(transcript_sections, "transcript.pkl")
    save_transcript(audio_sections, transcript_sections, "transcript.txt")

    transcript = load_transcript("transcript.txt")
    meeting_minutes = transcript_to_meeting_minutes(transcript, language, openai)
    write_text_to_file(meeting_minutes, "meeting_minutes.txt")
    print(meeting_minutes)


@click.command()
@click.option("-f", "--input-file", help="Path to the input audio file.")
@click.option(
    "-l",
    "--language",
    show_default=True,
    default="english",
    help="Select the language in which the meeting minutes should be generated. Currently supproted are 'english' and 'german'.",
)
@click.option(
    "--openai",
    is_flag=True,
    show_default=True,
    default=False,
    help="If set Meminto will use OpenAis gpt-3.5-turbo as default LLM. Otherwise it will use LLM specified in enviroment variables.",
)
def main(input_file, language, openai) -> None:
    audio_source = parse_input_file_path(input_file)
    language = select_language(language)
    print(language)
    create_meeting_minutes(audio_source, language, openai)


if __name__ == "__main__":
    main()
