import click
from audio_processing import split_audio
from decorators import log_time
from diarization import diarize_audio
from helpers import (
    load_pkl,
    parse_input_file_path,
    save_as_pkl,
    select_language,
    write_text_to_file,
)
from transcript_to_meeting_minutes import (
    batched_meeting_minutes_to_text,
    transcript_to_meeting_minutes,
)
from transcription import (
    TranscriptSection,
    create_transcript,
    save_transcript_as_txt,
)


@log_time
def create_meeting_minutes(audio_source, language):
    # diarization = diarize_audio(audio_source)
    # save_as_pkl(diarization, "output/diarization.pkl")

    # diarization = load_pkl("output/diarization.pkl")
    # audio_sections = split_audio(audio_source, diarization)
    # transcript = create_transcript(audio_sections, language)
    # save_as_pkl(transcript, "output/transcript.pkl")
    # save_transcript_as_txt(transcript, "output/transcript.txt")

    transcript: list[TranscriptSection] = load_pkl("output/transcript.pkl")
    merged_meeting_minutes, batched_meeting_minutes = transcript_to_meeting_minutes(
        transcript, language
    )
    print(merged_meeting_minutes)
    write_text_to_file(
        batched_meeting_minutes_to_text(batched_meeting_minutes),
        "output/batched_meeting_minutes.txt",
    )
    write_text_to_file(merged_meeting_minutes, "output/meeting_minutes.txt")


@click.command()
@click.option("-f", "--input-file", help="Path to the input audio file.")
@click.option(
    "-l",
    "--language",
    show_default=True,
    default="english",
    help="Select the language in which the meeting minutes should be generated. Currently supproted are 'english' and 'german'.",
)
def main(input_file, language) -> None:
    audio_source = parse_input_file_path(input_file)
    language = select_language(language)
    print(language)
    create_meeting_minutes(audio_source, language)


if __name__ == "__main__":
    main()
