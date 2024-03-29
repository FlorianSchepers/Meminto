from pathlib import Path
import click
from audio_processing import split_audio
from decorators import log_time
from diarization import diarize_audio
from helpers import (
    Language,
    load_pkl,
    parse_input_file_path,
    parse_output_folder_path,
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

EXAMPLE_INPUT_FILE = Path(__file__).parent.resolve() / "examples/Scoreboard.wav"
DEFAULT_OUTPUT_FOLDER = Path(__file__).parent.resolve() / "output"


@log_time
def create_meeting_minutes(audio_input_file_path: Path, output_folder_path: Path, language :Language):
    diarization = diarize_audio(audio_input_file_path)
    save_as_pkl(diarization, output_folder_path / "diarization.pkl")

    diarization = load_pkl(output_folder_path / "diarization.pkl")
    audio_sections = split_audio(audio_input_file_path, diarization)
    transcript = create_transcript(audio_sections, language)
    save_as_pkl(transcript, output_folder_path / "transcript.pkl")
    save_transcript_as_txt(transcript, output_folder_path / "transcript.txt")
    
    transcript = load_pkl(
        output_folder_path / "transcript.pkl"
    )
    merged_meeting_minutes, batched_meeting_minutes = transcript_to_meeting_minutes(
        transcript, language
    )
    write_text_to_file(
        batched_meeting_minutes_to_text(batched_meeting_minutes),
        output_folder_path / "batched_meeting_minutes.txt",
    )
    write_text_to_file(
        merged_meeting_minutes, output_folder_path / "meeting_minutes.txt"
    )


@click.command()
@click.option(
    "-f",
    "--input-file",
    show_default=True,
    default=EXAMPLE_INPUT_FILE,
    help="Path to the input audio file.",
)
@click.option(
    "-o",
    "--output-folder",
    show_default=True,
    default=DEFAULT_OUTPUT_FOLDER,
    help="Path to the folder where the output files are stored.",
)
@click.option(
    "-l",
    "--language",
    show_default=True,
    default="english",
    help="Select the language in which the meeting minutes should be generated. Currently supproted are 'english' and 'german'.",
)
def main(input_file: str, output_folder: str, language :str) -> None:
    audio_input_file_path = parse_input_file_path(input_file)
    output_folder_path = parse_output_folder_path(output_folder)
    selected_language = select_language(language)
    create_meeting_minutes(audio_input_file_path, output_folder_path, selected_language)


if __name__ == "__main__":
    main()
