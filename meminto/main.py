import os
from pathlib import Path
import click
from meminto.llm.tokenizers import Tokenizer
from meminto.audio_processing import split_audio
from meminto.decorators import log_time
from meminto.diarizer import Diarizer
from meminto.helpers import (
    Language,
    load_pkl,
    parse_input_file_path,
    parse_output_folder_path,
    save_as_pkl,
    select_language,
    write_text_to_file,
)
from meminto.llm.llm import LLM
from meminto.meeting_minutes_generator import (
    MeetingMinutesGenerator,
)
from meminto.transcriber import (
    Transcriber,
    save_transcript_as_txt,
)
from dotenv import load_dotenv

EXAMPLE_INPUT_FILE = Path(__file__).parent.resolve() / "../examples/Scoreboard.wav"
DEFAULT_OUTPUT_FOLDER = Path(__file__).parent.resolve() / "../output"
DEFAULT_LANGUAGE = Language.ENGLISH


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
    default=DEFAULT_LANGUAGE,
    help="Select the language in which the meeting minutes should be generated. Currently supproted are 'english' and 'german'.",
)
def main(input_file: str, output_folder: str, language: str) -> None:
    load_dotenv()
    audio_input_file_path = parse_input_file_path(input_file)
    output_folder_path = parse_output_folder_path(output_folder)
    selected_language = select_language(language)
    create_meeting_minutes(audio_input_file_path, output_folder_path, selected_language)


@log_time
def create_meeting_minutes(
    audio_input_file_path: Path, output_folder_path: Path, language: Language
):
    diarizer = Diarizer(
        model="pyannote/speaker-diarization@2.1",
        hugging_face_token=os.environ["HUGGING_FACE_ACCESS_TOKEN"],
    )
    diarization = diarizer.diarize_audio(audio_input_file_path)

    for speech_turn, track, speaker in diarization.itertracks(yield_label=True):
        print(f"{speech_turn.start:4.1f} {speech_turn.end:4.1f} {speaker}")

    save_as_pkl(diarization, output_folder_path / "diarization.pkl")

    diarization = load_pkl(output_folder_path / "diarization.pkl")
    audio_sections = split_audio(audio_input_file_path, diarization)

    transcriber = Transcriber()
    transcript = transcriber.transcribe(audio_sections)

    save_as_pkl(transcript, output_folder_path / "transcript.pkl")
    save_transcript_as_txt(transcript, output_folder_path / "transcript.txt")

    tokenizer = Tokenizer(
        os.environ["LLM_MODEL"],
        hugging_face_acces_token=os.environ["HUGGING_FACE_ACCESS_TOKEN"],
    )
    llm = LLM(
        model=os.environ["LLM_MODEL"],
        url=os.environ["LLM_URL"],
        authorization=os.environ["LLM_AUTHORIZATION"],
        temperature=0.5,
        max_tokens=int(os.environ["LLM_MAX_TOKENS"]),
    )

    transcript = load_pkl(output_folder_path / "transcript.pkl")
    meeting_minutes_generator = MeetingMinutesGenerator(tokenizer=tokenizer, llm=llm)
    meeting_minutes = meeting_minutes_generator.generate(
        transcript=transcript, language=language
    )

    write_text_to_file(meeting_minutes, output_folder_path / "meeting_minutes.txt")


if __name__ == "__main__":
    main()
