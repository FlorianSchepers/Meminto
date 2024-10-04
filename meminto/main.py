import os
from pathlib import Path
import click
import torchaudio
from meminto.llm.tokenizers import Tokenizer
from meminto.audio_processing import load_audio, save_audio, split_audio
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
from meminto.transcriber import LocalTranscriber, RemoteTranscriber
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
@click.option(
    "-rt",
    "--remote-transcriber",
    is_flag=True,
    show_default=True,
    default=False,
    help="If selected the Meminto will use the remote transcriber. The enviroment variables 'TRANSCRIBER_URL' and 'TRANSCRIBER_AUTHORIZATION' need to be set.",
)
def main(
    input_file: str, output_folder: str, language: str, remote_transcriber: bool
) -> None:
    load_dotenv()
    audio_input_file_path = parse_input_file_path(input_file)
    output_folder_path = parse_output_folder_path(output_folder)
    selected_language = select_language(language)
    create_meeting_minutes(
        audio_input_file_path, output_folder_path, selected_language, remote_transcriber
    )


@log_time
def create_meeting_minutes(
    audio_input_file_path: Path,
    output_folder_path: Path,
    language: Language,
    remote_transcriber: bool,
):
    audio_output_file_path = "examples/tmp.wav"
    audio_file = load_audio(audio_input_file_path)
    save_audio(audio_file, audio_output_file_path)
    audio_file = load_audio(audio_output_file_path)
    #audio_input_file_path = audio_output_file_path
    ### Diarization ###
    diarizer = Diarizer(
        model="pyannote/speaker-diarization@2.1",
        hugging_face_token=os.environ["HUGGING_FACE_ACCESS_TOKEN"],
    )
    diarization = diarizer.diarize_audio(audio_input_file_path)

    diarization_text = diarizer.diarization_to_text(diarization)
    write_text_to_file(diarization_text, output_folder_path / "diarization.txt")
    save_as_pkl(diarization, output_folder_path / "diarization.pkl")

    ### Transcription ###
    diarization = load_pkl(output_folder_path / "diarization.pkl")
    audio_sections = split_audio(audio_input_file_path, diarization)

    if remote_transcriber:
        print("Using RemoteTranscriber.")
        transcriber = RemoteTranscriber(
            url=os.environ["TRANSCRIBER_URL"],
            authorization=os.environ["TRANSCRIBER_AUTHORIZATION"],
        )
    else:
        print("Using LocalTranscriber.")
        transcriber = LocalTranscriber()
    print(audio_sections)
    transcript = transcriber.transcribe(audio_sections)

    transcript_text = transcriber.transcript_to_txt(transcript)
    write_text_to_file(transcript_text, output_folder_path / "transcript.txt")
    save_as_pkl(transcript, output_folder_path / "transcript.pkl")

    ### Generation ###
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
