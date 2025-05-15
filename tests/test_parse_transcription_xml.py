import pytest
from pathlib import Path
import pandas as pd
from xml.etree.ElementTree import Element, SubElement, tostring
from src.parse_transcriptions_xml import (
    parse_xml,
    process_segment,
    save_annotations,
    load_audio_chunk,
    save_audio_chunk,
)

@pytest.fixture
def sample_xml_file(tmp_path):
    """Creates a temporary XML file for testing."""
    xml_content = """<ROOT>
        <S>
            <FORM>Test sentence 1</FORM>
            <AUDIO start="0.0" end="2.0" />
        </S>
        <S>
            <FORM>Test sentence 2</FORM>
            <AUDIO start="2.0" end="4.0" />
        </S>
    </ROOT>"""
    xml_file = tmp_path / "test.xml"
    xml_file.write_text(xml_content)
    return xml_file


@pytest.fixture
def sample_audio_file(tmp_path):
    """Creates a temporary WAV file for testing."""
    audio_file = tmp_path / "test.wav"
    # Generate a silent audio file for testing
    from pydub.generators import Sine
    silent_audio = Sine(440).to_audio_segment(duration=4000)  # 4 seconds
    silent_audio.export(audio_file, format="wav")
    return audio_file


def test_parse_xml(sample_xml_file):
    """Test parsing an XML file."""
    root = parse_xml(sample_xml_file)
    assert root.tag == "ROOT"
    segments = root.findall("S")
    assert len(segments) == 2
    assert segments[0].find("FORM").text == "Test sentence 1"


def test_process_segment(sample_audio_file, tmp_path):
    """Test processing a single XML segment."""
    segment = Element("S")
    SubElement(segment, "FORM").text = "Test sentence"
    SubElement(segment, "AUDIO", start="0.0", end="2.0")

    output_dir = tmp_path / "output"
    result = process_segment(segment, sample_audio_file, output_dir, segment_id=1)

    assert result["sentence"] == "Test sentence"
    assert Path(result["path"]).exists()


def test_save_annotations(tmp_path):
    """Test saving annotations to a TSV file."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    df = pd.DataFrame([{"path": "chunk1.wav", "sentence": "Test sentence 1"}])
    xml_file = tmp_path / "test.xml"

    save_annotations(df, output_dir, xml_file)

    tsv_file = output_dir / "test_annotations.tsv"
    assert tsv_file.exists()

    saved_df = pd.read_csv(tsv_file, sep="\t")
    assert len(saved_df) == 1
    assert saved_df.iloc[0]["sentence"] == "Test sentence 1"


def test_load_audio_chunk(sample_audio_file):
    """Test loading an audio chunk."""
    chunk = load_audio_chunk(sample_audio_file, start_time=0.0, end_time=2.0)
    assert len(chunk) == 2000  # 2 seconds in milliseconds


def test_save_audio_chunk(sample_audio_file, tmp_path):
    """Test saving an audio chunk."""
    chunk = load_audio_chunk(sample_audio_file, start_time=0.0, end_time=2.0)
    output_dir = tmp_path / "output"
    chunk_path = save_audio_chunk(chunk, output_dir, sample_audio_file, segment_id=1)

    assert chunk_path.exists()
    assert chunk_path.suffix == ".wav"