import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from pydub import AudioSegment
import pandas as pd
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_audio_chunk(audio_file, start_time, end_time):
    """Loads a chunk of audio from start_time to end_time."""
    audio = AudioSegment.from_wav(audio_file)
    return audio[start_time * 1000:end_time * 1000]  # pydub uses milliseconds


def save_audio_chunk(chunk, output_dir, audio_file, sentence_id, format="wav"):
    """Saves an audio chunk to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = output_dir / f"{Path(audio_file).stem}_s{sentence_id}.{format}"
    chunk.export(chunk_path, format=format)
    logging.info(f"Saved audio chunk: {chunk_path}")
    return chunk_path


def parse_xml(xml_file):
    """Parses the XML file and returns the root element."""
    try:
        tree = ET.parse(xml_file)
        return tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML file '{xml_file}': {e}")


def process_sentence(sentence, audio_file, output_dir, sentence_id):
    """Processes a single XML sentence and saves the corresponding audio chunk."""
    text = sentence.find('FORM').text
    audio_tag = sentence.find('AUDIO')
    start_time = float(audio_tag.get('start'))
    end_time = float(audio_tag.get('end'))
    logging.info(f"Processing sentence {sentence_id}: {start_time} to {end_time}, Text: {text}")

    chunk = load_audio_chunk(audio_file, start_time, end_time)
    chunk_path = save_audio_chunk(chunk, output_dir, audio_file, sentence_id)
    return {"path": str(chunk_path), "sentence": text}


def save_annotations(df, output_dir, xml_file):
    """Saves the annotations DataFrame to a TSV file."""
    tsv_path = output_dir / f"{Path(xml_file).stem}_annotations.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    logging.info(f"Annotations saved to {tsv_path}")


def parse_xml_and_extract_audio(xml_file, output_dir):
    """Main function to parse XML, process sentences, and save audio chunks and annotations."""
    root = parse_xml(xml_file)

    audio_file = Path(xml_file).with_suffix(".wav")
    output_dir = output_dir / Path(xml_file).stem
    rows = []
    sentence_id = 1

    for sentence in root.findall("S"):
        try:
            result = process_sentence(sentence, audio_file, output_dir, sentence_id)
            rows.append(result)
        except Exception as e:
            logging.error(f"Error processing sentence {sentence_id}: {e}")
        sentence_id += 1

    if not rows:
        raise ValueError("No valid sentences were processed.")

    df = pd.DataFrame(rows)
    save_annotations(df, output_dir, xml_file)
    logging.info("All audio chunks and annotations have been successfully saved.")

def extract_sentence_words(sentence_elem, sep="|"):
    """
    Concatenate all <W><FORM> elements in an <S> element into a sentence string.
    Words are separated by `sep` (default: '|').
    """
    word_forms = []
    for w in sentence_elem.findall('W'):
        form = w.find('FORM')
        if form is not None and form.text:
            word_forms.append(form.text.strip())
    return sep.join(word_forms)

def main():
    parser = argparse.ArgumentParser(description="Parse a Pangloss XML annotation file and create audio chunks.")
    parser.add_argument("xml_file", type=str, help="Path to the Pangloss XML annotation file")
    parser.add_argument("--output_dir", type=str, default="./output", help="Base directory to save the audio chunks and annotations")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    try:
        parse_xml_and_extract_audio(args.xml_file, output_dir)
    except Exception as e:
        logging.error(f"Failed to process XML file: {e}")
        exit(1)


if __name__ == "__main__":
    main()
