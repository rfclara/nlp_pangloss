import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from pydub import AudioSegment
import pandas as pd
import logging
"""<S id="Agriculture_S093">
    <AUDIO start="420.5601" end="423.8243" />
    <FORM>"nɑ˩-dʑi˧-dʑo˩, | sɑ˧! | nɑ˩hɑ˧-dʑo˩, | wo˩" pi˥-kv̩˩-tsɯ˩ mv̩˩! |</FORM>
    <NOTE xml:lang="fr" message="d'abord noté /‡wo˧, pi˩-kv̩˩ tsɯ˩ mv̩˩/, avec M23; puis corrigé en: /‡nɑ˩hɑ˧-dʑo˩, wo˩ | pi˥-kv̩˩-tsɯ˩ ◊ -mv̩˩/. En fait, /wo˩/ et /pi˥/ se trouvent intégrés dans le même groupe tonal." />
    <NOTE xml:lang="fr" message="Le ton du monosyllabe n'a pu être élicité, car en dehors de cette formule proverbiale il n'est pas usité: il faut dire /dʑi˧hv̩͂#˥/. Le ton du composé /nɑ˩-dʑi˧/ ne nous apprend pas grand'chose car c'est ce ton qui serait obtenu que le ton du 2e élément (dʑi) soit M, L ou #H. Dans l'annotation, le composé n'est donc pas analysé." />
    <TRANSL xml:lang="fr">le vêtement des Na, c'est le lin; la nourriture des Na, c'est le navet, voilà ce qu'on dit!</TRANSL>
    <TRANSL xml:lang="zh">“纳人的衣服，是麻布！纳人的饭（菜），是圆根叶子”，有这么一个说法。</TRANSL>
    <W>
      <FORM>nɑ˩dʑi˧</FORM>
      <TRANSL xml:lang="fr">le_vêtement_des_Na</TRANSL>
    </W>"""
# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_chunk(audio_file, start_time, end_time, output_dir, segment_id, format):
    """Saves a chunk of audio from start_time to end_time, names it using segment_id."""
    try:
        audio_path = Path(audio_file)
        audio = AudioSegment.from_wav(audio_file)
        chunk = audio[start_time*1000:end_time*1000]  # pydub uses milliseconds
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = output_dir / f"{audio_path.stem}_{segment_id}.{format}"
        chunk.export(chunk_path, format=format)
        return chunk_path
    except Exception as e:
        logging.error(f"Failed to save audio chunk: {e}")
        return None

def parse_xml_and_extract_audio(xml_file, output_dir):
    """Parses XML file, extracts segments, saves them as audio files, and logs in a TSV."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        audio_file = xml_file.replace('.xml', '.wav')
        df = pd.DataFrame(columns=['path', 'sentence'])
        segment_id = 1

        for segment in root.findall('S'):
            text = segment.find('FORM').text
            audio_tag = segment.find('AUDIO')
            start_time = float(audio_tag.get('start'))
            end_time = float(audio_tag.get('end'))
            logging.info(f"Processing segment {segment_id}: {start_time} to {end_time}, Text: {text}")
            format = 'wav'
            chunk_path = save_chunk(audio_file, start_time, end_time, output_dir, segment_id,format)
            if chunk_path:
                df = df.append({'path': str(chunk_path), 'sentence': text}, ignore_index=True)
            segment_id += 1

        tsv_path = output_dir / 'annotations.tsv'
        df.to_csv(tsv_path, sep='\t', index=False)
        return tsv_path
    except Exception as e:
        logging.error(f"Error processing XML file: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Parse a Pangloss XML annotation file and create audio chunks.')
    parser.add_argument('xml_file', type=str, help='Path to the Pangloss XML annotation file')
    parser.add_argument('--output_dir', type=str, default='./chunks', help='Directory to save the audio chunks and annotations')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    annotations_path = parse_xml_and_extract_audio(args.xml_file, output_dir)
    if annotations_path:
        logging.info(f"Annotations saved to {annotations_path}")
    else:
        logging.error("Failed to save annotations.")

if __name__ == "__main__":
    main()
