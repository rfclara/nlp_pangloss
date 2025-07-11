# ----------- Global variables ----------- #
MISC_SYMBOLS = [' ̩', '~', '=', ':', 'F', '¨', '↑', '“', '”', '…', '«', '»',
                'D', 'a', 'ː', '#', '$', "‡", "˞"]
BAD_NA_SYMBOLS = ['D', 'F', '~', '…', '=', '↑', ':']
PUNC_SYMBOLS = [',', '!', '.', ';', '?', "'", '"', '*', ':', '«', '»', '“', '”', "ʔ", "+"]
UNI_PHNS = {'q', 'p', 'ɭ', 'ɳ', 'h', 'ʐ', 'n', 'o', 'ɤ', 'ʝ', 'ɛ', 'g',
            'i', 'u', 'b', 'ɔ', 'ɯ', 'v', 'ɑ', 'l', 'ɖ', 'ɻ', 'ĩ', 'm',
            't', 'w', 'õ', 'ẽ', 'd', 'ɣ', 'ɕ', 'c', 'ʁ', 'ʑ', 'ʈ', 'ɲ', 'ɬ',
            's', 'ŋ', 'ə', 'e', 'æ', 'f', 'j', 'k', 'z', 'ʂ'}
BI_PHNS = {'dʑ', 'ẽ', 'ɖʐ', 'w̃', 'æ̃', 'qʰ', 'i͂', 'tɕ', 'v̩', 'o̥', 'ts',
           'ɻ̩', 'ã', 'ə̃', 'ṽ', 'pʰ', 'tʰ', 'ɤ̃', 'ʈʰ', 'ʈʂ', 'ɑ̃', 'ɻ̃', 'kʰ',
           'ĩ', 'õ', 'dz', "ɻ̍", "wæ", "wɑ", "wɤ", "jæ", "jɤ", "jo", "ʋ̩"}
FILLERS = {"əəə…", "mmm…"}
TRI_PHNS = {"tɕʰ", "ʈʂʰ", "tsʰ", "ṽ̩", "ṽ̩", "ɻ̩̃", "wæ̃", "w̃æ", "ʋ̩̃", "ɻ̩̃"}
UNI_TONES = {"˩", "˥", "˧"}
BI_TONES = {"˧˥", "˩˥", "˩˧", "˧˩"}
TONES = UNI_TONES.union(BI_TONES)


def preprocessing_text(sentence):
    if sentence[:4] in ["əəə…", "mmm…"]:
        return sentence[:4], sentence[4:]
    if sentence.startswith("ə…"):
        return "əəə…", sentence[2:]
    if sentence.startswith("m…"):
        return "mmm…", sentence[2:]
    if sentence.startswith("mm…"):
        return "mmm…", sentence[3:]
    if sentence[:3] == "wæ̃":
        return "w̃æ", sentence[3:]
    if sentence[:3] == "ṽ̩":
        return "ṽ̩", sentence[3:]
    if sentence[:3] in TRI_PHNS:
        return sentence[:3], sentence[3:]
    if sentence[:2] in BI_PHNS:
        return sentence[:2], sentence[2:]
    if sentence[:2] == "˧̩":
        return "˧", sentence[2:]
    if sentence[:2] == "˧̍":
        return "˧", sentence[2:]
    if sentence[0] in UNI_PHNS:
        return sentence[0], sentence[1:]
    if sentence[:2] in BI_TONES:
        return sentence[:2], sentence[2:]
    if sentence[0] in UNI_TONES:
        return sentence[0], sentence[1:]
    if sentence[0] in MISC_SYMBOLS:
        # We assume these symbols cannot be captured.
        return None, sentence[1:]
    if sentence[0] in BAD_NA_SYMBOLS:
        return None, sentence[1:]
    if sentence[0] in PUNC_SYMBOLS:
        return None, sentence[1:]
    if sentence[0] in ["-", "ʰ", "/"]:
        return None, sentence[1:]
    if sentence[0] in {"<", ">"}:
        # We keep everything literal, thus including what is in <>
        # brackets; so we just remove these tokens"
        return None, sentence[1:]
    if sentence[0] == "[":
        if sentence.find("]") == len(sentence) - 1:
            return None, ""
        else:
            return None, sentence[sentence.find("]") + 1:]
    if sentence[0] in {" ", "\t", "\n"}:
        # Return a space char so that it can be identified in word segmentation
        # processing.
        return " ", sentence[1:]
    if sentence[0] == "|" or sentence[0] == "◊":
        return "|", sentence[1:]
    if sentence[0] in "()":
        return None, sentence[1:]


def filter_for_phonemes(sentence):
    """ Returns a sequence of phonemes and pipes (word delimiters). Tones,
        syllable boundaries, whitespace are all removed."""
    filtered_sentence = []
    phonemes = []
    while sentence != "":
        phoneme, sentence = preprocessing_text(sentence)
        filtered_sentence.append(phoneme)
        phonemes.append(phoneme)
    filtered_sentence = [item for item in filtered_sentence if item != None]
    return "".join(filtered_sentence), phonemes


# IF WANT TO CUT BY PHONEMES
def final_text_phonemes(batch):
    if "BEGAIEMENT" in batch['sentence']:
        batch['sentence'] = batch['sentence'].replace('BEGAIEMENT', "")
    s, p = filter_for_phonemes(batch['sentence'])
    s = s.replace(' ', '|')
    s = s.replace('||', '|')
    s = s.replace('|||', '|')
    batch['sentence'] = s
    p = list(filter(None, p))
    batch['phonemes'] = p
    return batch


# IF WANT TO KEEP WORDS
def final_text_words(batch):
    if "BEGAIEMENT" in batch['sentence']:
        batch['sentence'] = batch['sentence'].replace('BEGAIEMENT', "")
    s, p = filter_for_phonemes(batch['sentence'])
    s = s.replace('|', ' ')
    s = ' '.join(s.split())
    s = s.replace(' ', '|')
    batch['sentence'] = s
    p = list(filter(None, p))
    batch['phonemes'] = p
    return batch
