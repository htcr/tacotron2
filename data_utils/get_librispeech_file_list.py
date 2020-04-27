"""
Generates training/eval file lists for GST training, with speaker id.
Output is text files, each line is one sample, formatted as audio_path.flac|TEXT|speaker_id.
Speaker id is an integer ranged in [0, speaker_num - 1), mapped from the original id.
"""

import os
import random
import enum

# Configs
# The directory containing train-clean-100
librispeech_root = '/scratch/jianrenw/tts/LibriSpeech'
folders = ['train-clean-100', 'train-clean-360', 'train-other-500']
eval_ratio = 0.01
output_dir = '/scratch/jianrenw/tts/LibriSpeech'
output_name = 'librispeech_full_gst_with_sex'
class AuxiliaryEmbeddingType(enum.Enum):
    speaker_id = 1
    sex = 2
auxiliary_embedding_type = AuxiliaryEmbeddingType.sex


# Const strings
kTransFilePattern = '{}-{}.trans.txt'
kSoundFileFormat = '.flac'
kOutputFileFormat = '.txt'
kSpeakerMetadata = 'SPEAKERS.TXT'

def process_trans_file(trans_file_path):
    # returns list of (sound_file_full_path, text) tuple
    with open(trans_file_path, 'r') as f:
        lines = f.readlines()
    tuples = list()
    for l in lines:
        spt = l.split(' ', maxsplit=1)
        sound_file_name = spt[0]+kSoundFileFormat
        sound_file_full_path = os.path.join(os.path.dirname(trans_file_path), sound_file_name)
        text = spt[1].strip()
        tuples.append((sound_file_full_path, text))
    return tuples

speaker_to_sex = dict() # original speaker id string to speaker sex id, 0 for female, 1 for male
speaker_map = dict() # original speaker id string to speaker embedding id
output_lines = list()

# Parse speaker metadata
metadata_path = os.path.join(librispeech_root, kSpeakerMetadata)
with open(metadata_path, 'r') as metadata_file:
    lines = metadata_file.readlines()
    for line in lines:
        if line[0] == ';':
            continue
        line_spt = [item.strip() for item in line.split('|')]
        speaker_id_str = line_spt[0]
        sex_id = 0 if line_spt[1] == 'F' else 1
        speaker_to_sex[speaker_id_str] = sex_id

# Make example list
for folder in folders:
    folder_path = os.path.join(librispeech_root, folder)
    speakers = [p for p in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, p))]
    for speaker in speakers:
        if speaker not in speaker_map:
            speaker_map[speaker] = len(speaker_map)
        print('Speaker id: {}'.format(speaker_map[speaker]))
        speaker_root = os.path.join(folder_path, speaker)
        sections = [s for s in os.listdir(speaker_root) if os.path.isdir(os.path.join(speaker_root, s))]
        for section in sections:
            section_root = os.path.join(speaker_root, section)
            trans_file_name = kTransFilePattern.format(speaker, section)
            trans_file_path = os.path.join(section_root, trans_file_name)
            trans_list = process_trans_file(trans_file_path)
            for sound_file_path, text in trans_list:
                if auxiliary_embedding_type == AuxiliaryEmbeddingType.speaker_id:
                    output_lines.append((sound_file_path, text, speaker_map[speaker]))
                elif auxiliary_embedding_type == AuxiliaryEmbeddingType.sex:
                    output_lines.append((sound_file_path, text, speaker_to_sex[speaker]))

random.shuffle(output_lines)
eval_num = int(len(output_lines) * eval_ratio)
eval_lines = output_lines[:eval_num]
train_lines = output_lines[eval_num:]

def write_lines(lines, split_name):
    output_file_path = os.path.join(output_dir, output_name + '_' + split_name + kOutputFileFormat)
    with open(output_file_path, 'w') as f:
        for sound_file_path, text, speaker_id in lines:
            f.write('{}|{}|{}\n'.format(sound_file_path, text, speaker_id))

write_lines(train_lines, 'train')
write_lines(eval_lines, 'eval')
