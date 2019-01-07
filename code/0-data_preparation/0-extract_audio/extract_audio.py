import os
from scipy.io.wavfile import read
import subprocess as sp
import numpy as np
import argparse

# Required directories
dir_text = os.path.expanduser('~/Downloads/voxceleb_data/voxceleb1_txt')
dir_audio = os.path.expanduser('~/Downloads/voxceleb_data/voxceleb1_audio')
root_path = os.path.expanduser('~/Downloads/voxceleb')

# path indicator
audio_file_path = os.path.join('%s', '%s.mp3')

parser = argparse.ArgumentParser(description='Extracting the utterances of POIs from OXFORD VOXCELEB dataset')
parser.add_argument('--root_path', default=root_path, help='The directory which the whole data files are available')
parser.add_argument('--dir_audio', default=dir_audio, help='The directory which the output audio files will be stored')
parser.add_argument('--dir_text', default=dir_text, help='The directory which the annotations exist')
args = parser.parse_args()


def extract():

    for root, dirs, files in os.walk(args.dir_text, topdown=False):


        ### Get the information ###
        # Example:
        #   norm_path = /home/sina/Downloads/voxceleb_data_test/voxceleb1_txt/A.J._Buckley
        #   ID = A.J._Buckley
        norm_path = os.path.normpath(root)
        ID = os.path.basename(norm_path)
        dir_output_path = os.path.join(args.dir_audio, ID)

        # Create output path if does not exist!
        if not os.path.exists(dir_output_path):
            os.makedirs(dir_output_path)

        # walk through the directory
        for name in files:

            # Check all files to be of format .txt
            # Get each file path
            file_path = os.path.join(root, name)
            assert os.path.splitext(file_path)[1] == '.txt', "file extension is not .txt: %s" % file_path

        for name in files:

            # Get each file path
            #   file_path = /home/sina/Downloads/voxceleb_data_test/voxceleb1_txt/A.J._Buckley/9mQ11vBs1wc.txt
            file_path = os.path.join(root, name)
            assert os.path.splitext(file_path) != '.txt', "file extension is not .txt: %s" % file_path

            #  Parsing each line of text
            for line in open(file_path):

                # We only want to deal with the lines which are indicator of sound existence in different files.
                if line.startswith(ID):

                    # Split each line based on space and turn it into a list.
                    line_split = line.split()

                    # Get the file name based on the naming convention.
                    # line_split[0] = A.J._Buckley/9mQ11vBs1wc_0000001
                    # file_name = 9mQ11vBs1wc
                    file_name = os.path.basename(line_split[0]).split('_')[0]
                    file_output_path = os.path.join(dir_output_path,os.path.basename(line_split[0])) + '.wav'

                    # Start of the speech by the POI.
                    start = float(line_split[1])

                    # End of the speech by the POI.
                    end = float(line_split[2])

                    # Duration of the utterance
                    duration = end - start
                    print("start=%f , end=%f , duration=%f"  % (start, end, duration))

                    # Refer to the full path of the main audio file which has POI.
                    full_file_path_ID = audio_file_path % (args.root_path, file_name)

                    #################################
                    ##### Read mp3 using ffmpeg #####
                    #################################

                    # Extract the part of the sound file which is associated to the spoken utterance of POI using FFmpeg.
                    command = ['ffmpeg',
                               '-i', full_file_path_ID,
                               '-ss', str(start),
                               '-t', str(duration),
                               '-ar', '16000',  # ouput will have 48000 Hz
                               '-ac', '1',  # stereo (set to '1' for mono)
                               file_output_path]

                    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10 ** 8)


                    # #################################
                    # ##### Read mp3 using ffmpeg #####
                    # #################################
                    # FFMPEG_BIN = "ffmpeg"
                    # command = [FFMPEG_BIN,
                    #            '-i', full_file_path_ID,
                    #            '-f', 's16le',
                    #            '-acodec', 'pcm_s16le',
                    #            '-ar', '16000',  # ouput will have 44100 Hz
                    #            '-ac', '1',  # stereo (set to '1' for mono)
                    #            '-']
                    # pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10 ** 8)
                    # raw_audio = pipe.stdout.read()
                    #
                    # # Turn the read file to numpy array
                    # audio_array = np.fromstring(raw_audio, dtype="int16")
                    # audio_array = audio_array.reshape((len(audio_array), 1))
                    # print("length=%d" % audio_array.shape[0])
                    # print("max=%d , min=%d" % (np.max(audio_array),np.min(audio_array)))
                    #
                    # # Extract the part of the sound file which is associated to the spoken utterance of POI.


if __name__ == '__main__':
    extract()
