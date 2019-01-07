import os
from scipy.io.wavfile import read
import subprocess as sp
import numpy as np
import argparse
import random

# Required directories
dir_text = os.path.expanduser('~/Downloads/voxceleb_data/voxceleb1_txt')
dir_audio = os.path.expanduser('~/Downloads/voxceleb_data/voxceleb1_audio')
root_path = os.path.expanduser('~/Downloads/voxceleb')
development_path = os.path.expanduser('~/Downloads/voxceleb_data/voxceleb1_development.txt')
enrollment_path = os.path.expanduser('~/Downloads/voxceleb_data/voxceleb1_enrollment.txt')
evaluation_path = os.path.expanduser('~/Downloads/voxceleb_data/voxceleb1_evaluation.txt')


parser = argparse.ArgumentParser(description='Extracting the utterances of POIs from OXFORD VOXCELEB dataset')
parser.add_argument('--root_path', default=root_path, help='The directory which the whole data files are available')
parser.add_argument('--dir_audio', default=dir_audio, help='The directory which the output audio files will be stored')
parser.add_argument('--dir_text', default=dir_text, help='The directory which the annotations exist')
parser.add_argument('--development_path', default=development_path, help='The list of files for developments phase')
parser.add_argument('--enrollment_path', default=enrollment_path, help='The list of files for enrollment phase')
parser.add_argument('--evaluation_path', default=evaluation_path, help='The list of files for evaluation phase')
args = parser.parse_args()


def extract():
    file_development = open(args.development_path, 'w')
    file_enrollment = open(args.enrollment_path, 'w')
    file_evaluation = open(args.evaluation_path, 'w')

    # This loop just check for file compatibility in which all files should be of type .wav!
    # It's not a requirement but it's just for convenience.
    for root, dirs, files in os.walk(args.dir_audio, topdown=False):

        # walk through the directory
        for name in files:
            # Check all files to be of format .txt
            # Get each file path
            file_path = os.path.join(root, name)
            assert os.path.splitext(file_path)[1] == '.wav', "file extension is not .txt: %s" % file_path

    # Get the list of subfolders (IDs) & check for compatibility
    list_text_IDs = [os.path.basename(x[0]) for x in os.walk(args.dir_text)]
    list_text_IDs = list_text_IDs[1:]
    list_audio_IDs = [os.path.basename(x[0]) for x in os.walk(args.dir_audio)]
    list_audio_IDs = list_audio_IDs[1:]
    assert len(list(set(list_audio_IDs) - set(
        list_text_IDs))) == 0, "Please check for extra redundant created folders (possibly empty folders)"

    # For development phase we follow the same procedure used in the paper of "VoxCeleb: a large-scale speaker identification dataset"
    # All POIs that their name starts with 'E' will be excluded from development
    list_IDs_development = []
    list_IDs_enrollmentandevaluation = []
    for item in list_audio_IDs:
        if item.startswith('E'):
            list_IDs_enrollmentandevaluation.append(item)
        else:
            list_IDs_development.append(item)

    print("Number of POIs for enrollment & evaluation: %d \nNumber of POIs for development: %d" % (
    len(list_IDs_enrollmentandevaluation), len(list_IDs_development)))


    #########################
    #### Pair Generation ####
    #########################
    for i,ID in enumerate(list_IDs_development):

        ###########################
        #### Development Phase ####
        ###########################

        # Get the full path to the folder of POI
        POI_files_path = os.path.join(args.dir_audio, ID)

        # Extract all the files and choose a portion of them at random.
        files = os.listdir(POI_files_path)
        files_development = random.sample(files, int(len(files) / 4.0))

        for file in files_development:
            # Create the text and the associated label for generating the text file for the genuine training pair.
            string_of_file = str(i) + ' ' + os.path.join(ID, file)

            # Write to file
            file_development.write(string_of_file + '\n')
            print("Writing genuine pair : %s" % string_of_file)


    ###################################
    #### Genuine impostor creation ####
    ###################################

    for i, ID in enumerate(list_IDs_enrollmentandevaluation):

        # Get the full path to the folder of POI
        POI_files_path = os.path.join(args.dir_audio, ID)

        # Extract all the files.
        files = os.listdir(POI_files_path)

        # Create enrollment and evaluation.
        files_enrollment = random.sample(files, int(len(files)/2.0))
        files_evaluation = [x for x in files if x not in files_enrollment]

        for file in files_enrollment:
            # Create the text and the associated label.
            string_of_file = str(i) + ' ' + os.path.join(ID, file)

            # Write to file
            file_enrollment.write(string_of_file + '\n')
            print("Writing enrollment files : %s" % string_of_file)


        for file in files_evaluation:
            # Create the text and the associated label.
            string_of_file = str(i) + ' ' + os.path.join(ID, file)

            # Write to file
            file_evaluation.write(string_of_file + '\n')
            print("Writing evaluation files : %s" % string_of_file)


if __name__ == '__main__':
    extract()
