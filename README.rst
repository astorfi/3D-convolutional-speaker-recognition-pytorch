.. image:: readme_images/follow-twitter.gif
   :height: 100px
   :width: 200 px
   :scale: 50 %
   :alt: alternate text
   :align: right
   :target: https://twitter.com/amirsinatorfi

=============================================================================================
3D Convolutional Neural Networks for Speaker Verification - `Official Project Page`_
=============================================================================================

.. image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    :target: https://github.com/astorfi/3D-convolutional-speaker-recognition/pulls
.. image:: https://badges.frapsoft.com/os/v2/open-source.svg?v=102
    :target: https://github.com/ellerbrock/open-source-badge/
.. image:: https://img.shields.io/twitter/follow/amirsinatorfi.svg?label=Follow&style=social
      :target: https://twitter.com/amirsinatorfi
    
==============================
Table of Contents
==============================
.. contents::
  :local:
  :depth: 4


This repository contains the Pytorch code release for our paper titled as *"Text-Independent
Speaker Verification Using 3D Convolutional Neural Networks"*. The link to the paper_ is
provided as well.


.. _Official Project Page: https://codeocean.com/2017/08/01/3d-convolutional-neural-networks-for-speaker-recognition/code

.. _paper: https://arxiv.org/abs/1705.09422
.. _Pytorch: https://pytorch.org

The code has been developed using Pytorch_. The input pipeline must be prepared by the users.
This code is aimed to provide the implementation for Speaker Verification (SR) by using 3D convolutional neural networks
following the SR protocol.

.. image:: readme_images/conv_gif.gif
    :target: https://github.com/astorfi/3D-convolutional-speaker-recognition/blob/master/_images/conv_gif.gif

------------
Citation
------------

If you used this code, please kindly consider citing the following paper:

.. code:: shell

    @article{torfi2017text,
      title={Text-independent speaker verification using 3d convolutional neural networks},
      author={Torfi, Amirsina and Nasrabadi, Nasser M and Dawson, Jeremy},
      journal={arXiv preprint arXiv:1705.09422},
      year={2017}
    }

--------------
General View
--------------

We leveraged 3D convolutional architecture for creating the speaker model in order to simultaneously
capturing the speech-related and temporal information from the speakers' utterances.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Speaker Verification Protocol(SVP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this work, a 3D Convolutional Neural Network (3D-CNN)
architecture has been utilized for text-independent speaker
verification in three phases.

     1. At the **development phase**, a CNN is trained
     to classify speakers at the utterance-level.

     2. In the **enrollment stage**, the trained network is utilized to directly create a
     speaker model for each speaker based on the extracted features.

     3. Finally, in the **evaluation phase**, the extracted features
     from the test utterance will be compared to the stored speaker
     model to verify the claimed identity.

The aforementioned three phases are usually considered as the SV protocol. One of the main
challenges is the creation of the speaker models. Previously-reported approaches create
speaker models based on averaging the extracted features from utterances of the speaker,
which is known as the d-vector system.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
How to leverage 3D Convolutional Neural Networks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In our paper, we propose the implementation of 3D-CNNs for direct speaker model creation
in which, for both development and enrollment phases, an identical number of
speaker utterances is fed to the network for representing the spoken utterances
and creation of the speaker model. This leads to simultaneously capturing the
speaker-related information and building a more robust system to cope with
within-speaker variation. We demonstrate that the proposed method significantly
outperforms the d-vector verification system.

--------------------
Dataset
--------------------

Unlike the `Original Implementaion <https://github.com/astorfi/3D-convolutional-speaker-recognition>`_, here we used the `VoxCeleb <http://www.robots.ox.ac.uk/~vgg/data/voxceleb/>`_ publicy available dataset. The dataset contains annotated audio files. For Speaker Verification, the parts of the audio associated with the subject of interest, however, must be extracted from the ``raw audio files``.

Three steps should be taken to prepare the data after downloading the data associated files.

  1. Extract the specific audio part that the subject of interest is speaking.[`extract_audio.py <https://github.com/astorfi/3D-convolutional-speaker-recognition-pytorch/blob/master/code/0-data_preparation/0-extract_audio/extract_audio.py>`_]
  2. Create train/test phase.[`create_phases.py <https://github.com/astorfi/3D-convolutional-speaker-recognition-pytorch/blob/master/code/0-data_preparation/2-create_phases/create_phases.py>`_]
  3. Voice Activity Detection to remove the silence. [`vad.py <https://github.com/astorfi/3D-convolutional-speaker-recognition-pytorch/blob/master/code/0-data_preparation/3-VAD/vad.py>`_]
  

Creating the dataset object, necessary preprocessing and feature extraction will be performed in the following data class:

.. code:: python

    class AudioDataset():
    """Audio dataset."""

        def __init__(self, files_path, audio_dir, transform=None):
            """
            Args:
                files_path (string): Path to the .txt file which the address of files are saved in it.
                root_dir (string): Directory with all the audio files.
                transform (callable, optional): Optional transform to be applied
                    on a sample.
            """

            # self.sound_files = [x.strip() for x in content]
            self.audio_dir = audio_dir
            self.transform = transform

            # Open the .txt file and create a list from each line.
            with open(files_path, 'r') as f:
                content = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
            list_files = []
            for x in content:
                sound_file_path = os.path.join(self.audio_dir, x.strip().split()[1])
                try:
                    with open(sound_file_path, 'rb') as f:
                        riff_size, _ = wav._read_riff_chunk(f)
                        file_size = os.path.getsize(sound_file_path)

                    # Assertion error.
                    assert riff_size == file_size and os.path.getsize(sound_file_path) > 1000, "Bad file!"

                    # Add to list if file is OK!
                    list_files.append(x.strip())
                except:
                    print('file %s is corrupted!' % sound_file_path)

            # Save the correct and healthy sound files to a list.
            self.sound_files = list_files

        def __len__(self):
            return len(self.sound_files)

        def __getitem__(self, idx):
            # Get the sound file path
            sound_file_path = os.path.join(self.audio_dir, self.sound_files[idx].split()[1]


--------------------
Code Implementation
--------------------

The input pipeline must be provided by the user. **Please refer to ``code/0-input/input_feature.py`` for having an idea about how the input pipeline works.**

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Input Pipeline for this work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: readme_images/Speech_GIF.gif
    :target: https://github.com/astorfi/3D-convolutional-speaker-recognition/blob/master/_images/Speech_GIF.gif

The MFCC features can be used as the data representation of the spoken utterances at the frame level. However, a
drawback is their non-local characteristics due to the last DCT 1 operation for generating MFCCs. This operation disturbs the locality property and is in contrast with the local characteristics of the convolutional operations. The employed approach in this work is to use the log-energies, which we
call MFECs. The extraction of MFECs is similar to MFCCs
by discarding the DCT operation. The temporal features are
overlapping 20ms windows with the stride of 10ms, which are
used for the generation of spectrum features. From a 0.8-
second sound sample, 80 temporal feature sets (each forms
a 40 MFEC features) can be obtained which form the input
speech feature map. Each input feature map has the dimen-
sionality of ζ × 80 × 40 which is formed from 80 input
frames and their corresponding spectral features, where ζ is
the number of utterances used in modeling the speaker during
the development and enrollment stages.



The **speech features** have been extracted using [SpeechPy]_ package.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implementation of 3D Convolutional Operation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following script has been used for our
implementation:

.. code:: python

        self.conv11 = nn.Conv3d(1, 16, (4, 9, 9), stride=(1, 2, 1))
        self.conv11_bn = nn.BatchNorm3d(16)
        self.conv11_activation = torch.nn.PReLU()
        self.conv12 = nn.Conv3d(16, 16, (4, 9, 9), stride=(1, 1, 1))
        self.conv12_bn = nn.BatchNorm3d(16)
        self.conv12_activation = torch.nn.PReLU()
        self.conv21 = nn.Conv3d(16, 32, (3, 7, 7), stride=(1, 1, 1))
        self.conv21_bn = nn.BatchNorm3d(32)
        self.conv21_activation = torch.nn.PReLU()
        self.conv22 = nn.Conv3d(32, 32, (3, 7, 7), stride=(1, 1, 1))
        self.conv22_bn = nn.BatchNorm3d(32)
        self.conv22_activation = torch.nn.PReLU()
        self.conv31 = nn.Conv3d(32, 64, (3, 5, 5), stride=(1, 1, 1))
        self.conv31_bn = nn.BatchNorm3d(64)
        self.conv31_activation = torch.nn.PReLU()
        self.conv32 = nn.Conv3d(64, 64, (3, 5, 5), stride=(1, 1, 1))
        self.conv32_bn = nn.BatchNorm3d(64)
        self.conv32_activation = torch.nn.PReLU()
        self.conv41 = nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 1, 1))
        self.conv41_bn = nn.BatchNorm3d(128)
        self.conv41_activation = torch.nn.PReLU()


As it can be seen, ``slim.conv2d`` has been used. However, simply by using 3D kernels as ``[k_x, k_y, k_z]``
and ``stride=[a, b, c]`` it can be turned into a 3D-conv operation. The base of the ``slim.conv2d`` is
``tf.contrib.layers.conv2d``. Please refer to official Documentation_ for further details.

.. _Documentation: https://www.tensorflow.org/api_docs/python/tf/contrib/layers


--------
License
--------

The license is as follows:

.. code:: shell


   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "{}"
      replaced with your own identifying information. (Don't include the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright {2017} {Amirsina Torfi}

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


Please refer to LICENSE_ file for further detail.

.. _LICENSE: https://github.com/astorfi/3D-convolutional-speaker-recognition/blob/master/LICENSE


-------------
Contribution
-------------

We are looking forward to your kind feedback. Please help us to improve the code and make
our work better. For contribution, please create the pull request and we will investigate it promptly.
Once again, we appreciate your feedback and code inspections.


.. rubric:: references

.. [SpeechPy] Amirsina Torfi. 2017. astorfi/speech_feature_extraction: SpeechPy. Zenodo. doi:10.5281/zenodo.810392.
