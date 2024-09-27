# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import math
import os.path
import random
import re

import codecs

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'

def as_bytes(bytes_or_text, encoding='utf-8'):
  """Converts `bytearray`, `bytes`, or unicode python input types to `bytes`.

  Uses utf-8 encoding for text by default.

  Args:
    bytes_or_text: A `bytearray`, `bytes`, `str`, or `unicode` object.
    encoding: A string indicating the charset for encoding unicode.

  Returns:
    A `bytes` object.

  Raises:
    TypeError: If `bytes_or_text` is not a binary or unicode string.
  """
  # Validate encoding, a LookupError will be raised if invalid.
  encoding = codecs.lookup(encoding).name
  if isinstance(bytes_or_text, bytearray):
    return bytes(bytes_or_text)
  elif isinstance(bytes_or_text, str):
    return bytes_or_text.encode(encoding)
  elif isinstance(bytes_or_text, bytes):
    return bytes_or_text
  else:
    raise TypeError('Expected binary or unicode string, got %r' %
                    (bytes_or_text,))


def which_set(filename, validation_percentage, testing_percentage):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(as_bytes(hash_name)).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result

def prepare_data_index(walker_files, unknown_percentage,
                        wanted_words, validation_percentage,
                        testing_percentage, random_seed=59185):
  """Prepares a list of the samples organized by set and label.

  The training loop needs a list of all the available data, organized by
  which partition it should belong to, and with ground truth labels attached.
  This function analyzes the folders below the `data_dir`, figures out the
  right
  labels for each file based on the name of the subdirectory it belongs to,
  and uses a stable hash to assign it to a data set partition.

  Args:
    silence_percentage: How much of the resulting data should be background.
    unknown_percentage: How much should be audio outside the wanted classes.
    wanted_words: Labels of the classes we want to be able to recognize.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    Dictionary containing a list of file information for each set partition,
    and a lookup map for each class to determine its numeric index.

  Raises:
    Exception: If expected files are not found.
  """
  # Make sure the shuffling and picking of unknowns is deterministic.
  random.seed(random_seed)
  wanted_words_index = {}
  for index, wanted_word in enumerate(wanted_words):
    wanted_words_index[wanted_word] = index + 2
  data_index = {'validation': [], 'testing': [], 'training': []}
  unknown_index = {'validation': [], 'testing': [], 'training': []}
  all_words = {}
  # Look through all the subfolders to find audio samples
  for wav_path in walker_files:
    _, word = os.path.split(os.path.dirname(wav_path))
    word = word.lower()
    # Treat the '_background_noise_' folder as a special case, since we expect
    # it to contain long audio samples we mix in to improve training.
    if word == BACKGROUND_NOISE_DIR_NAME:
      continue
    all_words[word] = True
    set_index = which_set(wav_path, validation_percentage, testing_percentage)
    # If it's a known class, store its detail, otherwise add it to the list
    # we'll use to train the unknown label.
    if word in wanted_words_index:
      #data_index[set_index].append({'label': word, 'file': wav_path})
      data_index[set_index].append(wav_path)
    else:
      #unknown_index[set_index].append({'label': word, 'file': wav_path})
      unknown_index[set_index].append(wav_path)
  if not all_words:
    raise Exception('No .wavs found at ' + walker_files)
  for index, wanted_word in enumerate(wanted_words):
    if wanted_word not in all_words:
      raise Exception('Expected to find ' + wanted_word +
                      ' in labels but only found ' +
                      ', '.join(all_words.keys()))
    
  # # We need an arbitrary file to load as the input for the silence samples.
  # # It's multiplied by zero later, so the content doesn't matter.
  # silence_wav_path = data_index['training'][0]['file']
  for set_index in ['validation', 'testing', 'training']:
    set_size = len(data_index[set_index])
    
    # Pick some unknowns to add to each partition of the data set.
    random.shuffle(unknown_index[set_index])
    unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
    data_index[set_index].extend(unknown_index[set_index][:unknown_size])

  # Make sure the ordering is random.
  for set_index in ['validation', 'testing', 'training']:
    random.shuffle(data_index[set_index])

  return data_index

