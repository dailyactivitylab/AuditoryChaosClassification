#########################################################################
# If you use the chaos model, please cite the following paper:
# Khante, P., Thomaz, E., & de Barbaro, K. Detecting Auditory Household Chaos 
# in Child-worn Real-world Audio. (Revise & resubmit). Frontiers in Digital Health: 
# Special Issue on Artificial Intelligence for Child Health and Wellbeing (2023).
#########################################################################

# This file chunks participant audio recordings into audio segments of 5s each
# Outputs chunked audio segments (no overlap between segments) which are saved in a folder 
# 0.wav signifes Chunk 1 - starting at 0th sec, ending at 5th sec of the raw audio recording
# 1.wav signifies Chunk 2 - starting at 5th sec, ending at 10th sec and so on...
# Also outputs a csv with chunk filenames

# If there is not enough data in the recording to create a 5 second chunk,
# the chunk is shortened to only include the data available (i.e. last chunk can be less than 5 seconds)

from pydub import AudioSegment
from pydub.utils import make_chunks
import pandas as pd
import os
import shutil
import sys

# Read in the audio filename from args
audio_filename = sys.argv[1]

# Set the export folder for audio chunks
export_folder = "Audio_segments"

# Set length of chunks to 5 seconds
# pydub calculates in millisec
chunk_length_ms = 5000 

# Dataframe to store chunk names
df = pd.DataFrame(columns=['fname'])

# Load the participant audio
participant_audio = AudioSegment.from_file(audio_filename, "wav") 

#Make chunks of 5 secs
chunks = make_chunks(participant_audio, chunk_length_ms) 

# Check if export folder exists. If yes, delete it and create a new one
if os.path.isdir(export_folder):
    shutil.rmtree(export_folder)
os.makedirs(export_folder) 

#Export all of the individual chunks as wav files
for i, chunk in enumerate(chunks):
    chunk_name = "{0}.wav".format(i)
    df = pd.concat([df, pd.DataFrame.from_records([{'fname': chunk_name}])])
    chunk.export(export_folder + '/' +chunk_name, format="wav")
    
# Write out chunk names into a csv file
df.to_csv("Audio_segments.csv", sep=",", index=False)



