import numpy as np
import scipy
import librosa
import soundfile
import os, re

#########################################################################
# If you use the chaos model, please cite the following paper:
# Khante, P., Thomaz, E., & de Barbaro, K. Detecting Auditory Household Chaos 
# in Child-worn Real-world Audio. (Revise & resubmit). Frontiers in Digital Health: 
# Special Issue on Artificial Intelligence for Child Health and Wellbeing (2023).
#########################################################################

#########################################################################
# These functions have been inspired/taken from Eduardo Fonseca 2018, v1.0
# https://github.com/edufonseca/icassp19
#########################################################################

def load_audio_file(file_path, input_fixed_length=0, params_extract=None):
    """

    :param file_path:
    :param input_fixed_length:
    :param params_extract:
    :return:
    """
    data, source_fs = soundfile.read(file=file_path)
    data = data.T

    # Resample if the source_fs is different from expected
    if params_extract.get('fs') != source_fs:
        data = librosa.core.resample(data, source_fs, params_extract.get('fs'))
        print('Resampling to %d: %s' % (params_extract.get('fs'), file_path))

    if len(data) > 0:
        data = get_normalized_audio(data)
    else:
        # 3 files are corrupted in the test set. They belong to the padding group (not used for evaluation)
       data = np.ones((input_fixed_length, 1))
       print('File corrupted. Could not open: %s' % file_path)

    # careful with the shape
    data = np.reshape(data, [-1, 1])
    return data
    
def modify_file_variable_length(data=None, input_fixed_length=0, params_extract=None):
    """

    :param data:
    :param input_fixed_length:
    :param params_extract:
    :return:
    """

    if params_extract.get('load_mode') == 'varup':
        # deal with short sounds
        if len(data) < input_fixed_length:
            # if file shorter than input_length, replicate the sound to reach the input_fixed_length
            nb_replicas = int(np.ceil(input_fixed_length / len(data)))
            # replicate according to column
            data_rep = np.tile(data, (nb_replicas, 1))
            data = data_rep[:input_fixed_length]

    return data

    
def get_normalized_audio(y, head_room=0.005):

    mean_value = np.mean(y)
    y -= mean_value

    max_value = max(abs(y)) + head_room
    return y / max_value
    
def get_mel_spectrogram(audio, params_extract=None):
    """

    :param audio:
    :param params_extract:
    :return:
    """

    # make sure rows are channels and columns the samples
    audio = audio.reshape([1, -1])
    window = scipy.signal.hamming(params_extract.get('win_length_samples'), sym=False)

    mel_basis = librosa.filters.mel(sr=params_extract.get('fs'),
                                    n_fft=params_extract.get('n_fft'),
                                    n_mels=params_extract.get('n_mels'),
                                    fmin=params_extract.get('fmin'),
                                    fmax=params_extract.get('fmax'),
                                    htk=False,
                                    norm=None)

    # init mel_spectrogram expressed as features: row x col = frames x mel_bands = 0 x mel_bands (to concatenate axis=0)
    feature_matrix = np.empty((0, params_extract.get('n_mels')))
    for channel in range(0, audio.shape[0]):
        spectrogram = get_spectrogram(
            y=audio[channel, :],
            n_fft=params_extract.get('n_fft'),
            win_length_samples=params_extract.get('win_length_samples'),
            hop_length_samples=params_extract.get('hop_length_samples'),
            spectrogram_type=params_extract.get('spectrogram_type') if 'spectrogram_type' in params_extract else 'magnitude',
            center=True,
            window=window,
            params_extract=params_extract
        )

        mel_spectrogram = np.dot(mel_basis, spectrogram)
        mel_spectrogram = mel_spectrogram.T

        if params_extract.get('log'):
            mel_spectrogram = np.log10(mel_spectrogram + params_extract.get('eps'))

        feature_matrix = np.append(feature_matrix, mel_spectrogram, axis=0)

    return feature_matrix
    
def get_spectrogram(y,
                    n_fft=1024,
                    win_length_samples=0.04,
                    hop_length_samples=0.02,
                    window=scipy.signal.hamming(1024, sym=False),
                    center=True,
                    spectrogram_type='magnitude',
                    params_extract=None):

    if spectrogram_type == 'power':
        return np.abs(librosa.stft(y + params_extract.get('eps'),
                                   n_fft=n_fft,
                                   win_length=win_length_samples,
                                   hop_length=hop_length_samples,
                                   center=center,
                                   window=window)) ** 2
                                                            
    
def save_tensor(var, out_path=None, suffix='_mel'):
    """
    Saves a numpy array as a binary file
    -review the shape saving when it is a label
    """
    assert os.path.isdir(os.path.dirname(out_path)), "path to save tensor does not exist"
    var.tofile(out_path.replace('.data', suffix + '.data'))
    save_shape(out_path.replace('.data', suffix + '.shape'), var.shape)
    
    
def save_shape(shape_file, shape):
    """
    Saves the shape of a numpy array
    """
    with open(shape_file, 'w') as fout:
        fout.write(u'#'+'\t'.join(str(e) for e in shape)+'\n')
        

def get_shape(shape_file):
    """
    Reads a .shape file
    """
    with open(shape_file, 'rb') as f:
        line=f.readline().decode('ascii')
        if line.startswith('#'):
            shape=tuple(map(int, re.findall(r'(\d+)', line)))
            return shape
        else:
            raise IOError('Failed to find shape in file')
            
            
def load_tensor(in_path, suffix=''):
    """
    Loads a binary .data file
    """
    assert os.path.isdir(os.path.dirname(in_path)), "path to load tensor does not exist"
    f_in = np.fromfile(in_path.replace('.data', suffix + '.data'))
    shape = get_shape(in_path.replace('.data', suffix + '.shape'))
    f_in = f_in.reshape(shape)
    return f_in



