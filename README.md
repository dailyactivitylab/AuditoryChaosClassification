# AuditoryChaosClassification
This repository contains the code to run the auditory chaos model for continuous audio collected in real-world environments, as described in this [paper](). Given continuous audio recordings, the code segments the raw audio into 5s segments and predicts an auditory chaos level (0 (No Chaos), 1 (Low Chaos), 2 (Medium Chaos), 3 (High Chaos)) for each segment. If you use the model or code, please cite the following paper.

## Citation Information
Khante, P., Thomaz, E., & de Barbaro, K. Auditory Chaos Classification in Real-world Environments (Revise & Resubmit). Frontiers in Digital Health: Special Issue on Artificial Intelligence for Child Health and Wellbeing (2023).

## Models and Main Package Versions 
Trained CNN model can be found in this repository: Final_Chaos_Model.h5  
Download Scaler from Zenodo: [https://zenodo.org/record/8435643](https://zenodo.org/record/8435643). Ensure that you unzip the downloaded scaler file and put it in the same place as the above pretrained model. 

### Versions
python==3.9.7  
librosa==0.10.1  
numpy==1.19.5  
pandas==1.3.3  
pydub==0.25.1  
PyYAML==5.4.1  
scikit_learn==1.2.2  
scipy==1.11.3  
soundfile==0.12.1  
tensorflow_gpu==2.6.1  
tqdm==4.65.0  

# Code
There are two scripts that you need to run sequentially: *segment_audio.py* and *predict.py*.

*segment_audio.py* chunks continuous audio recordings into audio segments of 5s each (no overlap between chunks). It reads in a raw audio file and outputs chunked audio segments which are saved in a folder along with a csv file with the chunked audio segment names. Pass the raw audio filename to chunk when running the code as follows. 
```
./segment_audio.py Example_audio/Silence.wav
```
Chunked audio segments are saved in the *Audio_segment* and generated output csv is *Audio_segments.csv*.



*predict.py* gives predictions of auditory household chaos predictions for every audio chunk i.e. 5s audio segment. It reads in audio segments and the csv file with the filenames of the audio segments and outputs a csv file containing the predictions (Chaos level 0 to 3) for every segment. Run the code in the following way. 

```
./predict.py
```
Input is audio chunks in the *Audio_segments* folder and *Audio_segments.csv* and output is *Predictions.csv*.


## Other resources
HomeBank English deBarbaro Auditory Chaos Corpus [https://homebank.talkbank.org/access/Password/deBarbaroChaos.html](https://homebank.talkbank.org/access/Password/deBarbaroChaos.html)

It contains a subset of the Annotated dataset (from the citation) that was used to train and test the model as 8 out of 22 participants did not give us permission to share their data. 40 hours of data balanced across all four levels of auditory chaos were randomly sampled to train and test the model. The complete dataset totals 54.6h of labelled data and we make 39.4 hours publicly available on Homebank.

## Contact
Contact Priyanka Khante and Kaya de Barbaro should you have any question/suggestion or if you have any problems running the code at [priyanka.khante@utexas.edu](mailto:priyanka.khante@utexas.edu) and [kaya@austin.utexas.edu](mailto:kaya@austin.utexas.edu).


