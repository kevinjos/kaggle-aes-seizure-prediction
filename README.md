To Do
=====

Implement, test, analyze and reimplement everything
Research clinician observations of epileptic patients

Data Exploration
================

Sampleing Frequency:  
  * Dogs 400Hz  
  * Humans 5000Hz  

Noise reduction:

  * 60hz filter appears to be unnessary after watching a number of data files FFT plots (explore.py)
  * We might consider a high pass filter for ~0hz DC interference

data structure is a follows:  
data[0] == channel # x sample count array of raw data from   
data[1] == data_length_sec: the time duration of each data row  
data[2] == sampling_frequency: the number of data samples representing 1 second of EEG data  
data[3] == channels: a list of electrode names corresponding to the rows in the data field  
data[4] == sequence: the index of the data segment within the one hour series of clips. For example, preictal_segment_6.mat has a sequence number of 6, and represents the iEEG data from 50 to 60 minutes into the preictal data.  


Features
========

Zero crossings:

  * Count local maxima

Total length of wave  

Frequency Bin Intervals  
Delta FBI  
Information content:

  * Self similarity (Hausdorff)
  * Shannon entropy  

github.com/lambdaloop feature extractor?  
Phase syncronisity of electrodes (cohearence)  
pyeeg  

  * Without proper mapping of electrodes, electrode choice for phase sync is arbitraty

Autocorrelation  

Random projection  
Convolution  
Some genetic algorythm  
Neural net on 1D pixal:amplitude transformed into a hilbert system

Check small time windows  
  * EX: Seizure signaled in 10s window during 60min preictal




Clinical Background
-------------------

http://www.mc.vanderbilt.edu/documents/neurology/files/Lecture%205-%20EEG%20in%20focal%20epilepsies-handout.pdf

Electrode naming conventions
----------------------------

Electrode naming across datasets is not standard.

Patient_1: 15 electrodes named LD_1, LD_3,..., LD_8, RD_1, ..., RD_8

Patient_2: 24 electrodes named LGT_01, ..., LGT_24

Dog_1: 16 electrodes named NVC1202_32_002_Ecog_c001, ..., NVC1202_32_002_Ecog_c016

Dog_2: 16 electrodes named NVC0905_22_002_Ecog_c001, ..., NVC0905_22_002_Ecog_c016

Dog_3: 16 electrodes named NVC0906_22_007_Ecog_c001, ..., NVC0906_22_007_Ecog_c016

Dog_4: 16 electrodes named NVC1202_26_003_Ecog_c001, ..., NVC1202_26_003_Ecog_c016

Dog_5: 15 electrodes named NVC0905_22_004_Ecog_c001, ..., NVC0905_22_004_Ecog_c016; NVC0905_22_004_Ecog_c004 missing;

Electrode mapping
-----------------

For the dogs, the electrodes are positioned horizontally with two 4-contact strips on each side of the brain. Numbering goes from anterior to posterior, starting on the left superior strip. For the humans, there really is no standard system. Electordes are numbered on subdural grids, but grids can be inserted in any orientation. The best one can glean from the naming system is general anatomic coverage.  

Scoring
--------

Submission are in csv format  
sample_name,probability_preictal  
Scoring is done using the Area Under the Curve (AUC) of the Receiver operating characteristic (ROC)  
