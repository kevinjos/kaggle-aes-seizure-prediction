To Do
=====

Implement, test, analyze and reimplement everything
Research clinician observations of epileptic patients

Data Exploration
================

Noise reduction:

  * 60hz filter appears to be unnessary after watching a number of data files FFT plots (explore.py)
  * We might consider a high pass filter for ~0hz DC interference

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

Patient_1: 16 electrodes named LD_1, ..., LD_8, RD_1, ..., RD_8

Patient_2: 24 electrodes named LGT_01, ..., LGT_24

Dog_1: 16 electrodes named NVC1202_32_002_Ecog_c001, ..., NVC1202_32_002_Ecog_c016

Dog_2: 16 electrodes named NVC0905_22_002_Ecog_c001, ..., NVC0905_22_002_Ecog_c016

Dog_3: 16 electrodes named NVC0906_22_007_Ecog_c001, ..., NVC0906_22_007_Ecog_c016

Dog_4: 16 electrodes named NVC1202_26_003_Ecog_c001, ..., NVC1202_26_003_Ecog_c016

Dog_5: 16 electrodes named NVC0905_22_004_Ecog_c001, ..., NVC0905_22_004_Ecog_c016
