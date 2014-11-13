from scipy.signal import wavelets
import file_handler as fh

'''
Exploring wavelets in EEG data
'''

FH = fh.FileHandler()
FH.set_data()
data = FH.data[0][0,:]

Fs = 400

wavelet = wavelets.ricker
widths = range(4,401)
