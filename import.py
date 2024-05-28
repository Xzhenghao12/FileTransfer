## This file is to define a function to import edf file correctly

# environment
import numpy as np
import pandas as pd
import mne
class processing():
    # read edf & edf+ file to obtain eeg and hypnogram
    def read_edf():
        signal = mne.io.read_raw_edf()
        hypno = mne.io.read_raw_edf()
        return signal, hypno
    
    # get information from signal
    
