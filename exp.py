import file_handler as fh
import feature_explorer as fe

def main():
  FH = fh.FileHandler()
  FH.file_in = 'Dog_1_preictal_segment_0011.mat'
  FH.set_data()
  data = FH.data[0]
  FE = fe.FeatureExtractor(data)
  FE.set_features(FH.file_in)
  return FE
