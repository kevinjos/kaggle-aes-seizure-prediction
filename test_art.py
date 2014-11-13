import file_handler as FH
import feature_extractor as FE
fh = FH.FileHandler()
fh.file_in = 'Patient_2_test_segment_0003.mat'
fh.set_data()

fe = FE.FeatureExtractor(fh.data[0])
fe.set_filen(fh.file_in)
fe.apply_frequency()
fe.set_medianval()
fe.apply_first_order_diff()
fe.apply_artifact_removal()
