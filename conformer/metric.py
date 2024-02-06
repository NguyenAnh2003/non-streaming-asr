import jiwer

def compute_metric():
    """ measure model performance by WER """
    error_rate = jiwer.wer()
    return error_rate