from typing import List

class CTCDecoding():
  # init function
  def __init__(self, label):
    self.label = label
    pass
  
  # decode tokens2str
  def decode_tokens2str(self, tokens: List[int]) -> str:
    hypothesis = ""
    return hypothesis
  
  # decode ids2tokens
  def decode_ids2tokens(self, tokens: List[int]) -> List[str]:
    token_list = [self.label]