import json


class TextTransform:
  # transform text to character
  def __init__(self, char_map_str) -> None:
    self.char_map_str = char_map_str
    self.blank_id = 0

  def get_char_transcript(self, text):
    int_sequence = []
    for item in text:
      
    
if __name__ == "__main__":
  text_transforms = TextTransform()