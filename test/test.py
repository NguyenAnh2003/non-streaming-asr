import jiwer

label = "Hello dit me may"
prediction = "Hello 1 2 3"

wer = jiwer.wer(label, prediction)
print(wer)