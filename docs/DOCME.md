# Documents
1. Convolution Subsampling: [link](https://www.tutorialexample.com/understand-convolution-subsampling-module-in-conformer-deep-learning-tutorial/)
2. Conv Subsampling: [link](https://blog.csdn.net/ldy007714/article/details/127086170?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171118103416800184166564%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171118103416800184166564&biz_id=0&spm=1018.2226.3001.4187)
3. Separable Convolution: [link](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)
4. Separable Convolution: [link](https://blog.csdn.net/yyp1998/article/details/121048613?spm=1001.2101.3001.4242.1&utm_relevant_index=3)
5. DepthWise: [link](https://www.youtube.com/watch?v=ftc7rj7kzQ0)
6. DepthWise-tt: [link](DepthWise: [link](https://www.youtube.com/watch?v=ftc7rj7kzQ0))
7. Add noises: [link](https://www.linkedin.com/pulse/signal-to-noise-ratio-snr-explained-leonid-ayzenshtat/)
8. Downsample ResNet: [link](https://stackoverflow.com/questions/55688645/how-downsample-work-in-resnet-in-pytorch-code)
9. Downsample ResetNet(tt): [link](https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/)
10. src: [link](https://gitlab.com/nguyentri.alan/conformer)
11. LSTM dimension: [link](https://stackoverflow.com/questions/61632584/understanding-input-shape-to-pytorch-lstm)
12. LSTM output: [link](https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm)
13. CTC Loss: [link](https://stackoverflow.com/questions/62251289/how-to-correctly-use-ctc-loss-with-gru-in-pytorch)
14. Approaches to S2T: [link](https://theaisummer.com/speech-recognition/)
15. Special: [link](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#conformer-ctc)
16. Special: [link](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/api.html#nemo.collections.asr.models.EncDecCTCModelBPE)
17. Intuition: [link](https://www.youtube.com/watch?v=co1ny5ztYCI)
18. Fine-tuning ASR: [link](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html#asr-tutorial-notebooks)
19. Discussion LibriSpeech vocab: [link](https://discourse.mozilla.org/t/building-lm-noticed-vocab-txt-and-librispeech-lm-norm-txt-have-a-lot-of-low-quality-words/33261/4)
20. ASR Decoder Inference: [link](https://pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html#beam-search-decoder)
21. Process asr text: [link](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py)
22. Prepare DS: [link](https://wenet.org.cn/wenet/tutorial_librispeech.html)
23. Longformer Attention: [link](https://ahelhady.medium.com/understanding-longformers-sliding-window-attention-mechanism-f5d61048a907)
### Guides
1. build guide: [link](https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/)
2. kernel size for audio: [link](https://stats.stackexchange.com/questions/441847/conv2d-kernel-size-for-audio-related-tasks)
3. Nemo ASR different language: [link](https://developer.nvidia.com/blog/jump-start-training-for-speech-recognition-models-with-nemo/)

## Understanding
1. Convolution Subsampling: used for sampling the input data, with kernel 3x3 abd stride 2
2. DNN can ignore input topology and resize into column vector, but with the audio feature (mel-spectroram or log mel) these contain low level features that needed to learn so that CNN is suitable for this solution.
3. LSTM out (out, hn, cn) while hn, cn is used for feed into another NN, output here is prediction.


# Development
1. ZipFormer: [link](https://arxiv.org/pdf/2310.11230.pdf)
2. ZipFormer paper explained: [link](https://www.youtube.com/watch?v=jvtTs9q1l8w)