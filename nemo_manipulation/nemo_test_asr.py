import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import WER
from parts.utils.utils import get_configs


def test(MODEL_NAME: str, params, use_cer):
    # prepare model
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(MODEL_NAME)
    asr_model._wer = WER(
        decoding=asr_model.decoding,
        batch_dim_index=0,
        use_cer=use_cer,
        dist_sync_on_step=True,
        log_prediction=True,
    )

    print(f"Prepare testing model: {MODEL_NAME}")
    asr_model.setup_test_data(test_data_config=params["model"]["test_ds"])
    asr_model.cuda()
    asr_model.eval()

    # We will be computing Word Error Rate (WER) metric between our hypothesis and predictions.
    # WER is computed as numerator/denominator.
    # We'll gather all the test batches' numerators and denominators.
    wer_nums = []
    wer_denoms = []

    # Loop over all test batches.
    # Iterating over the model's `test_dataloader` will give us:
    # (audio_signal, audio_signal_length, transcript_tokens, transcript_length)
    # See the AudioToCharDataset for more details.
    for test_batch in asr_model.test_dataloader():
        test_batch = [x.cuda() for x in test_batch]
        targets = test_batch[2]
        targets_lengths = test_batch[3]
        log_probs, encoded_len, greedy_predictions = asr_model(
            input_signal=test_batch[0], input_signal_length=test_batch[1]
        )
        # Notice the model has a helper object to compute WER
        asr_model.wer.update(
            predictions=greedy_predictions,
            predictions_lengths=None,
            targets=targets,
            targets_lengths=targets_lengths,
        )
        _, wer_num, wer_denom = asr_model.wer.compute()
        asr_model.wer.reset()
        wer_nums.append(wer_num.detach().cpu().numpy())
        wer_denoms.append(wer_denom.detach().cpu().numpy())

        # Release tensors from GPU memory
        del (
            test_batch,
            log_probs,
            targets,
            targets_lengths,
            encoded_len,
            greedy_predictions,
        )

    # We need to sum all numerators and denominators first. Then divide.
    metric = "WER"
    if use_cer:
        metric = "CER"
    print(f"{metric} = {sum(wer_nums) / sum(wer_denoms)}")


def inference(array, MODEL_NAME):
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=MODEL_NAME)
    asr_model.cuda()
    result = asr_model.transcribe(array)
    print(result)


if __name__ == "__main__":
    SAMPLE_RATE = 16000
    path = "../data_manipulation/librispeech/augmented-train"
    params = get_configs("../configs/conformer_ctc_bpe.yaml")
    MODEL_LARGE = "nvidia/stt_en_conformer_ctc_large"
    FCONFORMER_LARGE = "nvidia/stt_en_fastconformer_ctc_large"
    MODEL_SMALL = "nvidia/stt_en_conformer_ctc_small"

    # dataloader
    params["model"]["train_ds"]["sample_rate"] = SAMPLE_RATE
    params["model"]["validation_ds"]["sample_rate"] = SAMPLE_RATE
    params["model"]["test_ds"]["sample_rate"] = SAMPLE_RATE
    params["model"]["train_ds"][
        "manifest_filepath"
    ] = "../data_manipulation/metadata/ls/manifests/train-clean-manifest.json"
    params["model"]["validation_ds"][
        "manifest_filepath"
    ] = "../data_manipulation/metadata/ls/manifests/dev-clean-manifest.json"
    params["model"]["test_ds"][
        "manifest_filepath"
    ] = "../data_manipulation/metadata/ls/manifests/test-aug-manifest.json"

    # main(MODEL_NAME=MODEL_LARGE, params=params)
    test(MODEL_LARGE, params, use_cer=True)
    # audio_array, _ = torchaudio.load("../data_manipulation/examples/kkk.flac")
    # audio_array = audio_array.squeeze(0)
    # inference(audio_array, MODEL_LARGE) em train thifthi tat vs code di
