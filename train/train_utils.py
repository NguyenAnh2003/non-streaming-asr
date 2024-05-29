import torch
from parts.modules.conformer.model import SpeechModel
from utils.utils import get_configs
from core.metric.wer import compute_wer
from tqdm import tqdm

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def setup_speech_model():
    # get params
    params = get_configs("../configs/model_params.yaml")
    encoder_dim = params['encoder_dim']
    decoder_dim = params['decoder_dim']
    input_dims = params['input_dims']
    in_channels = params['channels']
    num_classes = params['num_classes']
    subsample_stride = params['subsampling_stride']
    conv_module_stride = params['normal_stride']
    num_layers = params['num_layers']
    dw_kernel_size = params['depthwise_kernel_size']
    sub_kernel_size = params['subsampling_kernel_size']
    pw_kernel_size = params['pointwise_kernel_size']
    expansion_factor = params['expansion_factor']

    # setup model
    model = SpeechModel(
        input_dims=input_dims,
        in_channels=in_channels,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        padding=1,
        sub_kernel_size=sub_kernel_size,
        dw_kernel_size=dw_kernel_size,
        pw_kernel_size=pw_kernel_size,
        num_layers=num_layers,
        subsample_stride=subsample_stride,
        normal_stride=conv_module_stride,
        expansion_factor=expansion_factor,
        num_classes = num_classes)

    return model

def train_one_epoch(train_loader, model, optimizer, loss_fn):
    """ 
    :param train dataloader
    :param model
    :param optimizer
    :param loss_fn - ctc loss
    """
    epoch_losses = []
    batch_losses = []
    total_errs = 0
    total_tokens = 0
    """ setup train data_manipulation loader for 1 epoch """
    for step, (log_mel, transcripts, inputs_sizes, target_sizes) in tqdm(enumerate(train_loader)):
        # batch_log_mel, batch transcript
        # get input from batch

        if torch.cuda.is_available():
            log_mel, transcripts = log_mel.cuda(), transcripts.cuda()
            inputs_sizes, target_sizes = inputs_sizes.cuda(), target_sizes.cuda()

        optimizer.zero_grad() # zero grad after batch trained
        log_probs, prediction, lengths = model(log_mel, inputs_sizes) # get model log_probs per batch
        # log_probs, transcripts, input_size, transcript_size
        loss = loss_fn(log_probs, transcripts, lengths, target_sizes)
        # needed to transpose log_probs
        batch_errs, batch_tokens = compute_wer(prediction.transpose(0, 1),
                                               inputs_sizes, transcripts,
                                               target_sizes)
        print(f"errors: {batch_errs} tokens: {batch_tokens} ")
        # accuracy
        total_errs += batch_errs
        total_tokens += batch_tokens
    
        # backward process
        loss.backward()
    
        # adjust weights
        optimizer.step()
    
        # batch_loss processing
        batch_losses.append(loss.item())
    
    # metric WER
    WER = total_errs / total_tokens

    epoch_losses.append(sum(batch_losses)/len(batch_losses))
    print(f"Train loss: {sum(epoch_losses)/len(epoch_losses)} "
          f"Train Acc: {WER}")
    return sum(epoch_losses)/len(epoch_losses), WER

def eval_one_epoch(val_loader, model, loss_fn):
    """ setup validation data_manipulation loader for 1 epoch
    :param val_loader
    :param model
    :param loss_fn - ctc loss
    """
    batch_losses = []
    epoch_losses = []
    total_errs = 0
    total_tokens = 0
    with torch.no_grad():
        for step, (log_mel, transcripts, inputs_sizes, target_sizes) in tqdm(enumerate(val_loader)):
            # get inputs from batch
            if torch.cuda.is_available():
                log_mel, transcripts = log_mel.cuda(), transcripts.cuda()
                inputs_sizes, target_sizes = inputs_sizes.cuda(), target_sizes.cuda()

            log_probs, prediction, lengths = model(log_mel, inputs_sizes)
            # ctc loss
            loss = loss_fn(log_probs, transcripts, lengths, target_sizes) # log_probs, transcripts, input_size, transcript_size
            batch_errs, batch_tokens = compute_wer(prediction.transpose(0, 1),
                                                   inputs_sizes,
                                                   transcripts,
                                                   target_sizes)

            # errors, tokens
            total_errs += batch_errs
            total_tokens += batch_tokens

            # batch loss
            batch_losses.append(loss.item())

    # metric WER
    WER = total_errs / total_tokens

    # epoch loss
    epoch_losses.append(sum(batch_losses)/len(batch_losses))
    print(f"Dev los: {sum(epoch_losses)/len(epoch_losses)} "
          f"WER Dev: {WER}")

    return sum(epoch_losses)/len(epoch_losses), WER