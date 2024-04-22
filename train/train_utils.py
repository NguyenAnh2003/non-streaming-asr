import torch
from conformer.model import SpeechModel
from utils.utils import get_configs
from conformer.metric import compute_wer

def setup_speech_model():
    # get params
    params = get_configs("../configs/model_params.yaml")
    encoder_dim = params['encoder_dim']
    decoder_dim = params['decoder_dim']
    in_channels = params['channels']
    input_dims = params['input_dims']
    num_classes = params['num_classes']
    subsample_stride = params['subsampling_stride']
    conv_module_stride = params['normal_stride']

    # setup model
    model = SpeechModel(
                    input_dims=input_dims,
                    in_channels=in_channels,
                    encoder_dim=encoder_dim,
                    decoder_dim=decoder_dim,
                    kernel_size=3, padding=1,
                    num_layers=16,
                    subsample_stride=subsample_stride,
                    normal_stride=conv_module_stride,
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
    for step, (log_mel, transcript, inputs_sizes, target_sizes) in enumerate(train_loader):
        # batch_log_mel, batch transcript
        # get input from batch

        if torch.cuda.is_available():
            log_mel, transcript = log_mel.cuda(), transcript.cuda()
        
        optimizer.zero_grad() # zero grad after batch trained
        print(f"Input: {log_mel.shape}")
        prediction = model(log_mel) # get model prediction per batch

        # prediction, transcripts, input_size, transcript_size
        loss = loss_fn(prediction, transcript, inputs_sizes, target_sizes)
        _, index_max = torch.max(prediction, dim=-1)
        # needed to transpose prediction
        batch_errs, batch_tokens = compute_wer(index_max.transpose(0, 1),
                                               inputs_sizes, transcript,
                                               target_sizes)
        print(f"Batch errors: {batch_errs} Batch tokens: {batch_tokens}")

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

    # append batch_loss
    epoch_losses.append(sum(batch_losses)/len(batch_losses))
    
    return epoch_losses, 1-WER

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
        for step, (log_mel, transcripts, inputs_sizes, target_sizes) in enumerate(val_loader):
            # get inputs from batch
            if torch.cuda.is_available():
                log_mel, transcripts = log_mel.cuda(), transcripts.cuda()

            prediction = model(log_mel)
            # get index max
            _, index_max = torch.max(prediction, dim=-1)

            # ctc loss
            loss = loss_fn(prediction, transcripts, inputs_sizes, target_sizes) # prediction, transcripts, input_size, transcript_size
            batch_errs, batch_tokens = compute_wer(index_max.transpose(0, 1),
                                                   inputs_sizes,
                                                   transcripts,
                                                   target_sizes)
            print(f"Eval batch errors: {batch_errs} batch tokens: {batch_tokens}")

            # errors, tokens
            total_errs += batch_errs
            total_tokens += batch_tokens

            # batch loss
            batch_losses.append(loss.item())

    # metric WER
    WER = total_errs / total_tokens

    # epoch loss
    epoch_losses.append(sum(batch_losses)/len(batch_losses))

    return epoch_losses, 1 - WER