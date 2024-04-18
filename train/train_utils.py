import torch
from conformer.model import SpeechModel
from utils.utils import get_configs
import jiwer

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
    """ setup train data_manipulation loader for 1 epoch """
    for step, (log_mel, transcript, inputs_sizes, target_sizes) in enumerate(train_loader):
        # batch_log_mel, batch transcript
        # get input from batch

        if torch.cuda.is_available():
            log_mel, transcript = log_mel.cuda(), transcript.cuda()
        
        optimizer.zero_grad() # zero grad after batch trained
        print(f"Input: {log_mel.shape}")
        prediction = model(log_mel) # get model prediction per batch
        print(f"Preidction: {prediction.shape} In size: {inputs_sizes}"
              f"Transcript: {transcript.shape}")
              
        # prediction, transcripts, input_size, transcript_size
        loss = loss_fn(prediction, transcript, inputs_sizes, target_sizes)
        
        # backward process
        # loss.backward()
        
        # adjust weights
        # optimizer.step()
        
        # batch_loss processing
        # batch_losses.append(loss.item())

    # append batch_loss
    # epoch_losses.append(sum(batch_losses/len(batch_losses)))
    
    # return poss per epoch
    # return epoch_losses

def eval_one_epoch(val_loader, model, loss_fn):
    """ setup validation data_manipulation loader for 1 epoch
    :param val_loader
    :param model
    :param loss_fn - ctc loss
    """
    batch_losses = []
    epoch_losses = []
    with torch.no_grad():
        for step, (log_mel, transcripts) in enumerate(val_loader):
            # get inputs from batch
            if torch.cuda.is_available():
                log_mel, transcripts = log_mel.cuda(), transcripts.cuda()

            prediction = model(log_mel)
            
            # ctc loss
            loss = loss_fn(prediction, transcripts) # prediction, transcripts, input_size, transcript_size
            
            # batch loss
            batch_losses.append(loss.item())
        
        # epoch loss
    epoch_losses.append(sum(batch_losses)/len(batch_losses))

    return epoch_losses