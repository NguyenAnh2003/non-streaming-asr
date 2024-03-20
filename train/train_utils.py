import torch
from torch.utils.data import DataLoader


def train_one_epoch(train_loader: DataLoader, model, optimizer, loss_fn):
    """ 
    :param train dataloader
    :param model
    :param optimizer
    :param loss_fn - ctc loss
    """
    epoch_losses = []
    batch_losses = []
    """ setup train data_manipulation loader for 1 epoch """
    for step, (log_mel, transcript) in enumerate(train_loader):
        # batch_log_mel, batch transcript
        # get input from batch

        if torch.cuda.is_available():
            log_mel, transcript = log_mel.cuda(), transcript.cuda()
        
        optimizer.zero_grad() # zero grad after batch trained
        
        prediction = model(log_mel) # get model prediction per batch
        
        loss = loss_fn(prediction, transcript) # prediction, transcripts, input_size, transcript_size
        
        # backward process
        loss.backward()
        
        # adjust weights
        optimizer.step()
        
        # batch_loss processing
        batch_losses.append(loss.item())

    # append batch_loss
    epoch_losses.append(sum(batch_losses/len(batch_losses)))
    
    # return poss per epoch
    return epoch_losses

def eval_one_epoch(val_loader: DataLoader, model, loss_fn):
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