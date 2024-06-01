# Description: Saving and loading model checkpoints

import torch

# Saving the optimizer state
def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)


# Loading the optimizer state
def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


''' 

Practical Tips

    Saving Frequency: Save the model after every epoch or after a fixed number of iterations. 
    You can also save the model whenever it achieves a new best performance on the validation set.
    File Management: Use meaningful filenames that include the epoch number or validation performance to easily identify different checkpoints.
    Handling Large Models: 
    For large models, ensure you have sufficient disk space and manage the number of saved checkpoints to avoid clutter.


'''