import torch
import torchvision.transforms as T

def resize_sequences(sequences,target_size):
    """resize sequence
    Args:
        sequences (Tensor): input sequence with shape (n, t, c, h, w)
        target_size (tuple): the size of output sequence with shape (H, W)
    Returns:
        Tensor: Output sequences with shape (n, t, c, H, W)
    """
    seq_list=[]
    for sequence in sequences:
        img_list=[T.Resize(target_size,interpolation=T.InterpolationMode.BICUBIC)(lq_image) for lq_image in sequence]
        seq_list.append(torch.stack(img_list))
    
    return torch.stack(seq_list)