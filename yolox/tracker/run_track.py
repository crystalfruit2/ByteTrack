import sys
import torch

# Monkey-patch torch.cuda.FloatTensor before importing ByteTrack
original_cuda_float = torch.cuda.FloatTensor
original_cuda_half = torch.cuda.HalfTensor

def patched_cuda_float(x):
    if torch.cuda.is_available():
        return original_cuda_float(x)
    else:
        return x.float()

def patched_cuda_half(x):
    if torch.cuda.is_available():
        return original_cuda_half(x)
    else:
        return x.half()

torch.cuda.FloatTensor = patched_cuda_float
torch.cuda.HalfTensor = patched_cuda_half

# Now run the actual track.py
sys.argv = ['track.py', '-f', 'exps/example/mot/yolox_x_mix_det.py', '-c', 'pretrained/bytetrack_x_mot17.pth.tar', '--fuse', '-b', '1']
exec(open('tools/track.py').read())