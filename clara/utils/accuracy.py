import torch
import torch.nn as nn
from typing_extensions import Literal

def accuracy(output, target, topk=(1,)):
    """Compute the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class BidirectionalAccuracy(nn.Module):
    """
    Computes accuracy metrics for models that output features for both forward
    and reversed input sequences.
    """
    def __init__(self, top_k=(1,), cache_labels: bool = False):
        super().__init__()
        self.cache_labels = cache_labels
        self.top_k = top_k

        # Cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, text_features_forward, text_features_reverse, audio_features_forward, audio_features_reverse, text_temperature=1.0, audio_temperature=1.0):
        device = audio_features_forward.device

        # Combine features from both directions
        text_features = (text_features_forward + text_features_reverse) / 2
        audio_features = (audio_features_forward + audio_features_reverse) / 2

        logits_per_audio = audio_temperature * audio_features @ text_features.T
        logits_per_text = text_temperature * text_features @ audio_features.T

        # Calculate ground truth and cache if enabled
        num_logits = logits_per_audio.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        acc = accuracy(logits_per_audio, labels, topk=self.top_k)
        return acc

# Example usage
# Assume dimensions [batch, features] for simplicity
#text_features_forward = torch.randn(10, 512)
#text_features_reverse = torch.randn(10, 512)
#audio_features_forward = torch.randn(10, 512)
#audio_features_reverse = torch.randn(10, 512)

#acc_module = BidirectionalAccuracy(top_k=(1,))
#acc = acc_module(text_features_forward, text_features_reverse, audio_features_forward, audio_features_reverse)
#print(acc)
