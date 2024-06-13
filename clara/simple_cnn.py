import torch
import torch.nn as nn
import torch.nn.functional as F

class BidirectionalSimpleCNN(nn.Module):
    '''
    Simple Conv1d model with batchnorm and dropout for bidirectional processing.
    '''
    def __init__(self, in_channel:int, out_channel:int) -> None:
        super().__init__()
        self.cnn1 = nn.Conv1d(in_channel, out_channel, 3, padding=1)
        self.cnn2 = nn.Conv1d(out_channel, out_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.bn2 = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        # Process original sequence
        x_orig = F.dropout(F.relu(self.bn1(self.cnn1(x))), 0.5)
        x_orig = F.dropout(F.relu(self.bn2(self.cnn2(x_orig))), 0.5)
        
        # Process reversed sequence
        x_rev = torch.flip(x, dims=[2])
        x_rev = F.dropout(F.relu(self.bn1(self.cnn1(x_rev))), 0.5)
        x_rev = F.dropout(F.relu(self.bn2(self.cnn2(x_rev))), 0.5)

        # Combine features from both directions
        x_combined = x_orig + x_rev
        
        return x_combined

class BidirectionalSimpleCNNLarge(nn.Module):
    '''
    Larger Simple Conv1d model with batchnorm and dropout for bidirectional processing.
    '''
    def __init__(self, in_channel:int, out_channel:int) -> None:
        super().__init__()
        self.cnn1 = nn.Conv1d(in_channel, 2048, 3, padding=1)
        self.cnn2 = nn.Conv1d(2048, 2048, 3, padding=1)
        self.cnn3 = nn.Conv1d(2048, 2048, 3, padding=1)
        self.cnn4 = nn.Conv1d(2048, 2048, 3, padding=1)
        self.cnn5 = nn.Conv1d(2048, out_channel, 3, padding=1)

        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.bn5 = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        # Process original sequence
        x_orig = F.relu(self.bn1(self.cnn1(x)))
        x_orig = F.relu(self.bn2(self.cnn2(x_orig)))
        x_orig = F.relu(self.bn3(self.cnn3(x_orig)))
        x_orig = F.relu(self.bn4(self.cnn4(x_orig)))
        x_orig = F.dropout(F.relu(self.bn5(self.cnn5(x_orig))), 0.5)

        # Process reversed sequence
        x_rev = torch.flip(x, dims=[2])
        x_rev = F.relu(self.bn1(self.cnn1(x_rev)))
        x_rev = F.relu(self.bn2(self.cnn2(x_rev)))
        x_rev = F.relu(self.bn3(self.cnn3(x_rev)))
        x_rev = F.relu(self.bn4(self.cnn4(x_rev)))
        x_rev = F.dropout(F.relu(self.bn5(self.cnn5(x_rev))), 0.5)

        # Combine features from both directions
        x_combined = x_orig + x_rev
        
        return x_combined

# Example Usage
#model = BidirectionalSimpleCNNLarge(in_channel=1, out_channel=10)
#input_tensor = torch.rand(1, 1, 100)  # Example input tensor
#output = model(input_tensor)
