from .clara import CLARA, PLCLARA, LinearProbeCLARA
from tools import *
from stft import STFT, TacotronSTFT,MelSpecPipeline
from audio_processing import *
from .base_tdm import BaseTDM
from .td_datamodule import MultilingualTDM
from .td_tensored import TensoredTDM
from .esc_50 import ESC50TDM
from .vox_celeb import VoxCelebTDM
from .emns import EMNSTDM
from .common_voice import CommonVoiceTDM
from .emov_db import EmovDBTDM
from .audioset import AudioSetTDM
from .audiocap import AudioCapTDM
from .crema_d import CremaDTDM
from .ravdess import RavdessTDM
from .us8k import Urbansound8KTDM
from .fsd50k import FSD50KTDM
from .mswc import MSWCTDM
from .whisper import WhisperAudioEncoder
from .simple_cnn import SimpleCNN, SimpleCNNLarge
from .cnn import Cnn1D12, Cnn1D10, Cnn2D10
from .resnet import resnet18
from .resnext import CifarResNeXt as ResNeXt
from .perceiver import PerceiverIOEncoder
from .simple_transformer import SimpleTransformer
""" from https://github.com/keithito/tacotron """