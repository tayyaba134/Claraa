
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

from audiocap import *
from audioset import *
from base_tdm import *
from common_voice import *
from crema_d import *
from emns import *
from emov_db import *
from esc_50 import *
from fsd50k import *
from mswc import *
from ravdess import *
from td_datamodule import *
from td_tensored import *
from us8k import *
from vox_celeb import*
from wds_datamodule import *
from utils import get_lists
from text.tokeniser import BidirectionalTokeniser

##############
# Zeroshot fn
##############

def zeroshot_text(model, classnames, templates, language='en'):
	tokenizer = BidirectionalTokeniser()
	device = model.device
	with torch.no_grad():
		zeroshot_weights = []
		all_texts = []
		for classname in classnames:
			texts = [torch.tensor(tokenizer.encode(template.format(classname), language)) for template in templates]
			texts = pad_sequence(texts).T.contiguous().to(device)
			all_texts.append(texts)
			class_embeddings = model.encode_text(texts)
			class_embedding = model.model.text_transform(class_embedding)
			class_embedding = F.normalize(class_embeddings, dim=-1)
			class_embedding /= class_embedding.norm()
			zeroshot_weights.append(class_embedding)
		zeroshot_weights = torch.stack(zeroshot_weights).to(device)
	return zeroshot_weights, all_texts

##############
# Fewshot fn
##############

def fewshot_text(model, classnames, templates, language='en', num_examples=5):
    tokenizer = BidirectionalTokeniser()
    device = model.device
    with torch.no_grad():
        fewshot_weights = []
        all_texts = []
        for classname in classnames:
            # For fewshot, select a limited number of examples
            selected_templates = templates[:num_examples]
            texts = [torch.tensor(tokenizer.encode(template.format(classname), language)) for template in selected_templates]
            texts = pad_sequence(texts).T.contiguous().to(device)
            all_texts.append(texts)
            class_embeddings = model.encode_text(texts)
            class_embedding = model.model.text_transform(class_embeddings)
            class_embedding = F.normalize(class_embeddings, dim=-1)
            class_embedding /= class_embedding.norm(dim=0, keepdim=True)
            fewshot_weights.append(class_embedding)
        fewshot_weights = torch.stack(fewshot_weights).to(device)
    return fewshot_weights, all_texts


def get_dataset(task, dataset_name, root_cfg_path, root_data_path='s3://laion-west-audio/webdataset_tar/', batch_size=1, num_workers=0):
    ##########
	# Sounds
	##########

	if task == 'sounds':
		if dataset_name == 'esc50':
			templates = get_lists(os.path.join(root_cfg_path , "classification/sounds/esc-50/templates.txt"))
			with open(os.path.join(root_cfg_path , "classification/sounds/esc-50/minor_classes.json")) as f:
				classes = json.load(f)
			dataset = ESC50TDM(
						root_data_path=root_data_path,
						classes=classes,
						batch_size = batch_size,
						num_workers = num_workers,
					)
		elif dataset_name == 'audioset':
			templates = get_lists(os.path.join(root_cfg_path , "classification/sounds/audioset/templates.txt"))
			with open(os.path.join(root_cfg_path , "classification/sounds/audioset/classes.json")) as f:
				classes = json.load(f)
			dataset = AudioSetTDM(
						root_data_path=root_data_path,
						classes=classes,
						batch_size = batch_size,
						num_workers = num_workers,
					)
		elif dataset_name == 'us8k':
			templates = get_lists(os.path.join(root_cfg_path , "classification/sounds/us8k/templates.txt"))
			with open(os.path.join(root_cfg_path , "classification/sounds/us8k/classes.json")) as f:
				classes = json.load(f)
			dataset = Urbansound8KTDM(
						root_data_path=root_data_path,
						batch_size = batch_size,
						num_workers = num_workers,
					)
		elif dataset_name == 'fsd50k':
			templates = get_lists(os.path.join(root_cfg_path , "classification/sounds/fsd50k/templates.txt"))
			with open(os.path.join(root_cfg_path , "classification/sounds/fsd50k/classes.json")) as f:
				classes = json.load(f)
			dataset = FSD50KTDM(
						root_data_path=root_data_path,
						classes=classes,
						batch_size = batch_size,
						num_workers = num_workers,
					)
		else:
			raise ValueError(f"Dataset {dataset_name} not supported for task: {task}")

	##########
	# Gender
	##########
	elif task == 'gender':
		# waning that gender is no longer supported
		Warning("Due to changes gender is no longer supported.")
		sys.exit("task: gender is no longer supported. Exiting.")
		# dataset = VoxCelebTDM(
		# 			test_urls=['s3://laion-west/knoriy/VoxCeleb_gender/'],
		# 			batch_size = batch_size,
		# 			num_workers = num_workers,
		# 		)
		# templates = get_lists(os.path.join(root_cfg_path , "classification/gender/templates.txt"))
		# with open(os.path.join(root_cfg_path , "classification/gender/classes.json")) as f:
		# 	classes = json.load(f)

	##########
	# Emotion
	##########
	elif task == 'emotion':
		templates_path = os.path.join(root_cfg_path , f"classification/{task}/{dataset_name}/templates.txt")
		classes_path = os.path.join(root_cfg_path , f"classification/{task}/{dataset_name}/classes.json")

		if dataset_name == 'emns':
			dataset = EMNSTDM(
						root_data_path=root_data_path,
						batch_size = batch_size,
						num_workers = num_workers,
					)

			templates = get_lists(templates_path)
			with open(classes_path) as f:
				classes = json.load(f)
		elif dataset_name == 'emov-db':
			dataset = EmovDBTDM(
				root_data_path=root_data_path,
				batch_size = batch_size,
				num_workers = num_workers,
			)

			templates = get_lists(templates_path)
			with open(classes_path) as f:
				classes = json.load(f)
		elif dataset_name == 'crema-d':
			Warning("CREMA-D is not supported yet")
			dataset = CremaDTDM(
				root_data_path=root_data_path,
				batch_size = batch_size,
				num_workers = num_workers,
			)

			templates = get_lists(templates_path)
			with open(classes_path) as f:
				classes = json.load(f)
		elif dataset_name == 'ravdess':
			dataset = RavdessTDM(
						root_data_path=root_data_path,
						batch_size = batch_size,
						num_workers = num_workers,
					)

			templates = get_lists(templates_path)
			with open(classes_path) as f:
				classes = json.load(f)
		else:
			raise ValueError(f"Dataset {dataset_name} not supported for task: {task}")

	elif task == 'speech':
		if dataset_name == 'mswc':
			dataset = MSWCTDM(
					root_data_path=root_data_path,
					exclude_list=os.path.join(root_cfg_path , "exclude_list.txt"),
					batch_size = batch_size,
					num_workers = num_workers,
			)
			templates = None
			classes = None
		else:
			raise ValueError(f"Dataset {dataset_name} not supported for task: {task}")

	##########
	# age
	##########
	elif task == 'age':
		dataset = CommonVoiceTDM(
					test_urls=['s3://s-laion-audio/webdataset_tar/common_voice/test/'],
					batch_size = batch_size,
					num_workers = num_workers,
				)

		templates = get_lists(os.path.join(root_cfg_path , "classification/age/common_voice/templates.txt"))
		with open(os.path.join(root_cfg_path , "classification/age/common_voice/classes.json")) as f:
			classes = json.load(f)

	else:
		raise ValueError(f"Task {task} not supported")

	return dataset, templates, classes