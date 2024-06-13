import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import tqdm
import torch
from torch.nn import functional as F

from torchmetrics import MetricCollection, Recall, Accuracy, Precision, AveragePrecision

from clara import PLCLARA
from utils import calculate_average
from evutil import get_dataset, fewshot_text

from pprint import pprint

def run(model, support_set_loader, query_set_loader, metric_fn:MetricCollection, task:str, support_set_size: int, query_set_size: int):
	device = model.device
	model.train()  # Switch to training mode for few-shot learning

	with torch.no_grad():
		# Train on the support set
		for batch in tqdm.tqdm(support_set_loader, desc='Support Set'):
			labels, mels, _, _ = batch
			labels = labels[task].to(device)
			mels = mels.to(device)

			# Forward pass for training (implement according to model specifics)
			model.optimize_parameters(mels, labels)

	# Evaluate on the query set
	model.eval()
	metrics = []
	metric_fn = metric_fn.to(device)

	for batch in tqdm.tqdm(query_set_loader, desc='Query Set'):
		labels, mels, _, _ = batch
		labels = labels[task].to(device)
		mels = mels.to(device)

		# Your existing code to compute logits and metrics
		audio_features = F.normalize(model.encode_audio(mels), dim=-1)
		text_features = F.normalize(model.encode_text(labels), dim=-1)

		logits_per_audio = (audio_features @ text_features.T)

		metric = metric_fn(logits_per_audio, labels)
		metrics.append(metric)

	avg_metric = calculate_average(metrics)
	return avg_metric

def main(args):
	# Model loading
	model = PLCLARA.load_from_checkpoint(args.model_path).to(device)

	# DataModule setup for few-shot
	dataset, templates, classes = get_dataset(
		task=args.task,
		dataset_name=args.dataset_name,
		root_cfg_path=args.root_cfg_path,
		support_set_size=args.support_set_size,
		query_set_size=args.query_set_size,
		batch_size=args.batch_size,
		num_workers=args.num_workers
	)

	# Metric definition
	num_classes = len(classes)
	metric = MetricCollection({
		f"rec@{top_k}": Recall(task='multiclass', num_classes=num_classes, top_k=top_k)
		for top_k in args.top_k if top_k <= num_classes
	})
	metric.add_metrics({f"AP": AveragePrecision(task='multiclass', num_classes=num_classes)})

	# Run evaluation
	dataset.setup()
	support_set_loader = dataset.support_set_dataloader()
	query_set_loader = dataset.query_set_dataloader()
	results = run(model, support_set_loader, query_set_loader, metric, args.task)

	return results

if __name__ == '__main__':
	import argparse
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, help='Path to model')
	parser.add_argument('--task', type=str, choices=['texts', 'gender', 'emotion', 'age', 'sounds', 'speech'], help='Task to run')
	parser.add_argument('--dataset_name', type=str, required=True, help='if task is sounds or emotion, specify dataset name')
	parser.add_argument('--root_cfg_path', type=str, default='./config/', help='root path to config files')
	parser.add_argument('--top_k', type=int, nargs='+', default=[1,5,10], help='Top k metrics to use')
	parser.add_argument('--batch_size', type=int, default=8, help='Dataloader batch size')
	parser.add_argument('--num_workers', type=int, default=12, help='Dataloader number of workers')
	parser.add_argument('--support_set_size', type=int, default=5, help='Number of samples in the support set')
	parser.add_argument('--query_set_size', type=int, default=20, help='Number of samples in the query set')

	args = parser.parse_args()

	results = main(args)

	pprint(results)
