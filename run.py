import argparse
import transformers

def my_parse_args():

	# can add directory options
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_type', type = str, default = "bert")
	parser.add_argument('--mode', type = str, default = "train")
	parser.add_argument('--n_epochs', type = int, default = 3)
	parser.add_argument('--batch_size', type = int, default = 8)
	parser.add_argument('--num_attention_heads', type = int, default = 3)
	parser.add_argument('--cnn_feature_dim', type = int, default = 1024)
	parser.add_argument('--seed', type = int, default = 123)
	parser.add_argument('--learning_rate', type = float, default = 1e-3)
	args = parser.parse_args()
	return args

def main():
	args = my_parse_args()
	# set seed for random, torch, and numpy
	set_seed(args.seed)
	
	if args.model_type.lower() == "bert":
		config = BertConfig.from_pretrained("bert-base-uncased")
		tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
	elif args.model_type.lower() == "electra": # electra
		config = ElectraConfig.from_pretrained("google/electra-large-discriminator")
		tokenizer = ElectraTokenizer.from_pretrained("google/electra-large-discriminator")
		model = BertForQuestionAnswering.from_pretrained("google/electra-large-discriminator")
	else:
		print("Please enter either bert or electra for model type")
		exit()

	# set model to call to VPLM constructor passing the original model
	# if args.mode == "train":
		# load train dataset