import argparse
import json
from funcs import load_checkpoint, predict

parser = argparse.ArgumentParser()
parser.add_argument('input', action = 'store')

parser.add_argument('checkpoint', action = 'store')

parser.add_argument('--topk', action='store', type=int, default = 3)

parser.add_argument('--category_names', action = 'store', default = "cat_to_name.json")

parser.add_argument('--gpu', action = "store_true", default = False)

results = parser.parse_args()

checkpoint_in = results.checkpoint

with open(results.category_names, 'r') as f:
    class_to_index = json.load(f)
    
model = load_checkpoint(checkpoint_in)        

probs, labels = predict(results.input, model, results.gpu, results.topk)
typenames = [class_to_index [item] for item in labels]
print(typenames, probs)
