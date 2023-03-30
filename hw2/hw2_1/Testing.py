import sys
import torch
import json
from TrainTestAnalysis import TestDataset, test_model
from torch.utils.data import DataLoader
from bleu_eval import BLEU
import pickle




model_path = './SavedModel/my_model.h5'
model = torch.load(model_path, map_location=lambda storage, loc: storage)
test_data_path = '/testing_data/feat'
test_dataset = TestDataset('{}'.format(sys.argv[1]))
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=8)

with open('i2w.pickle', 'rb') as handle:
    i2w = pickle.load(handle)

model = model.cuda()
predicted_captions = test_model(test_dataloader, model, i2w)
output_path = sys.argv[2]
with open(output_path, 'w') as f:
    for id, caption in predicted_captions:
        f.write('{},{}\n'.format(id, caption))


test_annotations = json.load(open('/testing_label.json'))
predicted_captions = {}
with open(output_path, 'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        predicted_captions[test_id] = caption

bleu_scores = []
for item in test_annotations:
    caption_references = [x.rstrip('.') for x in item['caption']]
    bleu_scores.append(BLEU(predicted_captions[item['id']], caption_references, True))

# Compute the average BLEU score for all the videos
average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print("Average BLEU score is " + str(average_bleu_score))
