from datetime import datetime
from pathlib import Path

import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.chrf_score import sentence_chrf
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.nist_score import sentence_nist
from rouge import Rouge
from subtree_metric import stm
from torch.utils.data import Dataset, DataLoader

from data_preparation import DATA

POS_WEIGHTS = {
    # e.g.: big, old, incomprehensible
    'ADJ': .13,
    # e.g.: in, to, during
    'ADP': .04,
    # e.g.: up, down, then, sometime, well, exactly
    'ADV': .1,
    # e.g.: has been
    'AUX': .13,
    # e.g.: and, or, but, if, while
    'CCONJ': .06,
    'SCONJ': .06,
    # e.g.: a, an, the, this
    'DET': .01,
    # e.g.: psst, hello
    'INTJ': 0,
    # e.g.: dog, cloud
    'NOUN': .13,
    # e.g.: 0, 1, 2, 123; one, two, seven
    'NUM': .21,
    # e.g.: not, 's
    'PART': .11,
    # e.g.: I, you, he, everyone; Mary, John
    'PRON': .09,
    'PROPN': 0.09,
    # e.g.: ",", "."
    'PUNCT': 0,
    # e.g.: $, <, =, emojis
    'SYM': 0,
    # e.g.: run, eat
    'VERB': 0.13,
    # e.g.: other
    'X': 0
}


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ref_str, hyp_str, target = self.data[idx]
        return ref_str, hyp_str, torch.tensor([target], dtype=torch.float32)


class EMEQT(nn.Module):
    def __init__(self):
        super(EMEQT, self).__init__()

        self.n_scores = 14
        self.spacy_model = spacy.load('en_core_web_md')
        self.rouge = Rouge()

        # Define the layers of the network
        self.fc1 = nn.Linear(self.n_scores, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.15)
        self.dropout3 = nn.Dropout(p=0.15)

    def forward(self, ref_str, hyp_str):
        # Tokenize the reference and hypothesis strings
        ref_tokens = word_tokenize(ref_str)
        hyp_tokens = word_tokenize(hyp_str)

        # Compute the scores for the reference and hypothesis strings
        meteor_score = single_meteor_score(reference=ref_tokens,
                                           hypothesis=hyp_tokens)
        bleu_score = sentence_bleu(references=[ref_tokens],
                                   hypothesis=hyp_tokens)
        chrf_score = sentence_chrf(reference=ref_tokens,
                                   hypothesis=hyp_tokens)
        try:
            nist_score = sentence_nist(references=[ref_tokens],
                                       hypothesis=hyp_tokens)
        except ZeroDivisionError:
            nist_score = 0
        stm_score = stm.sentence_stm_augmented(reference=ref_str,
                                               hypothesis=hyp_str,
                                               nlp_model=self.spacy_model,
                                               depth=3,
                                               pos_weights=POS_WEIGHTS)
        rouge_scores = self.rouge.get_scores(hyps=hyp_str,
                                             refs=ref_str)[0]

        # Combine the scores into a single tensor
        scores = torch.tensor([meteor_score,
                               bleu_score,
                               chrf_score,
                               nist_score,
                               stm_score,
                               rouge_scores['rouge-1']['r'],
                               rouge_scores['rouge-1']['p'],
                               rouge_scores['rouge-1']['f'],
                               rouge_scores['rouge-2']['r'],
                               rouge_scores['rouge-2']['p'],
                               rouge_scores['rouge-2']['f'],
                               rouge_scores['rouge-l']['r'],
                               rouge_scores['rouge-l']['p'],
                               rouge_scores['rouge-l']['f']
                               ],
                              dtype=torch.float32)

        # Pass the scores through the fully-connected layers
        x = self.fc1(scores)
        x = self.dropout1(x)
        x = nn.functional.leaky_relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = nn.functional.leaky_relu(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = nn.functional.leaky_relu(x)
        x = self.fc4(x)

        # Apply a sigmoid activation function to produce a value in the range from 0 to 1
        output = nn.functional.sigmoid(x)

        return output


def train(net, optimizer, criterion, dataloader, num_epochs) -> None:
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            ref_strs, hyp_strs, targets = batch

            # Set the gradients to zero
            optimizer.zero_grad()
            for ref, hyp, tar in zip(ref_strs, hyp_strs, targets):
                # Compute the output of the network
                output = net(ref, hyp)

                # Compute the loss
                loss = criterion(output, tar)

                # Backpropagate the gradients
                loss.backward()

                # Update the parameters
                optimizer.step()

                running_loss += loss.item()

        # Print the average loss for the epoch
        print("Epoch {}: Loss = {}".format(epoch + 1, running_loss / len(dataloader)))


if __name__ == '__main__':
    dataset = MyDataset(DATA)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Instantiate the network and optimizer
    net = EMEQT()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)

    # Define the loss function
    criterion = nn.MSELoss()
    try:
        # Train the network
        train(net=net,
              optimizer=optimizer,
              criterion=criterion,
              dataloader=dataloader,
              num_epochs=50)
    except:
        pass
    finally:
        # Save the network
        current_date = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        save_path = Path(__file__).parent.joinpath('models').joinpath(f'metric_ensemble_{current_date}.pt')
        torch.save(net, save_path)
