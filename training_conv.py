from datetime import datetime
from pathlib import Path

import spacy
from sklearn.model_selection import train_test_split
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

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
        scores = torch.tensor([[meteor_score,
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
                               ]],
                              dtype=torch.float32)

        # Pass the scores through the convolutional and fully-connected layers
        x = scores.unsqueeze(1)  # add a channel dimension
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool1d(x, kernel_size=x.size(-1))  # global max pooling
        x = x.view(-1, 128)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.sigmoid(self.fc2(x))
        return x


def train(net,
          optimizer,
          criterion,
          dataloader_train,
          reducer,
          dataloader_val,
          num_epochs) -> None:
    for epoch in range(num_epochs):
        running_loss = 0.0
        net.train()
        for i, batch in enumerate(dataloader_train):
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
        print("Training. Epoch {}: Loss = {}".format(epoch + 1, running_loss / len(dataloader_train)))

        running_loss_val = 0.0
        net.eval()
        for i, batch in enumerate(dataloader_val):
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

                running_loss_val += loss.item()

        reducer.step(running_loss_val / len(dataloader_val))

        # Print the average loss for the epoch
        print("Eval. Epoch {}: Loss = {}".format(epoch + 1, running_loss_val / len(dataloader_val)))
        print('-' * 25)


if __name__ == '__main__':
    DATA_TRAIN, DATA_VAL = train_test_split(DATA, train_size=0.9, test_size=0.1, shuffle=True)
    print(f'TRAIN LENGTH: {len(DATA_TRAIN)}, VAL LENGTH: {len(DATA_VAL)}')
    dataset_train = MyDataset(DATA_TRAIN)
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

    dataset_val = MyDataset(DATA_VAL)
    dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=True)

    # Instantiate the network and optimizer
    net = EMEQT()

    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)

    # Define the loss function
    criterion = nn.MSELoss()

    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5)

    # Train the network
    try:
        # Train the network
        train(net=net,
              optimizer=optimizer,
              criterion=criterion,
              dataloader_train=dataloader_train,
              dataloader_val=dataloader_val,
              reducer=scheduler,
              num_epochs=100)
    except:
        pass
    finally:
        # Save the network
        current_date = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        save_path = Path(__file__).parent.joinpath('models').joinpath(f'metric_ensemble_conv_{current_date}.pt')
        torch.save(net, save_path)
