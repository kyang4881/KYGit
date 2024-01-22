<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Reinforcement%20Learning%20Projects/Blackjack%20Game/docs/images/blackjack.jpg" width="1200" />
</p>

---

In the game of Blackjack, a modified version of the classic casino game is played where there are no Kings in the deck, and the cards range from Ace to Queen. The objective is to train an agent using reinforcement learning techniques to play Blackjack and optimize winnings or minimize losses when facing the House. The environment is defined with number cards having their face value, pictures (Jack and Queen) counting as 10, and Aces being flexible with values of 1 or 11. A "blackjack" occurs when a player reaches 21 on the first two cards, the player wins unless the dealer also has a "blackjack." Wins are paid 1 to 1, while player "blackjacks" are paid 3 to 2 odds. The player can either "hit" (take a card) or "stand" (end turn) with the goal of not exceeding 21 points to avoid a "bust." The dealer, representing the house, follows specific rules for drawing cards based on their hand's value.

---

<p align="left">
  <img src="https://github.com/kysrc/ky_mas/blob/main/feature_selection_timeseries/docs/images/py_ver.png" width="150" />
</p>


## Overview

* Deck: Ace, 2-10, Jack and Queen
* Objective: Maximize winnings or minimize losses

Env:

* Ace: {1, 11}, Jack or Queen = 10, else = face value
* Win if total = 21 on the first 2 cards = blackjack
* Tie if dealer also has blackjack
* Win payout = +1, blackjack payout = +1.5, tie = 0
* Player is dealt two cards, action = {"hit", "stand"}
* Card total > 21 = busts (loses)
* Dealer has 1 face up card and draw iff hard 16, soft 17 or lower

Agents:

* Q Learning
* Monte Carlo

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/NLP%20Projects/LLMs/Fedspeak/docs/images/fedspeak_table.png" width="1200" />
</p>

---

## Environment

The class BlackjackEnvironment simulates a game of Blackjack. The class initializes the game environment with a deck of cards, assigns values to each card in the deck, and offers methods for simulating the game. The deal_card method simulates drawing a card from the deck, and the card_mapper method maps card values to their corresponding numeric values. The initial_state method initializes the game state, generating the two player cards and one face up dealer card. The player can choose to "hit" (draw a card) or "stand" (end their turn) using the player_action method. The hand_total methods calculate the total value of a hand, considering possible values for Aces. The dealer_action method determines the dealer's action based on their hand, adhering to typical Blackjack rules. The accepted_value method helps identify acceptable values for the player and dealer, ensuring values do not exceed 21. The step method simulates a single step in the game, updating player and dealer hands, calculating rewards, and determining if the game is over. The play method simulates an entire game and returns the final reward.

The following assumptions were implemented in the environment and state-transition.
* Deck Composition: The deck is modified to exclude Kings, leaving only Ace, 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, and Queen.
    * The provided code defines a Python class called BlackjackEnvironment, which simulates a simplified game of Blackjack. The class initializes the game environment with a deck of cards as defined in the assumption.

* Card Values: Number cards are considered at face value, Jack and Queen cards are both valued at 10, and Aces can be counted as either 1 or 11.

A card_mapper method maps card values to their corresponding numeric values
Blackjack Definition: A player achieving a total of 21 on the first two cards is considered to have a "blackjack." In this case, the player wins unless the dealer also has a "blackjack," leading to a "tie."

In the transition method, the player goes first and it is checked whether the player has 21, if the player has a total of 21 and the dealer doesn't, it's a blackjack and the turn ends. If the dealer also has 21, it's a tie game, otherwise the game continues until the player either stand or bust.
Payouts: Payouts for regular wins are simplified at 1 to 1, and player "blackjacks" are paid out at 3 to 2 odds.

Rewards are assigned in the transition method with +1.5 for blackjack, +1 for a normal win, 0 for a tie and -1 for a loss.
Player Actions: The agent is given two possible actions: "hit" (take a card) or "stand" (end the turn). This simplifies the agent's decision-making process.

The player's action is determined by the Agent class and consists of only the two listed actions.
Dealer Rules: The dealer (representing the house) has a single faced-up card, and they must draw an additional card if their hand is currently "hard 16" or "soft 17" (which includes an Ace) or lower.

This is implemented by the dealer_action method

```python
import jsonlines
import pandas as pd
import torch
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import Trainer, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np
import time
from datasets import DatasetDict, Dataset, concatenate_datasets
import evaluate
nltk.download("punkt")
from random import randrange
import sentencepiece
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import seaborn as sns
import matplotlib.pyplot as plt
import random
```

A class for preparing/compiling the data for preprocessing.

```python
class compileData:
    """A class for compiling the necessary json files into a dataset
    Args:
      train_data_fname (str): file name for the train data
      dev_data_fname (str): file name for the validation data
      test_data_fname (str): file name for the test data
      train_data (list): a list containing the train data from the json file
      dev_data (list): a list containing the validation data from the json file
      test_data (list): a list containing the test data from the json file
      prompt (str): A prompt for the model
    """
    def __init__(self, X_train, X_validation, X_test, y_train, y_validation, y_test, prompt):
        self.X_train = X_train
        self.X_validation = X_validation
        self.X_test = X_test
        self.y_train = y_train
        self.y_validation = y_validation
        self.y_test = y_test
        self.prompt = prompt


    def compile_dataset(self):
        """Compile the dataframes into datasets
        Returns:
            A dataset containing the train, validation, and test data, and one that doesn't include the test data
        """
        train_dataset = Dataset.from_dict({
            'original_sentence': self.X_train['sentence'],
            'features': [f"{self.prompt}\n\n{self.X_train['sentence'][i]}\n- {self.X_train['option1'][i]}\n- {self.X_train['option2'][i]}\n- {self.X_train['option3'][i]}\n- {self.X_train['option4'][i]}\n- {self.X_train['option5'][i]}" for i in range(len(self.X_train['sentence']))],
            'option1': self.X_train['option1'],
            'option2': self.X_train['option2'],
            'option3': self.X_train['option3'],
            'option4': self.X_train['option4'],
            'option5': self.X_train['option5'],
            'labels': self.y_train['answer'],
            'labels_int': self.y_train['answer_int']
        })

        dev_dataset = Dataset.from_dict({
            'original_sentence': self.X_validation['sentence'],
            'features': [f"{self.prompt}\n\n{self.X_validation['sentence'][i]}\n- {self.X_validation['option1'][i]}\n- {self.X_validation['option2'][i]}\n- {self.X_validation['option3'][i]}\n- {self.X_validation['option4'][i]}\n- {self.X_validation['option5'][i]}" for i in range(len(self.X_validation['sentence']))],
            'option1': self.X_validation['option1'],
            'option2': self.X_validation['option2'],
            'option3': self.X_validation['option3'],
            'option4': self.X_validation['option4'],
            'option5': self.X_validation['option5'],
            'labels': self.y_validation['answer'],
            'labels_int': self.y_validation['answer_int']
        })

        test_dataset = Dataset.from_dict({
            'original_sentence': self.X_test['sentence'],
            'features': [f"{self.prompt}\n\n{self.X_test['sentence'][i]}\n- {self.X_test['option1'][i]}\n- {self.X_test['option2'][i]}\n- {self.X_test['option3'][i]}\n- {self.X_test['option4'][i]}\n- {self.X_test['option5'][i]}" for i in range(len(self.X_test['sentence']))],
            'option1': self.X_test['option1'],
            'option2': self.X_test['option2'],
            'option3': self.X_test['option3'],
            'option4': self.X_test['option4'],
            'option5': self.X_test['option5'],
            'labels': self.y_test['answer'],
            'labels_int': self.y_test['answer_int']
        })

        nlp_dataset_dict_wtest = DatasetDict({
            'train': train_dataset,
            'validation': dev_dataset,
            'test': test_dataset
        })
        return  nlp_dataset_dict_wtest
    
```

A class for preprocessing the data.

```python
class preprocessor:
    """A Preprocessing class for tokenizing the features and labels
    Args:
        data_dict (dataset): A dataset containing train, validation, and test data
        padding (bool/str): A boolean or string for specifying the padding requirement
        truncation (bool): A boolean or string for specifying the truncation requirement
        tokenizer (obj): A transformer object
    """
    def __init__(self, data_dict, padding, truncation, tokenizer):
        self.data_dict = data_dict
        self.padding = padding
        self.truncation = truncation
        self.tokenizer = tokenizer
        tokenized_features = concatenate_datasets([self.data_dict["train"], self.data_dict["validation"], self.data_dict["test"]]).map(lambda x: tokenizer(x["features"], truncation=self.truncation))
        tokenized_labels = concatenate_datasets([self.data_dict["train"], self.data_dict["validation"], self.data_dict["test"]]).map(lambda x: tokenizer(x["labels"], truncation=self.truncation))
        self.max_source_length = max([len(x) for x in tokenized_features["features"]])
        self.max_target_length = max([len(x) for x in tokenized_labels["labels"]])

    def preprocess(self, data):
        """Preprocessing the data by tokenizing the features and labels
        Args:
            data (dataset): A dataset containing the train, validation, and test data
        Returns:
            An updated dataset containing tokenized inputs
        """
        # Tokenize the features
        model_inputs = tokenizer(data["features"], max_length=self.max_source_length, padding=self.padding, truncation=self.truncation)
        # Tokenize the labels
        labels = tokenizer(text_target=data["labels"], max_length=self.max_target_length, padding=self.padding, truncation=self.truncation)
        # For max length padding, replace tokenizer.pad_token_id with -100 to ignore padding in the loss
        if self.padding == "max_length": labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def map_inputs(self):
        """Map the dataset to preprocess the train, validation, and test data
        Returns:
            A preprocessed dataset
        """
        tokenized_dict = self.data_dict.map(self.preprocess, batched=True)
        return tokenized_dict
```

A class for training.

```python
class trainPipeline(preprocessor):
    """A class for computing evaluation metrics and training the model
    Args:
        model (obj): A pre-trained model
        repository_id (str): A string id for the repository
        learning_rate (float): The initial learning rate for AdamW optimizer
        per_device_train_batch_size (int): The batch size per GPU/TPU core/CPU for training
        per_device_eval_batch_size (int): The batch size per GPU/TPU core/CPU for evaluation
        weight_decay (float): The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
        save_total_limit (int): If a value is passed, will limit the total amount of checkpoints
        num_train_epochs (int): Number of epoch to train
        data_dict (dataset): A dataset containing the train, validation, and test data
        padding (bool): A boolean or string for specifying the padding requirement
        truncation (bool): A boolean or string for specifying the truncation requirement
        tokenizer (obj): A transformer object
        tokenized_dict (dataset): A dataset containing tokenized train, validation, and test data
        evaluation_strategy (str): The evaluation strategy to adopt during training
        save_strategy (str): The checkpoint save strategy to adopt during training
        load_best_model_at_end (bool): Whether or not to load the best model found during training at the end of training
        logging_strategy (str): The logging strategy to adopt during training
        logging_steps (int): Number of update steps between two logs
        overwrite_output_dir (bool): If True, overwrite the content of the output directory
        device (obj): Specifies whether to use cpu or gpu
        metric_for_best_model (str): What type of metric to use for selecting the best model
        greater_is_better (bool): Defines whether greater is better in the metric for the best model
        seed (int): A random seed

    """
    def __init__(
        self,
        model,
        repository_id,
        learning_rate,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        weight_decay,
        save_total_limit,
        num_train_epochs,
        data_dict,
        padding,
        truncation,
        tokenizer,
        tokenized_dict,
        evaluation_strategy,
        save_strategy,
        load_best_model_at_end,
        logging_strategy,
        logging_steps,
        overwrite_output_dir,
        device,
        metric_for_best_model,
        greater_is_better,
        seed

    ):
        super().__init__(data_dict, padding, truncation, tokenizer)

        self.model = model
        self.repository_id = repository_id
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.weight_decay = weight_decay
        self.save_total_limit = save_total_limit
        self.num_train_epochs = num_train_epochs
        self.fp16 = False
        self.predict_with_generate = True
        self.skip_special_tokens = True
        self.output_dir = self.repository_id
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy
        self.load_best_model_at_end = load_best_model_at_end
        self.tokenized_dict = tokenized_dict
        self.logging_dir = self.repository_id  + "/logs"
        self.logging_strategy = logging_strategy
        self.logging_steps = logging_steps
        self.overwrite_output_dir = overwrite_output_dir
        self.device = device
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.seed = seed

    def compute_metrics(self, eval_preds):
        """Compute the evaluation metrics
        Args:
            eval_preds(arr): predictions and labels
        Returns:
            Evaluation results
        """
        #metric = evaluate.load("rouge")
        preds, labels = eval_preds
        # Remove the -100
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        # Decode the predictions and true labels
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=self.skip_special_tokens)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=self.skip_special_tokens)
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        # Comptue the evaluation metric
        try:
            acc = accuracy_score(decoded_labels, decoded_preds)
        except:
            print(decoded_labels)
            print(decoded_preds)
        return {'accuracy': acc}


    def training(self):
        """A method for training the model
        Returns:
            The trained model
        """
        # Defining the data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )
        # Defining the training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy=self.evaluation_strategy,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            weight_decay=self.weight_decay,
            save_total_limit=self.save_total_limit,
            num_train_epochs=self.num_train_epochs,
            fp16=self.fp16,
            predict_with_generate=self.predict_with_generate,
            save_strategy=self.save_strategy,
            load_best_model_at_end=self.load_best_model_at_end,
            logging_dir = self.logging_dir,
            logging_strategy = self.logging_strategy,
            logging_steps = self.logging_steps,
            overwrite_output_dir = self.overwrite_output_dir,
            metric_for_best_model = self.metric_for_best_model,
            greater_is_better = self.greater_is_better,
            seed = self.seed#,
            #deepspeed="./ds_config_zero3.json"

        )
        # Defining the trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dict["train"],
            eval_dataset=self.tokenized_dict["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        return trainer
```

A class for using a pre-trained model.

```python
class pretrainedModel:
    """A class for generating results using a pretrained language model
    Args:
        data_dict (dataset): A dataset containing the train, validation, and test data
        data_selected (str): A string for selecting either the train, validation, or test data
        truncation (bool): A boolean or string for specifying the truncation requirement
        padding (bool): A boolean or string for specifying the padding requirement
        checkpoint (str): A checkpoint for the pretrained model
        tokenizer (obj): A transformer object
        model (obj): A pre-trained model
        device (obj): Specifies whether to use cpu or gpu
    """
    def __init__(self, data_dict, data_selected, truncation, padding, checkpoint, tokenizer, model, device):
        self.data_dict = data_dict
        self.data_selected = data_selected
        self.truncation = truncation
        self.padding = padding
        self.checkpoint = checkpoint
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def generate_results(self):
        """A method for generating responses based on the input text
        Returns:
            Predictions for the classifers
        """
        start_time = time.time()
        print(f"Generating outputs ...")
        print(f"Model used: {self.checkpoint}")
        preds = []
        for i in range(len(self.data_dict[self.data_selected])):
            # Encode the input sentence
            encoded_inputs = self.tokenizer(self.data_dict[self.data_selected]['features'][i], padding=self.padding, truncation=self.truncation, return_tensors="pt").to(self.device)
            # Generate the predictions
            outputs = self.model.generate(**encoded_inputs)
            # Decode the predictions
            preds.append(self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
        end_time = time.time()
        print(f"Total time taken: {(end_time-start_time)/60} mins")
        return preds


    def scoring_metrics(self, preds):
        """Compute and print the scoring metric
        Args:
            preds (list): A list containing the predictions
        """
        preds_int = []
        for i in range(len(self.data_dict[self.data_selected])):
            if self.data_dict[self.data_selected]['labels'][i] == preds[i]:
                preds_int.append(self.data_dict[self.data_selected]['labels_int'][i])
            else:
                if self.data_dict[self.data_selected]['labels_int'][i] == 1:
                    preds_int.append(2)
                else:
                    preds_int.append(1)
        print("Accuracy: ", accuracy_score(self.data_dict[self.data_selected]['labels'], preds))
```

A class for fine-tuning a model and saving the predictions.

```python
class finetunedModel:
    """A class for generating results using a finetuned model
    Args:
        repository_id (str): A string id for the repository
        device (str): A string for selecting either cpu or gpu
        model_type (str): A string indicating the type of model
        data_dict (dataset): A dataset containing the train, validation, and test data
        data_selected (str): A string for selecting either the train, validation, or test data
        pred_filepath (str): The folder path to save the predictions
    """
    def __init__(self, repository_id, device, model_type, data_dict, data_selected, pred_filepath):
        self.repository_id = repository_id
        self.device = device
        self.model_type = model_type
        self.data_dict = data_dict
        self.data_selected = data_selected
        self.predicted_labels = []
        self.predicted_labels_int = []
        self.input_data_selected = self.data_dict[self.data_selected]
        self.true_labels = self.input_data_selected["labels"]
        self.true_labels_int = self.input_data_selected["labels_int"]
        self.pred_filepath = pred_filepath

    def load_model(self):
        """A method for loading a fine-tuned model
        Returns:
            A fined-tuned model
        """
        # Load model
        start_time = time.time()
        print(f"Loading model...{self.repository_id}")
        loaded_model = pipeline(self.model_type, model=self.repository_id, device=self.device)
        end_time = time.time()
        print("Completed.")
        print(f"Total time taken: {(end_time-start_time)/60} mins")
        return loaded_model

    def generate_pred(self, model):
        """Generate the predicted labels
        Args:
            model (obj): A pre-trained model
        Returns:
            A dataframe containing the features, labels, and predicted labels
        """
        start_time = time.time()
        print("Generating predictions...")
        # Prepare input data
        input_data = self.input_data_selected['features']
        # Generate predictions
        for i in range(len(self.input_data_selected['features'])):
            predicted_label = model(input_data[i])[0]['generated_text']
            # Append the binary predicted values for the labels
            if predicted_label == self.input_data_selected['option1'][i]:
                self.predicted_labels_int.append(1)
                # Append the text predicted labels
                self.predicted_labels.append(model(input_data[i])[0]['generated_text'])
            elif predicted_label == self.input_data_selected['option2'][i]:
                self.predicted_labels_int.append(2)
                # Append the text predicted labels
                self.predicted_labels.append(model(input_data[i])[0]['generated_text'])
            elif predicted_label == self.input_data_selected['option3'][i]:
                self.predicted_labels_int.append(3)
                # Append the text predicted labels
                self.predicted_labels.append(model(input_data[i])[0]['generated_text'])
            elif predicted_label == self.input_data_selected['option4'][i]:
                self.predicted_labels_int.append(4)
                # Append the text predicted labels
                self.predicted_labels.append(model(input_data[i])[0]['generated_text'])
            elif predicted_label == self.input_data_selected['option5'][i]:
                self.predicted_labels_int.append(5)
                # Append the text predicted labels
                self.predicted_labels.append(model(input_data[i])[0]['generated_text'])
            else:
                rand_int = random.randint(1, 5)
                self.predicted_labels_int.append(rand_int)
                if rand_int == 1:
                    self.predicted_labels.append(self.input_data_selected['option1'][i])
                if rand_int == 2:
                    self.predicted_labels.append(self.input_data_selected['option2'][i])
                if rand_int == 3:
                    self.predicted_labels.append(self.input_data_selected['option3'][i])
                if rand_int == 4:
                    self.predicted_labels.append(self.input_data_selected['option4'][i])
                if rand_int == 5:
                    self.predicted_labels.append(self.input_data_selected['option5'][i])

        # Compile results into a dataframe
        res_df = pd.DataFrame({
            'original_sentence': self.input_data_selected['original_sentence'],
            "features": self.input_data_selected['features'],
            "labels": self.input_data_selected['labels'],
            "labels_int": self.input_data_selected['labels_int'],
            "option1": self.input_data_selected['option1'],
            "option2": self.input_data_selected['option2'],
            "option3": self.input_data_selected['option3'],
            "option4": self.input_data_selected['option4'],
            "option5": self.input_data_selected['option5'],
            "predicted_labels": self.predicted_labels,
            "predicted_labels_int": self.predicted_labels_int
        })
        end_time = time.time()
        print("Completed.")
        print(f"Total time taken: {(end_time-start_time)/60} mins")
        return res_df

    def scoring_metric(self):
        """Generate the accuracy score for the predicted labels
        """
        print("Accuracy: ", accuracy_score(self.true_labels_int, self.predicted_labels_int))

    def save_preds(self, preds):
        """Save predictions to a csv file
        """
        preds['predicted_labels_int'].to_csv(self.pred_filepath + '.csv', index=False, header = False)
        preds.to_excel(self.pred_filepath + '.xlsx', index=False)
        with open(self.pred_filepath + '.txt','w') as f:#, encoding='utf-16-le') as f:
            for p in preds['predicted_labels_int']: f.write(f"{strip(p)}\n")
        print(f"Predictions saved to: {self.pred_filepath}")
```

A function for mapping the data

```python
def map_data(file_name):
    data = pd.read_excel(file_name)
    data['Score'] = [str(s) for s in data['Score']]
    mapping = {'-1.0': 'Dovish', '-0.5':'Mostly Dovish', '0.0': 'Neutral', '0.5': 'Mostly Hawkish', '1.0':'Hawkish'}
    mapping2 = {'-1.0': 1, '-0.5': 2, '0.0': 3, '0.5': 4, '1.0': 5}
    mapped_values = [mapping[value] for value in data['Score']]
    mapped_values2 = [mapping2[value] for value in data['Score']]
    data['answer'] = mapped_values
    data['answer_int'] = mapped_values2
    data_labels = list(mapping.values())
    for i in range(len(data_labels)): data['option' + str(i+1)] = data_labels[i]
    data['sentence'] = [w.replace("_x000D_", "").strip() for w in data['Sentence']]
    return data
```

Prepare the data for training

```python
train_valid_df = map_data(file_name="kenn_fedspeak_20perc_train_small_mod_v5.xlsx")
display(train_valid_df.head())
test_df = map_data(file_name="kenn_fedspeak_20perc_test_small_v5.xlsx")
display(test_df.head())

X_train, X_validation, y_train, y_validation = train_test_split(train_valid_df[['sentence', 'option1', 'option2', 'option3', 'option4', 'option5']], train_valid_df[['answer', 'answer_int']], test_size=0.1, random_state=42, shuffle=True)
X_train = X_train.reset_index(drop=True)
X_validation = X_validation.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_validation = y_validation.reset_index(drop=True)

X_test = test_df[['sentence', 'option1', 'option2', 'option3', 'option4', 'option5']]
y_test = test_df[['answer', 'answer_int']]

print("X_train", np.shape(X_train))
print("X_validation", np.shape(X_validation))
print("X_test", np.shape(X_test))
print("y_train", np.shape(y_train))
print("y_validation", np.shape(y_validation))
print("y_test", np.shape(y_test))

# Compile dataset
data_compiler = compileData(X_train=X_train, X_validation=X_validation, X_test=X_test, y_train=y_train, y_validation=y_validation, y_test=y_test, prompt='What is the most logical completion for the following text?')
nlp_dataset_dict_wtest = data_compiler.compile_dataset()
nlp_dataset_dict_wtest
```
Initialize and start training

```python
# Load a pre-trained checkpoint model
checkpoint = "google/flan-t5-large"
repository_id=checkpoint + "_ky_test_copy_v11"
model=AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
tokenizer=AutoTokenizer.from_pretrained(checkpoint)

# Instantiate the preprocessor class
preprocess_data = preprocessor(
    data_dict = nlp_dataset_dict_wtest,
    padding = False,
    truncation = False,
    tokenizer = tokenizer
)
tokenized_dict = preprocess_data.map_inputs()

# Instantiate the trainPipeline class
make_pred = trainPipeline(
    model=model,
    repository_id=repository_id,
    learning_rate= 5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.001,
    save_total_limit=3,
    num_train_epochs=15,
    data_dict=nlp_dataset_dict_wtest,
    padding=True,
    truncation=True,
    tokenizer=tokenizer,
    tokenized_dict=tokenized_dict,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end="True",
    logging_strategy="steps",
    logging_steps=500,
    overwrite_output_dir=False,
    device=0,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    seed=42
)
trainer = make_pred.training()

# Train the model (medium)
trainer.train()
trainer.evaluate()
```

Training results printed

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/NLP%20Projects/LLMs/Fedspeak/docs/images/training_printouts.png" width="1200" />
</p>


Veiw the best model checkpoint

```python
trainer.state.best_model_checkpoint
```

Save the model

```python
# Path to save the best model
best_model_path = repository_id
# Save the tokenizer
tokenizer.save_pretrained(best_model_path)
# Save the model
trainer.save_model(best_model_path)
```


## Sources

1. Hansen, A. and Kazinnik, S., Can ChatGPT Decipher Fedspeak?, March 2023
2. Pan, T., and Lee, H., AI in Finance: Deciphering Fedspeak with Natural Language Processing, March 2021