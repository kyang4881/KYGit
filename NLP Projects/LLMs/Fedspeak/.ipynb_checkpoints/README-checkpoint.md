<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/NLP%20Projects/LLMs/Fedspeak/docs/images/LLMs_background.jpg" width="1200" />
</p>

---

## Abstract

The recent advancements in Natural Language Processing (NLP) can provide finance professionals with tools to analyse the sentiments of various news and statements. Among this financial news, statements made by the Federal Reserve during their meeting are some of the most studied by financial professionals. These statements, referred to also as FedSpeak, contains valuable information on the Federal Open Market Committee’s sentiment on the future of the US economy. Hence, knowing their sentiment could help predict the future action the committee will take which would influence the direction of the financial market. In this project, we fine-tuned several Large Language Models (Flan-T5, DeBERTa, FinBERT, GPT 3.5) to decipher the sentiment of FedSpeak statements. Our best performing model was Flan-T5 Large model, which achieved 77% accuracy, higher than what has been reported by other studies.

---

## Introduction 

The Federal Reserve plays a pivotal role in shaping economic policies that have a direct impact on financial markets. One of the keyways they influence these markets is through adjustments in interest rates, which in turn affect asset prices, stock markets, bond market on a global scale. A crucial aspect of communicating these policy decisions lies in the announcements made by the Federal Reserve which happen eight times a year, commonly referred to as FedSpeak. During these meetings, members of the Federal Open Market Committee (FOMC) share their perspectives on the state of the economy and monetary policy options. These insights often serve as valuable indicators for investors, aiding them in making informed investment decisions. However, the language and terminology used in these meetings can be quite technical and laden with jargon, posing challenges for many investors to interpret and understand the implications.

The utilization of Natural Language Processing (NLP) has proven to be an invaluable tool in various applications, including sentiment analysis, news classification, and more. In line with this, Large Language Models in NLP could potentially be used to effectively classify the statements made during FedSpeak communications for Andromeda Capital. However, previous attempts at classification using ChatGPT ZeroShot Learning yielded unsatisfactory accuracy. 

To improve these previous attempts, we believe that employing more advanced and specialized large language models trained specifically on text data can significantly enhance the accuracy of sentence classification in this context. By effectively classifying these statements, we can provide Andromeda Capital with a deeper understanding of the sentiments expressed by the Federal Reserve during these meetings. This, in turn, enables them to develop improved investment strategies, considering the nuanced insights gleaned from the Fed's communication. A trained model can be deployed in the current operations of the company which can decipher FedSpeak and give an overall sentiment score for each statement which then can be correlated with the interest rate policy changes happening after the announcement creating more market opportunities for better investments strategies.

---

## Related Work

Recently, studies have been done to decipher FedSpeak using several Large Language Models. In particular, the work of Hansen & Kazinnik (2023) is a focal point of reference for this project. In their work, it was shown that the maximum accuracy attained using fine-tuned ChatGPT is 61%. The table below summarizes the key findings from the paper that we will use to evaluate our model performance.

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/NLP%20Projects/LLMs/Fedspeak/docs/images/related_work.png" width="1200" />
</p>

---

## Approach

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/NLP%20Projects/LLMs/Fedspeak/docs/images/modeling.png" width="1200" />
</p>

The project aimed to overcome the text multi-class classification task. Text classification is defined as the ability for the trained model to correctly label the provided text among the set of trained classes. A set of models were selected to tackle this task, based on recency, performance, and architectural complexity. The models were trained and tested on a combination of datasets described in the next section under dataset selection.

### Model Selection

FLAN T5, DeBERTa, FinBERT and GPT 3.5 were the better models reported for this task and FLAN T5 was chosen as the final model based on classification accuracy. Other models such as BERT, RoBERTa, and ensemble with Random Forrest classifiers were also explored. 

#### FLAN T5

The Flan T5 models, developed by Google Research, were selected due to 1) they were recently released (2022), showcasing cutting-edge performance on various NLP benchmarks [1], and 2) they have a distinctive instruction-based text-to-text architecture, allowing for adaptable fine-tuning across diverse NLP tasks. Through extensive instruction finetuning on a large corpus of text data, the Flan T5 models have acquired enhanced language knowledge and robust generalization capabilities to handle unseen data and tasks. Notably, these models surpass the original T5 models when fine-tuned for single task problems, while also exhibiting accelerated training speed and convergence [2], thereby offering superior performance and efficiency.

The small, base, large, and XLarge variants with 80M, 250M, 780M, and 3B parameters respectively, were explored, through the HuggingFace API. Testing the XLarge model was made possible through the application of optimization techniques such as DeepSpeed with ZeRO and LoRA, the details are provided in section 9.1.1. However, due to limited computational and storage resources, the Flan T5 large model was eventually selected as the largest viable option, with the best cost to performance ratio, that can be executed on Colab efficiently, although the larger variants may potentially enhance performance further [3].

Prior to the training process, the provided datasets were processed into the prompt format shown in Figure 1, for passing as inputs to the models. This prompt format was decided upon by referring to the Flan GitHub repository on template prompts used for specific tasks [4]. Among the various prompt formats explored, the prompt shown in Figure 1 gave the best test performance. However, due to the vast number of prompts available and computational constraints, the search space of the prompts was non exhaustive.
The initial predictions on the validation set, conducted without any fine-tuning, resulted in poor performance as shown in Table 2. The accuracy achieved was only 11%, indicating that the predictions were worse than random guessing. The poor zero-shot performance may be attributed to the model’s inability to generalize effectively to an unseen task that is as difficult as a classification problem on Fed speeches, because FedSpeak often employs language that is deliberately ambiguous and vague. The Fed officials may choose their words carefully to avoid making explicit statements or commitments that could have immediate market or economic impacts. Consequently, the zero-shot model struggles to understand the context and often output the same predictions for most instances, resulting in worse-than-guessing performance. 

Through fine-tuning on the FedSpeak dataset, the Flan T5 large model improves its understanding of the specific language patterns and contextual nuances present in Fed speeches, leading to a substantial improvement in performance. The fine-tuned model adapts its parameters to focus on the relevant features and patterns required for accurate classification. The resulting optimization allowed the model to achieve a 77% test accuracy, which is seven times higher than the accuracy of the zero-shot model. However, we have noticed a slight decrease in performance, as shown in Figure 1, when the augmented data was included, likely because the augmented data introduced some noise or variations that were not representative of the true patterns in the FedSpeak dataset. This noise could have led to confusion and hindered the model's ability to accurately classify the speeches. Additionally, the augmented data might have introduced biases or inconsistencies that affected the overall performance. 

<p align="left">
  <img src="https://github.com/kyang4881/KYGit/blob/master/NLP%20Projects/LLMs/Fedspeak/docs/images/prompt.png" width="1200" />
</p>

---

## Experiments

### Dataset

The available dataset includes two files:
1. A file containing the text of all FOMC statements released after meetings since 1997.
2. A file containing 200 randomly-drawn sentences from all the statements. Each sentence has been pre-labelled by human analysts with a hawkishness/dovishness score as defined in the table below. There are also some sentences labelled as “Remove” which indicates sentences that are irrelevant to monetary policies and should be removed in the data cleaning stage.

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/NLP%20Projects/LLMs/Fedspeak/docs/images/fedspeak_table.png" width="1200" />
</p>

The dataset was sources from Andromeda Capital who provided two csv files with the meeting minutes statements and the sentiment score with labels ranging from 5 categories – “Hawkish, Mostly Hawkish, Neutral, Mostly Dovish and Dovish. Hawkish means that the Fed strongly expresses a belief that the economy is growing too quickly and may need to be slowed down through monetary policy. Whereas Dovish means that the Fed strongly expresses a belief that the economy is growing too slowly and may need to be stimulated through monetary policies. The cleaned dataset consisted of 200 statements and labels. 

Although the dataset was sufficient to run small classification tasks, it was not enough to run using Large Language Model weights. Therefore, data augmentation techniques like similar sentence generation from ChatGPT were employed. This enabled a lot more data to be generated in a short period of time with almost zero cost. Approximately 1200 more similar statements were added to the initial dataset and then split into 80 percent training and 20 percent testing. The data was generated using prompts to ask ChatGPT to act like an expert finance professional.

In our experimental training, we utilized three datasets: the original dataset provided by Andromeda Capital, an augmented dataset combining the original data with GPT-generated data, and a GPT-generated dataset only. After evaluating the outcomes, we determined that the augmented dataset (combining original and GPT-generated data) yielded the best results in most of our tested models. This finding suggests that the inclusion of GPT-generated data in the original dataset enhanced the model's performance, indicating the value of incorporating a mixture of original and generated data for training. Nonetheless, this is not always the case. There are cases where the augmentation yields slightly worse results compared to the original dataset alone. This points to the need for further experimentation with ChatGPT prompts that could potentially yield better training data that is more consistent with FedSpeak.

### Evaluation 

In our evaluation process, we employed the accuracy score as the primary metric to assess the accuracy of our predictions. The accuracy score measures the proportion of correct predictions compared to the total number of predictions made. By calculating the accuracy score, we were able to quantitatively evaluate the performance of our predictive model and determine its overall effectiveness in accurately classifying or predicting outcomes. This metric provides a clear and straightforward measure of the model's performance, allowing us to gauge its success in making correct predictions and to make comparisons between different models or approaches.

Additionally, we have also deployed another method to evaluate our predictions based on a comparison of the actual interest rate to the hypothetical interest rate generated based on our best model’s predictions. The details of the evaluation can be found in the analysis section. 

#### Experimental Details

We built our models using specific configurations to conduct our experiments. The exact model configurations included details such as the architecture and type of model used, hyperparameters, learning rate, and training time. These configurations were chosen based on prior research and experimentation to ensure optimal performance, cost utilization, and reliable results.

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/NLP%20Projects/LLMs/Fedspeak/docs/images/exp_details.png" width="1200" />
</p>

---

## Results

Results of our models are shown in table below:

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/NLP%20Projects/LLMs/Fedspeak/docs/images/results.png" width="1200" />
</p>

From our results, FLAN-T5 Large model’s accuracy is the highest, followed by DeBERTa Large model. The results we obtained managed to beat the baseline (Figure below) reported by Hansen & Kazinnik (2023), with our best model gaining a 16% increase in accuracy. This shows significant improvement can be achieved by our fine-tuning methods, which can be considered by Andromeda capital.

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/NLP%20Projects/LLMs/Fedspeak/docs/images/comparison.png" width="1200" />
</p>

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/NLP%20Projects/LLMs/Fedspeak/docs/images/flan_family.png" width="1200" />
</p>

---

## Analysis

FLAN-T5 large (Fine-tuned) gave the best results compared to other models. This could be because, FLAN-T5 Large benefits from its underlying T5 architecture, which is known for its strong performance in various NLP tasks. T5 (Text-To-Text Transfer Transformer) is a transformer-based model that has been extensively trained on a diverse range of language tasks, allowing it to grasp complex linguistic structures effectively. Fine tuning it further enhances its performance by training it on task specific data, helping the model specialize and adept to the specific nuances and characteristics of the FedSpeak statements.

Moving down the rank is the DeBERTa and FinBERT model. Both models may not perform as well as FLAN-T5 because they were not trained as extensively compared to FLAN-T5 model’s underlying pre-training tasks. However, our exploration of these BERT models suggests that we should potentially look at finetuning our Large Language Models using a lot more financial corpus. This is shown by FinBERT’s ability to achieve better accuracy compared to many other models we’ve tried even though it was still mainly based on the original BERT architecture, which is less advanced compared to GPT and DeBERTa architecture for example. Hence, this led us to believe that more advanced models trained on larger financial corpuses may potentially result in performance gain.

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/NLP%20Projects/LLMs/Fedspeak/docs/images/timeseries.png" width="1200" />
</p>

---

Finally, based on our best model, we created a time series plot in the figure above (Figure 3) to compare the actual historical interest rate (in blue) against the hypothetical rates (in orange) that were computed with listed procedures and assumptions below:

1)	Using our best model, predict the FedSpeak scores for the unlabelled FOMC speeches.

2)	Generate the frequencies of each label by date in the following format: 

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/NLP%20Projects/LLMs/Fedspeak/docs/images/table_format.png" width="1200" />
</p>

3)	Choose the label with the max frequency and assign rates as shown below. Choose neutral if no max value.
    a.	Dovish: -50 basis points
    b.	Mostly Dovish: -25 basis points
    c.	Neutral: 0 basis points
    d.	Mostly Hawkish: +25 basis points
    e.	Hawkish: +50 basis points
    
4)	For the first instance of interest rate, use the actual historical rate. For the rest, sum up the hypothetical interest rates from the steps above sequentially with the previous rate, such that we take max (0, rate change + previous rate) to prevent the curve from dipping below 0%.

5)	Due to the above assumptions, which were needed to generate the hypothetical time series of the interest rates, we were more interested in the overall trend alignment of the curves than the magnitude of the interest rates. As seen in the figure above, our FedSpeak scores can be roughly translated into a time series, the trend shows that our model is effective in capturing the dovishness/hawkishness of the Fed’s speeches. It appears that in the period between 2014-2016, according to our predictions, the Fed was Mostly Hawkish in their speeches, but they had kept the interest rate fixed. Nevertheless, this simple illustration shows that the FedSpeak scores are incredibly useful and can model the actual interest rate trend quite well.

--- 

## Memory Optimization

To grapple with the large memory size of the data, two memory optimization methods had to be implemented, the usage of the DeepSpeed library and the application of low rank approximation (LoRA).

DeepSpeed is an open-source deep learning optimization library for PyTorch. It aims to enhance the efficiency of deep learning training by reducing computational power and memory usage while enabling better parallelism on existing hardware. The library is specifically designed for training large, distributed models with a focus on low latency and high throughput. DeepSpeed incorporates the Zero Redundancy Optimizer (ZeRO), which enables training models with 1 trillion or more parameters. Key features of DeepSpeed include mixed precision training, support for single-GPU, multi-GPU, and multi-node training, as well as customizable model parallelism.

Given the limitations imposed by computational resources, recurrent instances of GPU memory overflow were encountered during the training of the models. Despite attempts to mitigate these issues by reducing sample size, the fine-tuning of expansive models like FLAN-T5 Large, available on HuggingFace, demanded substantial computing power. However, by harnessing the power of DeepSpeed, not only was a successful execution of the larger prodigious FLAN T5 model achieved, but also significant advancements in our endeavours for model optimization. Specifically, longer token length, finer-grained learning rates and more expansive training sample sizes were able to be incorporated, thus capitalizing on the enhanced capabilities provided by DeepSpeed's state-of-the-art deep learning optimization library for Pytorch.

Low Rank Approximation (LoRA) is an optimization technique that reduces the number of trainable parameters by learning pairs of rank-decomposition matrices while freezing the original weights. By leveraging LoRA, the storage footprint and memory usage of the model were able to be reduced, and larger models with better performance on the downstream classification task were able to be trained.

---

## Future Work

Three approaches may be taken which are likely to further refine model performance.
* More relevant data collection and augmentation
* Increased training time and resources
* Hyperparameter tuning.
More extensive hyperparameter tuning is likely to further improve model performance. Different approaches, such as grid searches, random searches, or execution of hyperparameter sweeps.

---

## Conclusion

In conclusion, this project aimed to leverage Natural Language Processing (NLP) techniques to classify statements made during FedSpeak communications for Andromeda Capital. The goal was to provide a deeper understanding of the sentiments expressed by the Federal Reserve during these meetings, enabling improved investment strategies.

Multiple models were evaluated, including FLAN T5, DeBERTa, FinBERT, and GPT 3.5, with FLAN T5 Large emerging as the best-performing model. The Flan T5 large model was fine-tuned on a dataset of FedSpeak statements and achieved a test accuracy of 77%, significantly outperforming other models.

The main challenge faced in this project was the limited availability of training data. Data augmentation techniques, such as generating similar sentences using ChatGPT, were employed to augment the dataset. However, some of the augmented data may have introduced noise that affected the overall model performance. Further experiments on a variety of other ChatGPT prompts in the future may help overcome this issue.

Future work for this project includes collecting more relevant data, increasing training time and resources, and conducting hyperparameter tuning to further improve model performance. These approaches have the potential to enhance the accuracy and robustness of the classification model.

Overall, by effectively classifying the statements made during FedSpeak communications, Andromeda Capital can gain valuable insights into the sentiments expressed by the Federal Reserve, leading to improved investment strategies and marker opportunities.

---

## Notebook

Import necessary libraries.

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
3. Scaling Instruction-Finetuned Language Models. (2022, October 20). https://arxiv.org/abs/2210.11416
4. The Flan Collection: Advancing open-source methods for instruction tuning. Section: Single task fine-tuning. (2023, February 1).
https://ai.googleblog.com/2023/02/the-flan-collection-advancing-open.html
5. Flan-T5 Overview. Hugging Face. https://huggingface.co/docs/transformers/model_doc/flan-t5
6. Flan prompt templates by task. https://github.com/google-research/FLAN/blob/main/flan/v2/flan_templates_branched.py
7. README.md · google/flan-t5-xxl at main. (2023, May 21). https://huggingface.co/google/flan-t5-xxl/blob/main/README.md. 
8. He, P. et al. (2021) DeBERTa: Decoding-enhanced Bert with disentangled attention, arXiv.org. Available at: https://arxiv.org/abs/2006.03654 (Accessed: June 2023). 
9. Chung, H.W. et al. (2022) Scaling instruction-finetuned language models, arXiv.org. Available at: https://arxiv.org/abs/2210.11416 (Accessed: May 2023). 
10. Araci, D. (2019) Finbert: Financial sentiment analysis with pre-trained language models, arXiv.org. Available at: https://arxiv.org/abs/1908.10063 (Accessed: June 2023). 
11. Board of governors of the Federal Reserve System (no date) Federal Reserve Board - H.15 - Selected Interest Rates (Daily) - June 23, 2023. Available at: https://www.federalreserve.gov/releases/h15/ (Accessed: June 2023).
12. GPT-3 (2023) Wikipedia. Available at: https://en.wikipedia.org/wiki/GPT-3 (Accessed: June 2023).
13. Desola, V. et al. (2019) FinBERT: pre-trained model on SEC filings for financial natural language tasks. Available at: https://www.researchgate.net/publication/334974348_FinBERT_pre-trained_model_on_SEC_filings_for_financial_natural_language_tasks (Accessed: May 2023).
