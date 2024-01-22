<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Computer%20Vision/Garbage%20Classification/docs/images/background.png" width="1200" />
</p>

---

## Introduction 

In today's world, the environmental challenges have become increasingly urgent, with 3.40 billion tonnes of waste expected to be generated annually by 2050 (Kaza et al, 2018), demanding innovative solutions to mitigate the detrimental impacts of human activities. One critical issue is the proper management of waste and the pressing need to minimise its adverse effects on the ecosystems. 

Recycling is one way to reduce waste. However, before waste can be recycled it needs to be sorted and processed. For example, segregating paper from plastics or removing hazardous waste ensures each type can be processed safely and appropriately. Apart from industrial waste production, consumers are also a first mile of the recycling process. 

---

## Motivation

Unfortunately, much more needs to be done to educate consumers on proper recycling. For example, Singapore’s domestic recycling rate declined from 22 per cent in 2018 to 17 per cent in 2019, below the European Union, which reported a domestic recycling rate of 46.4 per cent in 2017. (Channel News Asia, 2020). According to experts, a lack of recycling knowledge is one of the contributing factors to the low domestic recycling rate. A study by the Singapore Environment Council in 2018 also found that 70 per cent of respondents did not fully know what was recyclable and how waste should be sorted (Channel News Asia, 2020).

An application that can detect and inform consumers on whether an object may be recycled or not and which waste category they should be disposed to could be an intervention to potentially ameliorate the lack of education on domestic waste sorting. 

The project aims to use computer vision models to detect and classify waste materials. It is important that the models can differentiate between different waste materials so that it can inform consumers of proper waste sorting practice. For example, items that are detected as paper would be discarded into a different recycling bin compared to metal items. A trained model could be deployed to the use case of a waste sorting app, to increase domestic recycling rate by educating and informing consumers on proper waste sorting and waste identification. 

---

## Dataset

The dataset selected was sourced from Kumsetty et Al (2022). Titled ‘Trashbox’, the dataset was split into and labelled as seven distinct subcategories: cardboard, e-waste, glass, medical, metal, paper and plastic. Ultimately, five of the seven subcategories (cardboard, glass, metal, plastic and paper) were selected for the sake of brevity. The dataset consisted of 28,564 files and 4.29GB of memory. The images within the dataset largely comprised of stock images of varying sizes (figure 1, left), with few images representative of the waste found in an organic environment such as pavement, void decks, grassy environments etc. To better improve the performance of the models, the training data was altered from its initial state.

Thirty-one different background images containing no garbage were sourced from google images, with environments ranging from but not limited to roads, canals, tiled flooring, void decks, grassy environments etc. Rembg, PIL and OpenCv libraries were used to remove the existing background, crop the stock images found in the ‘Trashbox’ dataset and eventually overlaid at random locations on top of a random selection of the previously mentioned backgrounds. The locations and the image size of the overlaid garbage was noted as the ground truth of the object bounding boxes, used later in the assessment section of the project. 


<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Computer%20Vision/Garbage%20Classification/docs/images/data_overlay.png" width="1200" />
</p>

The resulting dataset consisted of 39,965 files and 67.6 GB of memory, too large of a dataset to be used unaltered without taking a significant amount of time for model training. The images were resized to 224 by 224 pixels from their original 1120 by 1120 pixels, accepting a trade-off in possible performance gains from the increased image resolution for training speed. 

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Computer%20Vision/Garbage%20Classification/docs/images/train_val_test.png" width="1200" />
</p>

---

## Methodology 

The project aimed to overcome two main tasks: Image classification and object detection. Image classification is defined as the ability of the ability for a trained model to correctly label a provided image among a set of trained classes, while object detection is defined as the ability for a train model to draw a bounding box around an object of interest within a provided image (thereby identifying the location) and then execute a proper classification of that object. A set of models were selected to tackle each of the two tasks, selected due to their recency, ease of implementation and performance whilst having a variety of architecture.

The models were trained and tested with the generated images, with the results documented in the attached appendix (figures 8 to 11). To validate the models, a curated set of non-generated images consisting of garbage found in organic environments (such as one shown in figure 2 below) were used.

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Computer%20Vision/Garbage%20Classification/docs/images/validation_img.png" width="600" />
</p>

---

## Model Selection (Classification)

Transfer learning using models Resnet 50, EfficientNetV2L and Vision Transformer (ViT) were selected for the task of image classification. 

### Resnet 50

Developed by He et al (2015), Resnet50 is a computer vision model on a 50-layer convolutional neural network architecture (CNN). Utilization of residual learning (figure 2) allows the convolution network to overcome commonly encountered degradation associated with vanishing and exploding gradients. The pre-trained model was trained on images available on ImageNet.

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Computer%20Vision/Garbage%20Classification/docs/images/resnet.png" width="1200" />
</p>

### EfficientNetV2L

EfficientNetV2L (Tan & Le, 2021) is built upon two main concepts, compound coefficient model scaling and neural architecture search (NAS). Often, the continual addition of neural network layers do not necessarily result in a performance improvement of a CNN. By having a set of scaling coefficients, EfficientNet architecture allows for neural networks to be developed with a uniform set of neural network widths, depth and resolution (figure 3). NAS allows for a systematic approach to model tuning via defining search space, search strategy and set performance metrics to further develop a model with good performance. 

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Computer%20Vision/Garbage%20Classification/docs/images/effnet.png" width="1200" />
</p>


### Vision Transformer (ViT)

Vision Transformer (henceforth referred to as ViT) was developed by Dosovitskiy et al (2020). Trained on Google’s JFT-300M image dataset, ViT architecture (figure 4) differs vastly from CNN architecture. Transformers, often used in natural language processing (NLP), focus on creating encodings for each set of data (such as a sentence, document or image) by forming associations between a token (or image pixel) and all other tokens. To apply a similar NLP approach to an image without alteration will be impractical, as the time complexity of such an operation would be O(n2), impractically large for images often thousands of pixels in width and height. Instead, ViT segments each image into multiple patches (sub-images 16 by 16 pixels in size), creates embeddings for each patch before creating a global association through a transformer encoder. Multi-layer perceptrons (MLP) consolidate the learned weights to form the classification layer of the neural network.

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Computer%20Vision/Garbage%20Classification/docs/images/vit.png" width="1200" />
</p>

---

## Model Selection (Object detection)

Transfer learning using model YOLOV3 was selected for the task of object identification. 

### YOLOV3

YOLOV3 (Redmon et al, 2016) tackles object identification in a method not unlike ViTs, by splitting a image into a series of sub-images in a grid like fashion. Conventionally, the sliding window object detection method is used for the task of object detection, which uses an approach similar to kernels in CNNs, the model learning from a window moved across the image with the image data and bounding box data. What makes YOLO unique in its approach to object detection is to first split an image into a grid and embedding visual and bounding box data within each cell of the grid. Feeding each sub-image through a CNN, the assessment of the location and appropriate label of the object of interest is assessed as a whole image. A trained model would then be able to predict several viable bounding boxes, with nonmax suppression, a method used to assess probabilities of the bounding boxes, used to determine the most appropriate bounding box for that image.

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Computer%20Vision/Garbage%20Classification/docs/images/yolo.png" width="1200" />
</p>

---

#### YOLO Methodology

After the data augmentation step, a separate input file was created for each image, containing the bounding box information in the format: class x coordinate, y coordinate, width, height. Subsequently, each image was labelled using the same name as the image file for the model's data processing. Then, the darknet weights and YOLO configuration, trained on the imagenet dataset, were utilized to train the model. Model training however, took a very long time due to the large size of the custom dataset and the specific requirements of the YOLO architecture. 
    
Consequently, a pretrained YOLO model was used to test the custom images. To achieve this, the hyperparameters were tuned according to the dataset, which consisted of 5 classes, and the output layer of the YOLO architecture was modified accordingly.
    
Regarding the architecture, the image was initially resized to 448x448 pixels, and the pixel values were normalized within the range of -1 to 1. These values were then processed through the network, which produced the network output. The architecture primarily consisted of a CNN network that extracted high-level features through a series of convolution and pooling layers. These layers captured contextual information, enabling the network to gain a deeper understanding of the image. The detection layers were responsible for predicting the bounding boxes and class probabilities. The output of the network was a tensor representing grid cells, which contained information about the bounding boxes and class probabilities.
    
When comparing the performance between overlayed images and natural images, the model exhibits superior performance on natural images compared to overlayed images. This discrepancy can be attributed to the dataset on which the model was trained. The pretrained model was trained on the COCO dataset, which consisted of common object images. As a result, the model is more adept at recognizing objects in their original context, where the background is coherent with the object itself, rather than objects superimposed on a different background.

---
    
##  Evaluation 

Evaluation of the previously mentioned models was be done primarily through the analysis of performance metrics. The main performance metrics are the accuracy, precision, recall and eventually the F1 score. For object detection, intersection over union (IOU) will be used as another evaluation metric. Intersection over union (IOU), the measure of the model’s ability to distinguish the objects from the background, will also be used as a performance metric. In addition, models were tested against a selection of images contributed by team members to roughly determine the model accuracies.

---

##  Results

For image classification, it appears that performance of the model improves with the more recent, sophisticated models. For YOLOV3, while the accuracy, precision and recall performance metrics are not as high as the test classification models, the IOU was found to be 57%, a relatively acceptable level for a model not yet trained specially on the selected dataset.
    
 <p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Computer%20Vision/Garbage%20Classification/docs/images/results.png" width="1200" />
</p>   
---

## Challenges

It is to be noted that the number of tests datapoints were not consistent between model evaluations. Through the refinement from Trashbox’s stock images, the resulting dataset consisted of images 1120 by 1120 pixels in size, approximately 40, 000 images over the 4 sub-categories, spanning 67.6 GB. With the limitation of time and computing resources, images were resized down to 224 by 224 pixels in size to allow for ease of model training. Despite this however, model training took a significant amount of time, with training of an object detection model with additional google colaboratory GPU resources on 20% of the training dataset took upwards of 7 hours. Each model was trained and tested on a variety of subset sizes in an attempt to feed in as much training and test data as possible in a reasonable amount of time. While this will introduce a level of inconsistency into model comparisons, representative sampling of both the training and testing datasets make it less likely to have large deviations in model performance than what has been tested.

---

## Memory Optimization

To grapple with the large memory size of the image data, two memory optimization methods had to be implemented, the usage of the DeepSpeed library and the application of low rank approximation (LoRA).

DeepSpeed is an open-source deep learning optimization library for PyTorch. It aims to enhance the efficiency of deep learning training by reducing computational power and memory usage while enabling better parallelism on existing hardware. The library is specifically designed for training large, distributed models with a focus on low latency and high throughput. DeepSpeed incorporates the Zero Redundancy Optimizer (ZeRO), which enables training models with 1 trillion or more parameters. Key features of DeepSpeed include mixed precision training, support for single-GPU, multi-GPU, and multi-node training, as well as customizable model parallelism.

Given the limitations imposed by computational resources, recurrent instances of GPU memory overflow were encountered during the training of the models. Despite attempts to mitigate these issues by reducing sample size and image resolution, the fine-tuning of expansive models like google/vit-large-patch16-224-in21k, available on HuggingFace, demanded substantial computing power and consistently led to runtime crashes. However, by harnessing the power of DeepSpeed, not only was a successful execution of the prodigious ViT model achieved, but significant advancements in the model optimization endeavours. Specifically, larger batch sizes, finer-grained learning rates and more expansive training sample sizes were able to be incorporated, thus capitalizing on the enhanced capabilities provided by DeepSpeed's state-of-the-art deep learning optimization library for PyTorch.
Low Rank Approximation (LoRA) is an optimization technique that reduces the number of trainable parameters by learning pairs of rank-decomposition matrices while freezing the original weights. By leveraging LoRA, the storage footprint and memory usage of the model were able to be reduced, and larger models with better performance on the downstream classification task were able to be trained.

---

## Future Work

Four approaches may be taken which are likely to further refine model performance.

### Further image augmentation

The current dataset is limited to the overlay of cropped stock images at random points on the selection of background. The inclusion of additional backgrounds, rotation and resizing of the cropped images during the overlay process would further increase the amount of training data available. Image level transformations such as flips and rotation, as well as pixel level transformations like brightness, contrast and hue adjustments is likely to provide a model performance improvement.

### Increased training time and resources

It is to be noted however, that the proposed image augmentation will further inflate the datasets, resulting in significant increases in the training and testing time required for each model. Given sufficient time and computing resources, the models may be trained and tested using the augmented dataset consisting of the original sized images (1120x1120). This will allow for the standardization of the training, as well as the testing of the models thereby resulting in a more objective comparison in model performance.

### Hyperparameter tuning

More extensive hyperparameter tuning is likely to further improve model performance. Different approaches, such as grid searches, random searches or execution of hyperparameter sweeps.


## Conclusion

The ability to classify images and identify objects was tested through transfer learning of models Resnet50, EfficientNetV2L, ViT and YOLOV3. Preliminary testing shows a general improvement of performance with more recent and sophisticated models. Given additional time and resources, a properly tuned model will be able to assist members of the public in the sorting of recyclables.

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

1. Channel News Asia. 2020. IN FOCUS: 'It is not easy, but it can be done' - The challenges of raising Singapore's recycling rate. https://www.channelnewsasia.com/singapore/in-focus-singapore-recycling-sustainability-blue-bins-waste-1339091 
2. Kaza, S., Yao, L. C., Bhada-Tata, P., & Van Woerden, F. (2018). What a Waste 2.0: A Global Snapshot of Solid Waste Management to 2050. Washington, DC: World Bank. https://doi.org/10.1596/978-1-4648-1329-0
3. N. V. Kumsetty, A. Bhat Nekkare, S. K. S. and A. Kumar M. 2018. TrashBox: Trash Detection and Classification using Quantum Transfer Learning. 31st Conference of Open Innovations. Association (FRUCT), 2022, pp. 125-130, doi: 10.23919/FRUCT54823.2022.9770922. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9770922&isnumber=9770880
4. He, K., Zhang, X., Ren, S., Sun, J. 2015. Deep Residual Learning for Image Recognition. arXiv.org. https://arxiv.org/abs/1512.03385 
5. Boesch, G. (n.d). Vision Transformer (ViT) in Image Recognition – 2023 Guide. https://viso.ai/deep-learning/vision-transformer-vit/
6. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021, June 3). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv.org. https://arxiv.org/abs/2010.11929
7. Tan, M., Le, Q., 2021. EfficientNetV2: Smaller Models and Faster Training. arXiv.org: https://arxiv.org/abs/2104.00298
8. Redmon, K. Divvala, S., Girshick, R., Farhadi, A. 2016, You Only Look Once: Unified, Real-Time Object Detection arXiv.org. https://arxiv.org/abs/1506.02640v5

