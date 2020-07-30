# STANCY

STANCY is a stance classification model based on consistency cues. The architecture of this model is propsed in the following paper:

[STANCY: Stance Classification Based on Consistency Cues](https://arxiv.org/abs/1910.06048). Kashyap Popat, Subhabrata Mukherjee, Andrew Yates, Gerhard Weikum
In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, EMNLP 2019. 

```bibtex
@inproceedings{popat-etal-2019-stancy,
    title = "{STANCY}: Stance Classification Based on Consistency Cues",
    author = "Popat, Kashyap  and
      Mukherjee, Subhabrata  and
      Yates, Andrew  and
      Weikum, Gerhard",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1675",
    doi = "10.18653/v1/D19-1675",
    pages = "6413--6418",
}
```
## Code Structure
The code of this model is based on the [Hugging Face](https://github.com/huggingface) library of pretrained transofrmers. It is structured in the following manner:

*  **run_classifier.py:**
This is the main code for training and evaluating the model. This is based on the pretrained BERT from huggingface.

* **modeling.py:**
This file contains our model ```BertForSequenceClassificationDualLoss``` for stance classificarion along with supporting models based on BERT.

* **file_utils.py:**
This file contains some util functions for processing the files.

* **BERT_CACHE:**
This folder contains the pre-trained BERT model. Download the folder from [here](https://www.dropbox.com/s/u2yz5vja4ed1v7d/BERT_CACHE.zip?dl=0).

* **data:**
This folder contains the train/dev/test datasets.

* **requirements.txt:**
Required dependencies to run the code. 


## Running the Code

To train and evaluate the model, run the ```run_classifier.py``` with input arguments. Detailed description of these arguments are in the ```main()``` function. For example, one can train and test the model, using the following command:

```console
python -u run_classifier.py --data_dir ../data/ --bert_model=bert-base-uncased --task_name=stance --output_dir=./bert_dual_loss_15 --cache_dir=./BERT_CACHE/ --do_train --do_eval --do_lower_case --learning_rate=1e-5 --num_train_epochs=15 --train_batch_size=24
```

