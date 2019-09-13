# OpenNER

## Requirements
- python3
- pip3 install -r requirements.txt

## Download

- Models:
  - OpenNER-base: [model](https://drive.google.com/file/d/1Zwkp6pvuqVn2idO5KQp_Casx4VjBxHyB/view?usp=sharing)
  - OpenNER-large: [model](https://drive.google.com/file/d/15ID9cOSJC2NMJNrv6vqbdXfOlHb7wT3w/view?usp=sharing)
- Data set
  - AnchorNER: [data](https://drive.google.com/file/d/1Qm3WCWLOPRgTJUuXBKrOLPr20V5yOa5i/view?usp=sharing)

You can download the models above and put them wherever you want. You only need to set the model_dir parameter in the OpenNER class to the address where the model is located. See the examples below.

## Example Usage

### 1. Load OpenNER in python

#### Parameter setting

```python
from OpenNER import OpenNER

# load OpenNER-base
tagger = OpenNER()

# load OpenNER-large
tagger = OpenNER(bert_model='bert-large-cased', model_dir='model/OpenNER_large')
```

The default value of bert_model is bert-base-cased and the default value of model_dir is 'model/OpenNER_base'.

If you want to load OpenNER-large, you need to change bert_model to bert-large-cased.

If the location of your model is not in the default address, please change model_dir to the address where your model is located.

#### Predict one sentence

```python
from OpenNER import OpenNER

# load OpenNER
tagger = OpenNER()

# run NER over sentence
print(tagger.predict("Despite winning the Asian Games title two years ago, Uzbekistan are in the finals as outsiders."))  

```

This should print:

```
['Despite O', 'winning O', 'the O', 'Asian B-MISC', 'Games I-MISC', 'title O', 'two O', 'years O', 'ago O', ', O', 'Uzbekistan B-LOC', 'are O', 'in O', 'the O', 'finals O', 'as O', 'outsiders O', '. O']  

```

#### Predict multiple sentences

You can also predict a batch of sentences, using the following codes:

```python
# run NER over a batch of sentences
print(tagger.predict_batch(["Despite winning the Asian Games title two years ago, Uzbekistan are in the finals as outsiders.", "William Wang is an Assistant Professor from UCSB."]))  

```

This should print:

```
[['Despite O', 'winning O', 'the O', 'Asian B-MISC', 'Games I-MISC', 'title O', 'two O', 'years O', 'ago O', ', O', 'Uzbekistan B-LOC', 'are O', 'in O', 'the O', 'finals O', 'as O', 'outsiders O', '. O'],   
['William B-PER', 'Wang I-PER', 'is O', 'an O', 'Assistant O', 'Professor O', 'from O', 'UCSB B-ORG', '. O']]  

```

#### Predict sentences in file

You can also predict sentences in file, using the following codes:

```python
# run NER over a file containing multiple sentences
tagger.predict_file("input.txt", "output.txt")  

```

The format of input file should be one word per line and each sentence is separated by a blank line.

For example:

```
William
Wang
is
from
UCSB
.
```

Each line of the output file is in this format: "token tag"

For example:

```
William B-PER
Wang I-PER
is O
from O
UCSB B-ORG
. O
```

### 2. Use OpenNER in command

```
CUDA_VISIBLE_DEVICES=0 python main.py --input=input.txt --output=output.txt --bert_model=bert-base-cased --model_dir=model/OpenNER_base/ --max_seq_length=128 --eval_batch_size=32
```

### 3. Train correction model

```
cd correction
python run_correction_model.py --train_file=data/D1_train.txt --dev_file=data/D1_dev.txt --test_file=data/D1_test.txt --bert_model=bert-base-cased --task_name=ner --output_dir=out_D1_model --max_seq_length=128 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.4
```

Each line of the input file is in this format: "token Wiki_label DocRED_label"

If you only want to do testing, just remove do_train from the script above.

