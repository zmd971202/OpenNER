# OpenNER

### Requirements
- python3
- pip3 install -r requirements.txt

### Example Usage

Let's run named entity recognition (NER) over an example sentence. All you need to do is load OpenNER and use it to predict tags for the sentence.

#### 1. Load OpenNER in python

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

#### 2. Use OpenNER in command

```
CUDA_VISIBLE_DEVICES=0 python main.py --input=data/input.txt --output=data/output.txt
```



#### 3. Train correction model

Each line of the input file is in this format: "token Wiki_label DocRED_label"

#### run

```
python run_correction_model.py --data_dir=data/ --dev_dir=data/ --test_dir=data/ --bert_model=bert-base-cased --task_name=ner --output_dir=out_xxx --max_seq_length=128 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.4
```

