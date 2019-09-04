# OpenNER

### Example Usage

Let's run named entity recognition (NER) over an example sentence. All you need to do is load OpenNER and use it to predict tags for the sentence.

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

#### Predict multiple sentence

You can also predict a batch of sentences, using the following codes:

```python
# run NER over a batch of sentences
print(tagger.predict(["Despite winning the Asian Games title two years ago, Uzbekistan are in the finals as outsiders.", "William Wang is an Assistant Professor from UCSB."]))
```

This should print:

```
[['Despite O', 'winning O', 'the O', 'Asian B-MISC', 'Games I-MISC', 'title O', 'two O', 'years O', 'ago O', ', O', 'Uzbekistan B-LOC', 'are O', 'in O', 'the O', 'finals O', 'as O', 'outsiders O', '. O'], 
['William B-PER', 'Wang I-PER', 'is O', 'an O', 'Assistant O', 'Professor O', 'from O', 'UCSB B-ORG', '. O']]
```



