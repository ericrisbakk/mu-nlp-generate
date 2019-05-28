# mu-nlp-generate
Project for Natural Language Processing course @ Maastricht University, bachelor of Data Science and Knowledge Engineering.

Note that the data used is not directly available in this repository - but is assumed to exist in the LocalData folder. The dataset with its respective filename and source location can be found below.

**Dataset / FileName / Location**
  
1. Poem Dataset / "PoetryData" / https://www.kaggle.com/ultrajack/modern-renaissance-poetry

## Resources of interest:
- https://www.kaggle.com/mousehead/songlyrics
- https://www.kaggle.com/paultimothymooney/poetry-generator-rnn-markov
- http://groups.inf.ed.ac.uk/cup/ddd/
- https://crtranscript.tumblr.com/transcripts
- http://www.erinhengel.com/software/textatistic/?fbclid=IwAR2O5evMt-xbYwOuLgdhYFU9ItmhiQ3bQhrjSAOFOF8aY0p_Ytu1b4GjKqw

Two approaches:
N-gram model combined with bag of words.
LSTM (has an error)

Approach 1:
generates a dictionairy for combination of words and counts how often they occur for each n-gram up to the maximum inputted size.

2 experiments:
- how does the n-gram size inpact the results?
- does cleaning the text generate better texts?

these asnwers are based on a scoring metric provided by textatistic.

further explanation about the approaches and tests can be found in the report.




