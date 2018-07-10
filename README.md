# Genomic Grammar
A series of neural networks and algorithms for identifying a genomic grammar that can be used to classify regulatory circuits in the mammalian genome. The current model, is a convolutional network that uses a dot-product attention mechanism. This network has performance on par with or exceeding current state-of-the-art methods such as DeepBind at distinguishing regulatory circuits from random genomic background. Additionally, the attention mechanism of the model can be used to extract the architectures of regulatory circuits in the genome via a post-processing step performed on the output of the model.

Please see the recent abstract submitted to the International Workshop on Bio-Design Automation for details: [Download abstract](https://jenhantao.github.io/files/2018-07-31_genomic_grammar.pdf)

## Model Overview
![](https://raw.githubusercontent.com/jenhantao/genomic_grammar/master/model_architecture.png "Genomic Grammar Model Overview")
