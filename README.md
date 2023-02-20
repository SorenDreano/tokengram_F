# Tokengram_F
A chrF-alternative Machine Translation score using SentencePiece
**Table of Contents**

[TOCM]

[TOC]

# [Installation](https://github.com/SorenDreano/tokengram_F/README.md "Installation")
## [pip](https://github.com/SorenDreano/tokengram_F/README.md "pip")
`python3 -m pip install tokengram_F --user`
## [Compile](https://github.com/SorenDreano/tokengram_F/README.md "Compile")
`git clone https://github.com/SorenDreano/tokengram_F && cd tokengram_F`
`python3 -m build $$ cd dist`
`python3 -m pip install tokengram_f-x.x.x-py3-none-any.whl`
# [Usage](https://github.com/SorenDreano/tokengram_F/README.md "Usage")
## [Sentence](https://github.com/SorenDreano/tokengram_F/README.md "Sentence")
    from tokengram_F.tokengram_F import get_tokenizer, compute_batch_tokengram_F
	hypothesis = "tokengram_F is based on chrF and uses SentencePiece instead of word n-grams"
	reference = "tokengram_F is a SentencePiece-based chrF-alternative"
	tokenizer = get_tokenizer("eng", 50000)
	score = compute_batch_tokengram_F([reference], [hypothesis], "eng", 2, 6, 3.0, 50000, tokenizer, None)[1]
### [Multiple sentences](https://github.com/SorenDreano/tokengram_F/README.md "Multiple sentences")
    from tokengram_F.tokengram_F import get_tokenizer, compute_batch_tokengram_F
	hypotheses = ["tokengram_F is based on chrF and uses SentencePiece instead of word n-grams",
	"How are you doing?"]
	references = ["tokengram_F is a SentencePiece-based chrF-alternative",
	"How are you going?"]
	tokenizer = get_tokenizer("eng", 50000)
	score = compute_batch_tokengram_F(references, hypotheses, "eng", 2, 6, 3.0, 50000, tokenizer, None)[1]
## [File](https://github.com/SorenDreano/tokengram_F/README.md "File")
    import os
    from tokengram_F.tokengram_F import get_tokenizer, compute_file_tokengram_F
	hypotheses = os.path.join("path", "to", "hypotheses.txt")
	references = os.path.join("path", "to", "references.txt")
	tokenizer = get_tokenizer("eng", 50000)
	score = compute_batch_tokengram_F(references, hypotheses, "eng", 2, 6, 3.0, 50000, tokenizer, None)[1]
