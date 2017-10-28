# OnlinePMI

The options of the program can be obtained by typing ```python3 online_pmi.py --h```.

An example to train the program on `SCA` representation and evaluate the output based on B-cubed F-scores.

```python3 online_pmi.py -i uniform_data/huon.tsv.uniform -mi 15 --eval -A sca --prune```

The program supports three sound classes: ASJP, DOLGO, SCA. All the three sound class representations are obtained through lingpy code by converting from IPA. I added sample datasets in uniform_data folder.

`--prune` The program can prune the word lists between iterations. This can cause the program to converge faster on large datasets but does not make any difference on small datasets such as Huon language group.

`--nexus` The program outputs a nexus file for the corresponding dataset. The nexus file can be read by MrBayes and then used to perform phylogenetic inference.

The program outputs two files ending with `.pmi` and `.cognates`. `.pmi` shows the PMI matrix. `.cognates` shows the cognate judgments given by the program.
