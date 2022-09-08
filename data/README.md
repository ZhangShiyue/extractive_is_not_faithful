### Human Annotation Data
**data_finalized.json** contains our human annotations. 

It has 1500 examples. Each example is in the following format:
```
key: {   # key = exampleId_modelName, e.g., 0_banditsumm
    "document":   the source document,
    "summary": the system summary, sentences (or units) are seperated by <t> </t>,
    "reference": the reference summary, sentences are seperated by <t> </t>,
    "document_for_annotation": the processed document to show on the annotation HTML webpage,
    "summary_for_annotation": the processed summary to show on the annotation HTML webpage,
    "incorrect_coref": whether it has incorrect coreference, yes or no,
    "incorrect_coref_answer": the problematic mentions, e.g., s1-she,
    "incorrect_coref_comment": the justification of the choice,
    "incomplete_coref": whether it has incomplete coreference, yes or no,
    "incomplete_coref_answer": the problematic mentions, e.g., s1-he,
    "incomplete_discourse": whether it has incomplete discourse, yes or no,
    "incomplete_discourse_answer": the problematic sentences,
    "incorrect_discourse": whether it has incorrect discourse, yes or no,
    "incorrect_discourse_answer": the problematic sentences,
    "incorrect_discourse_comment": the justification of the choice,
    "annotators": the ids of two annotators,
    "misleading1": the misleading label for annotator 1 (the first annotator of annotators),
    "misleading_comment1": the comment from annotator 1,
    "misleading2": the misleading label for annotator 2 (the second annotator of annotators),
    "misleading_comment2": the comment from annotator 2,
}
```

### Human Annotation Interface

mturk.html is the HTML interface we used for data annotation on Amazon Mechanical Turk Sandbox.


### Metric Scores

data_exteval.json and data_other_metrics.json contain the scores of our ExtEval and the other 5 metrics, respectively.

