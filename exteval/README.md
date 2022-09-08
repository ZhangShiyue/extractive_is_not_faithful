## ExtEval Metric


### Requirements

* Python 3
* requirements.txt

#

### Quick Start
Run the following command to get the ExtEval scores and its sub-metric scores 
for our data (../data/data_finalized.json).
The results will be save as ../data/data_exteval.json.
```
python exteval.py --data_file ../data/data_finalized.json --output_file ../data/data_exteval.json
```
Note that this requires each example in the input data file to have the following four items:
```
key: {   # key = exampleId_modelName, e.g., 0_banditsumm
    "document":   the source document,
    "summary": the system summary, sentences (or units) are seperated by <t> </t>,
    "document_for_annotation": the processed document to show on the annotation HTML webpage,
    "summary_for_annotation": the processed summary to show on the annotation HTML webpage,
}
``` 
If you want to get ExtEval scores for your own data, please see below and get the processed
"document_for_annotation" and "summary_for_annotation" before running the exteval.py script.

#

### Preprocess Documents and Summaries
Assume you have already compiled your data into the same format as data_sample.json in the current directory, 
then you can get run the preprocess.py script to get document and summary annotated with coreference clusters.
```
python preprocess.py --data_file data_sample.json --output_file data_sample_processed.json
``` 
Then, the data_sample_processed.json can be used to compute ExtEval scores as the input data file.

