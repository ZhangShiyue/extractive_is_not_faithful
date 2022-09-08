import json
import argparse
from tqdm import tqdm
from nltk import word_tokenize, sent_tokenize
from allennlp.predictors.predictor import Predictor
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
punkt_param = PunktParameters()
abbreviation = ["d.j", "p!nk"]
punkt_param.abbrev_types = set(abbreviation)
tokenizer = PunktSentenceTokenizer(punkt_param)

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz",
    cuda_device=0)

colors = ["#FF0000", "#00FFFF", "#0000FF",  "#800080",
          "#F5CE00", "#00FF00", "#FF00FF", "#FFC0CB", "#FFA500", "#A52A2A", "#800000", "#008000",
          "#808000", "#7FFD4", "#00008B", "#ADD8E6", "#C0C0C0", "#808080", "#008000"]


def locate_summary_in_document(summary, document, reorder=True):
    document = document.replace("p ! nk", "p!nk").replace("d.j .", "d.j.")
    doc_sents = tokenizer.tokenize(document)
    doc_sents_words = [word_tokenize(doc_sent) for doc_sent in doc_sents]
    if "<t>" in summary:
        summary_sents = summary.replace('<t>', '').split('</t>')[:-1]
    else:
        summary_sents = sent_tokenize(summary)

    summary = []
    for summary_sent in summary_sents:
        if summary_sent.strip():
            best_i, best_f1 = -1, -1
            summary_sent_words = word_tokenize(summary_sent)
            for i, doc_sent_words in enumerate(doc_sents_words):
                hit_words = len(set(doc_sent_words) & set(summary_sent_words))
                precision = hit_words / len(set(summary_sent_words))
                recall = hit_words / len(set(doc_sent_words))
                f1 = 2*precision*recall / (precision+recall+1e-10)
                if f1 > best_f1:
                    best_f1 = f1
                    best_i = i
            summary.append([best_i, summary_sent])

    if reorder:
        summary = sorted(summary, key=lambda x: x[0])
    summary_indexes = [i for i, _ in summary]
    summary = ' <s> '.join([sent for _, sent in summary])

    document = ' <s> '.join(["<t> "+sent+" </t>" if i in summary_indexes else
                             sent for i, sent in enumerate(doc_sents)])
    return summary, document


def find_coreference(document):
    try:
        res = predictor.predict(document=document)
    except:
        return -1
    document = res["document"]
    clusters = res["clusters"]
    if len(clusters) > len(colors):
        print("The number of correference clusters is larger than the number of colors!! Please expand the color list.")
        return -1
    else:
        cluster_map = {}
        for i, cluster in enumerate(clusters):
            for start, end in cluster:
                cluster_map[start] = [end, colors[i], i]

        new_document = []
        i = 0
        while i < len(document):
            if i in cluster_map:
                new_document.extend([f"<span style='color:{cluster_map[i][1]}'> [{cluster_map[i][2]}] "] +
                                    document[i:cluster_map[i][0] + 1] + ["</span>"])
                i = cluster_map[i][0]+1
            else:
                new_document.append(document[i])
                i += 1
        document = ' '.join(new_document)
        document = document.replace('< t >', '<u>').replace('< /t >', '</u>').split('< s >')
        document = '<br>'.join([f"{i+1}. " + sent for i, sent in enumerate(document)])
        return document, len(clusters)


def preprocess(data):
    for key in tqdm(data):
        document = data[key]["document"]
        summary = data[key]["summary"]
        res = locate_summary_in_document(summary, document, reorder=False)
        if res == -1:
            print(f"WARNING! Can't locate summary in the document of {key}")
            continue
        summary, document = res
        res = find_coreference(summary)
        if res == -1:
            print(f"WARNING! Failed to run coreference model on the summary of {key}")
            continue
        summary_for_annotation, num_sum_cluster = res
        res = find_coreference(document)
        if res == -1:
            print(f"WARNING! Failed to run coreference model on the document of {key}")
            continue
        document_for_annotation, num_doc_cluster = res
        data[key]["document_for_annotation"] = document_for_annotation
        data[key]["summary_for_annotation"] = summary_for_annotation
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default=None, required=True,
                        type=str, help="The input data file (in json format).")
    parser.add_argument("--output_file", default=None, required=True,
                        type=str, help="The output file")

    args = parser.parse_args()
    with open(args.data_file, 'r') as f:
        data = json.load(f)
    processed_data = preprocess(data)
    with open(args.output_file, 'w') as f:
        json.dump(processed_data, f, indent=4)