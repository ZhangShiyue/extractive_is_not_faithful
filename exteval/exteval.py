import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from nltk import sent_tokenize
from allennlp.predictors.predictor import Predictor

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/stanford-sentiment-treebank-roberta.2021-03-11.tar.gz",
    cuda_device=0)


def preprocess(document, summary):
    all_document_sents = document.lower().replace('"', "").replace("`", "").replace("''", "").replace('-', '').replace(
        "p ! nk", "p!nk").replace("d.j .", "d.j.").replace("d.c .", "d.c.").replace("u.s.", "u.s . ").replace(
        'p.m.', 'p.m .').replace('a.m.', 'a.m .').split("<br>")
    document_sents = [(di, sent.split("<u>")[1].split("</u>")[0].strip().lower().split())
                      for di, sent in enumerate(all_document_sents) if "<u>" in sent]
    summary = summary.lower().replace('"', "").replace("`", "").replace("-lrb-", "(").replace("-rrb-", ")").replace(
        "p ! nk", "p!nk").replace("''", "").replace("d.j .", "d.j.").replace("d.c .", "d.c.").replace(
        "u.s .", "u.s.").replace('-', '').replace("nyong  o", "nyong'o").replace("nyong ' o", "nyong'o").replace(
        'co .', 'co.').replace('mass.', 'mass').replace('dr.', 'dr .').replace('mexico.', 'mexico .').replace(
        "u.s.", "u.s . ").replace('a&m', 'a & m').replace('f * * * * * g', 'f*****g').replace(
        'p.m.', 'p.m .').replace('a.m.', 'a.m .')
    summary_sents = [sent.split('.', 1)[1].strip().split()
                     for sent in summary.split("<br>")]
    return all_document_sents, document_sents, summary_sents


def locate_summaries(document_sents, summary_sents):
    # locate summary sents
    locations = []
    summary_sent_indexes = []
    for senti, swords in enumerate(summary_sents):
        if len(swords) == 0:
            continue
        dptr, match = 0, 0
        while dptr < max(map(lambda x: len(x[1]), document_sents)):
            for di, dwords in document_sents:
                if dptr >= len(dwords):
                    continue
                sptr_tmp, dptr_tmp = 0, dptr
                while sptr_tmp < len(swords) and dptr_tmp < len(dwords):
                    dword = dwords[dptr_tmp]
                    sword = swords[sptr_tmp]
                    if "[" in dword or "<span" in dword or "span>" in dword or "color:#" in dword:
                        dptr_tmp += 1
                        continue
                    elif "[" in sword or "<span" in sword or "span>" in sword or "color:#" in sword:
                        sptr_tmp += 1
                        continue
                    elif dword == sword:
                        match += 1
                    else:
                        break
                    sptr_tmp += 1
                    dptr_tmp += 1
                    if match >= 5:
                        break
                if match >= 5 or match / len(swords) > 0.9:
                    locations.append([di, dptr])
                    summary_sent_indexes.append(senti)
                    break
                else:
                    match = 0
            if match >= 5 or match / len(swords) > 0.9:
                break
            dptr += 1
    return locations, summary_sent_indexes


def coref_disco_metric(document, summary):
    # get document and summary sentences
    all_document_sents, document_sents, summary_sents = preprocess(document, summary)

    # locate summary sentences in document
    locations, summary_sent_indexes = locate_summaries(document_sents, summary_sents)

    errors = {"IncorCorefEval": 0, "IncomCorefEval": 0, "IncomDiscoEval": 0}

    # match coreference
    scorefs, dscorefs, scorefs_map, rev_scorefs_map = {}, {}, {}, {}
    scoref, smention, dcoref, dmention = None, [], None, []

    for si, senti in enumerate(summary_sent_indexes):
        swords = summary_sents[senti]
        di, dptr = locations[si]
        dwords = dict(document_sents)[di]

        sptr, plain_words = 0, []
        while sptr < len(swords) and dptr < len(dwords):
            sword = swords[sptr]
            dword = dwords[dptr]

            if "<span" in sword or "color:#" in sword:
                # if <span or color in sword, then go to next word in the summary
                sptr += 1
            elif "[" in sword:
                # if coref number in sword, save it to scoref and go to next summary word
                scoref = sword
                sptr += 1
            elif "span>" in sword:
                # if it is the end of a span, save scoref and smention
                if scoref not in scorefs:
                    scorefs[scoref] = []
                scorefs[scoref].append(smention)

                if dcoref is not None and "span>" in dword and dmention == smention:
                    # if there is a same mention in document
                    if scoref not in scorefs_map:
                        scorefs_map[scoref] = dcoref  # save the map
                        rev_scorefs_map[dcoref] = scoref  # save the reverse map
                    else:
                        # IncorCoref: if the mapping contradicts previously saved mapping
                        if scorefs_map[scoref] != dcoref:
                            errors["IncorCorefEval"] += 1

                    dcoref, dmention = None, []
                    dptr += 1

                # if it is the first mention of scoref
                if len(scorefs[scoref]) == 1 and di != 0:
                    # IncomCoref
                    if len(smention) > 1 and smention[0] in ["the", "that", "this", "these", "those", "both"]:
                        if scoref in scorefs_map:
                            # check if there is antecedent in the document
                            exist_antecedent = False
                            for doc_sent in all_document_sents[:di]:
                                if scorefs_map[scoref] in doc_sent:
                                    exist_antecedent = True
                                    break
                            # there is an antecedent in the document, and the summary fails to include it
                            if exist_antecedent:
                                errors["IncomCorefEval"] += 1

                    # IncomCoref (only when mention length=1)
                    if len(smention) == 1 and smention[0] in ["he", "she", "him", "her", "his", "they",
                                                              "them", "their", "it", "this", "that", "those",
                                                              "these"]:
                        errors["IncomCorefEval"] += 1

                scoref, smention = None, []
                sptr += 1

            else:
                plain_words.append(sword)
                # if it is a normal word
                if scoref is not None:
                    smention.append(sword)  # update smention if there is a scoref

                # find the corresponding word in the document
                if dword != sword:
                    while dword != sword:
                        if "[" in dword:
                            dcoref = dword
                        elif dcoref is not None and "span>" in dword:
                            if dcoref in rev_scorefs_map:  # if this dcoref can be mapped to a previous scoref
                                scorefs[rev_scorefs_map[dcoref]].append(dmention)
                            else:
                                if dcoref not in dscorefs:  # save dcoref to dscorefs
                                    dscorefs[dcoref] = []
                                dscorefs[dcoref].append(dmention)

                                # if it is the first mention of dcoref in summary
                                if len(dscorefs[dcoref]) == 1 and di != 0:
                                    # IncomCoref
                                    if len(dmention) > 1 and dmention[0] in ["the", "that", "this", "these",
                                                                             "those", "both"]:
                                        # check if there is antecedent in the document
                                        exist_antecedent = False
                                        for doc_sent in all_document_sents[:di]:
                                            if dcoref in doc_sent:
                                                exist_antecedent = True
                                                break
                                        # there is an antecedent in the document, and the summary fails to include it
                                        if exist_antecedent:
                                            errors["IncomCorefEval"] += 1

                                    # IncomCoref (only when mention length=1)
                                    if len(dmention) == 1 and dmention[0] in ["he", "she", "him", "her", "his", "they",
                                                              "them", "their", "it", "this", "that", "those", "these"]:
                                        errors["IncomCorefEval"] += 1
                            dcoref, dmention = None, []
                        dptr += 1
                        if dptr >= len(dwords):
                            break
                        try:
                            dword = dwords[dptr]
                        except:
                            print(dwords)
                            print(swords)
                            print(sword)
                            exit()

                if dcoref is not None:
                    dmention.append(dword)  # if there is a dcoref, then update dmention
                dptr += 1
                sptr += 1

        # IncomDisco
        if locations[si][1] == 0:  # it starts from the beginning of a sentence
            if plain_words[0] in ["and", "so", "still"]:
                if (si == 0 and di != 0) or locations[si - 1][0] != di - 1:
                    errors["IncomDiscoEval"] += 1
            else:
                for key in ["also", "however", "but", "clearly", "meanwhile", "not only", "not just",
                            "on another", "then", "moreover"]:
                    if key in ' '.join(plain_words[:5]):
                        if (si == 0 and di != 0) or locations[si - 1][0] != di - 1:
                            errors["IncomDiscoEval"] += 1
                if "on one" in ' '.join(plain_words[:5]):
                    if si == len(locations) - 1 or locations[si + 1][0] != di + 1:
                        errors["IncomDiscoEval"] += 1
        else:  # starts from the middle of a sentence
            if (si == 0 and di != 0) or locations[si - 1][0] != di:
                errors["IncomDiscoEval"] += 1

    return errors


def get_sentiment(document, batch_size=32):
    # result["probs"][0] is the probability of positive sentiment
    sents = [{"sentence": sent.strip()} for sent in sent_tokenize(document) if sent.strip()]
    sentiments = []
    for j in range(len(sents) // batch_size + 1):
        batch = sents[j * batch_size:(j + 1) * batch_size]
        if len(batch):
            results = predictor.predict_batch_json(batch)
            for result in results:
                sentiments.append(result["probs"][0])
    return np.mean(sentiments)


def exteval(data, batch_size=32):
    all_exteval = {}
    for key in tqdm(data):
        example = data[key]
        assert "document_for_annotation" in example and "summary_for_annotation" in example
        document_for_annotation = example["document_for_annotation"]
        summary_for_annotation = example["summary_for_annotation"]
        # get IncorCorefEval, IncomCorefEval, and IncomDiscoEval
        res = coref_disco_metric(document_for_annotation, summary_for_annotation)
        # binarize
        res = {metric: 1 if res[metric] else 0 for metric in res}
        # get sentiment bias
        document = example["document"]
        summary = example["summary"].replace("<t>", "").replace("</t>", " ")
        doc_sentiment = get_sentiment(document, batch_size=batch_size)
        summ_sentiment = get_sentiment(summary, batch_size=batch_size)
        sentibias = abs(doc_sentiment - summ_sentiment)
        res["SentiBias"] = sentibias
        # get exteval score
        exteval = sum([res[metric] for metric in res])
        res["ExtEval"] = exteval
        all_exteval[key] = res
    return all_exteval


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default=None,
                        type=str, help="The input data file (in json format).")
    parser.add_argument("--output_file", default=None,
                        type=str, help="The output file")
    parser.add_argument("--batch_size", default=32, type=int, help="Eval batch size")

    args = parser.parse_args()
    with open(args.data_file, 'r') as f:
        data = json.load(f)
    start_time = time.time()
    all_exteval = exteval(data, batch_size=args.batch_size)
    end_time = time.time()
    print(end_time - start_time)
    with open(args.output_file, 'w') as f:
        json.dump(all_exteval, f, indent=4)