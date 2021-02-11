import codecs
import glob
import json
import os
import string

import pke
from nltk.corpus import stopwords
from tqdm.auto import tqdm

punctuations = list(string.punctuation)
punctuations += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']

stoplist = stopwords.words('english') + punctuations


class RunBaselines:
    def __init__(self, parameter_file, path_ake_datasets, return_dict=False):
        with open(parameter_file) as f:
            self.params = json.load(f)

        self.params["path"] = os.path.join(path_ake_datasets,
                                           self.params["path"])
        self.params["reference"] = os.path.join(path_ake_datasets,
                                                self.params["reference"])

        self.path_to_test = os.path.join(self.params["path"], 'test')
        self.path_to_train = os.path.join(self.params["path"], 'train')
        self.path_to_val = os.path.join(self.params["path"], 'dev')

        self.return_dict = return_dict

    def run_one_model(self, model, train_files=False, val_files=False):

        # container for keyphrases
        keyphrases = {}
        stemmed_keyphrases = {}

        # get class from module
        class_ = getattr(pke.unsupervised, model, None)

        files = glob.glob(os.path.join(self.path_to_test, '*.' + self.params['extension']))

        if train_files:
            files += glob.glob(os.path.join(self.path_to_train, '*.' + self.params['extension']))

        if val_files:
            files += glob.glob(os.path.join(self.path_to_val, '*.' + self.params['extension']))

        # loop through the documents
        for input_file in tqdm(files):

            # get the document identifier
            file_id = input_file.split("/")[-1][:-4]

            # initialize the ake model
            extractor = class_()

            # read the document
            extractor.load_document(input=input_file,
                                    language=self.params["language"],
                                    normalization=self.params["normalization"])

            # extract the keyphrase candidates
            extractor.grammar_selection(grammar=self.params["grammar"])

            # filter candidates containing stopwords or punctuation marks
            extractor.candidate_filtering(stoplist=stoplist,
                                          minimum_length=3,
                                          minimum_word_size=2,
                                          valid_punctuation_marks='-',
                                          maximum_word_number=5,
                                          only_alphanum=True)

            if model in ['PositionRank', 'TextRank']:
                extractor.candidate_weighting(pos=self.params["pos"])
            else:
                extractor.candidate_weighting()

            # pour the nbest in the containers
            kps = extractor.get_n_best(n=self.params["nbest"], stemming=False)
            keyphrases[file_id] = [[u] for (u, v) in kps]

            s_kps = extractor.get_n_best(n=self.params["nbest"], stemming=True)
            stemmed_keyphrases[file_id] = [[u] for (u, v) in s_kps]

        return keyphrases, stemmed_keyphrases

    def run_baselines(self):
        for model in self.params["models"]:
            output_file = "{}/{}.{}.json".format(self.params["output"],
                                                 self.params["dataset_identifier"],
                                                 model)

            stemmed_output_file = "{}/{}.{}.stem.json".format(
                self.params["output"],
                self.params["dataset_identifier"],
                model)

            if not os.path.isfile(stemmed_output_file):
                keyphrases, stemmed_keyphrases = self.run_one_model(model)

                with codecs.open(output_file, 'w', 'utf-8') as o:
                    json.dump(keyphrases, o, sort_keys=True, indent=4)

                with codecs.open(stemmed_output_file, 'w', 'utf-8') as o:
                    json.dump(stemmed_keyphrases, o, sort_keys=True, indent=4)
