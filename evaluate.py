"""
function: evaluate blue-4, meteor, cider and rough-l
"""
import io
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from util.CustomedBleu.smooth_bleu import smooth_bleu
from util.CustomedBleu.bleu import _bleu
import argparse
from util.meteor.meteor import Meteor
from util.rouge.rouge import Rouge
from util.cider.cider import Cider

class Evaluate():

    def bleu_so_far(self, refs, preds):
        # https://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu_score.SmoothingFunction

        c_bleu1 = corpus_bleu(refs, preds, weights=(1, 0, 0, 0))
        c_bleu2 = corpus_bleu(refs, preds, weights=(0.5, 0.5, 0, 0))
        c_bleu3 = corpus_bleu(refs, preds, weights=(1/3, 1/3, 1/3, 0))
        c_bleu4 = corpus_bleu(refs, preds, weights=(0.25, 0.25, 0.25, 0.25))

        i_bleu2 = corpus_bleu(refs, preds, weights=(0, 1, 0, 0))
        i_bleu3 = corpus_bleu(refs, preds, weights=(0, 0, 1, 0))
        i_bleu4 = corpus_bleu(refs, preds, weights=(0, 0, 0, 1))

        c_bleu1 = round(c_bleu1 * 100, 2)
        c_bleu2 = round(c_bleu2 * 100, 2)
        c_bleu3 = round(c_bleu3 * 100, 2)
        c_bleu4 = round(c_bleu4 * 100, 2)
        i_bleu2 = round(i_bleu2 * 100, 2)
        i_bleu3 = round(i_bleu3 * 100, 2)
        i_bleu4 = round(i_bleu4 * 100, 2)

        ret = ''
        # https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
        print('Cumulative 4-gram BLEU (BLEU-4): ', c_bleu4)
        print('Cumulative 1/2/3-gram BLEU: ', c_bleu1, c_bleu2, c_bleu3)
        print('Individual 2/3/4-gram BLEU: ', i_bleu2, i_bleu3, i_bleu4)

        sf = SmoothingFunction()
        all_score = 0.0
        count = 0
        for r, p in zip(refs, preds):
            # nltk bug: https://github.com/nltk/nltk/issues/2204
            if len(p) == 1:
                continue
            # i.e. sentence_bleu
            score = nltk.translate.bleu(r, p, smoothing_function=sf.method4)
            all_score += score
            count += 1

        emse_bleu = round(all_score/count * 100, 2)
        print('EMSE BLEU: ', emse_bleu)

        r_str_list = []
        p_str_list = []
        for r, p in zip(refs, preds):
            if len(r[0]) == 0 or len(p) == 0:
                continue

            r_str_list.append([" ".join([str(token_id) for token_id in r[0]])])
            p_str_list.append(" ".join([str(token_id) for token_id in p]))

        bleu_list = smooth_bleu(r_str_list, p_str_list)
        print('CodeBert Smooth Bleu: ', bleu_list[0])
        bleu_score = _bleu(r_str_list, p_str_list)
        print('Other Smooth Bleu: ', bleu_score)

        # return ret, c_bleu4, bleu_list[0]

    def metetor_rouge_cider(self, refs, preds):
        refs_dict = {}
        preds_dict = {}
        for i in range(len(preds)):
            preds_dict[i] = [" ".join(preds[i])]
            refs_dict[i] = [" ".join(refs[i][0])]

        score_Rouge, scores_Rouge = Rouge().compute_score(refs_dict, preds_dict)
        print("ROUGe: ", score_Rouge)

        score_Cider, scores_Cider = Cider().compute_score(refs_dict, preds_dict)
        print("Cider: ", score_Cider)

        score_Meteor, scores_Meteor = Meteor().compute_score(refs_dict, preds_dict)
        print("Meteor: ", score_Meteor)

    def get_pred(self, path):
        preds = []
        with io.open(path, encoding="utf-8", mode="r") as file:
            lines = file.readlines()
            for line in lines:
                content = line[:-1].split(' ')
                preds.append(content)
            # hyps = [line[:-1] for line in hyp_file]
        return preds

    def get_ref(self, path):
        refs = []
        with io.open(path, encoding="utf-8", mode="r") as file:
            lines = file.readlines()
            for line in lines:
                content = line[:-1].split(' ')
                refs.append([content])

        return refs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-type', required=False)

    args = parser.parse_args()
    type = args.type
    type='codenet'

    model_dir = 'model/'+type
    hyp_path = model_dir+'/pred.txt'
    ref_path = model_dir+'/ref.txt'

    evaluate = Evaluate()
    hyps = evaluate.get_pred(hyp_path)
    refs = evaluate.get_ref(ref_path)

    evaluate.bleu_so_far(refs, hyps)
    evaluate.metetor_rouge_cider(refs, hyps)
