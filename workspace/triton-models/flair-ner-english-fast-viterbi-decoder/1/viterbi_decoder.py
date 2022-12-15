import torch
from flair.models.sequence_tagger_utils.bioes import get_spans_from_bio
import logging

logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class TritonFastNERViterbi(torch.nn.Module):
    def __init__(self, viterbi_decoder):
        super(TritonFastNERViterbi, self).__init__()
        self.viterbi_decoder = viterbi_decoder

    def forward(self, sentences, features, sorted_lengths, transitions):

        embedding = (features, sorted_lengths, transitions)

        predictions, all_tags = self.viterbi_decoder.decode(
            embedding, True, sentences)

        for sentence, sentence_predictions in zip(sentences, predictions):
            sentence_tags = [label[0] for label in sentence_predictions]
            sentence_scores = [label[1] for label in sentence_predictions]
            predicted_spans = get_spans_from_bio(
                sentence_tags, sentence_scores)
            for predicted_span in predicted_spans:
                span = sentence[predicted_span[0]
                                [0]: predicted_span[0][-1] + 1]
                span.add_label(
                    "ner", value=predicted_span[2], score=predicted_span[1])

        dict_format = {}
        sentence_list = []
        for entity in sentences[0].get_spans("ner"):
            sentence_list.append(
                {
                    "entity_group": entity.tag,
                    "start": entity.start_position,
                    "word": entity.text,
                    "end": entity.end_position,
                    "score": int(entity.score * 100),
                }
            )
            dict_format[sentence.text] = sentence_list

        return dict_format
