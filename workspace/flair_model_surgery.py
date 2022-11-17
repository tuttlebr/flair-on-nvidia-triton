import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from flair.data import Sentence
from flair.models import SequenceTagger


string_list = [
    "With the belief that the PC one day would become a consumer device for enjoying games and multimedia, NVIDIA is founded by Jensen Huang, Chris Malachowsky and Curtis Priem."
]

save_as = "/workspace/triton-models/flair-ner-english-fast/1/model.pt"


class TritonFastNERTagger(torch.nn.Module):
    def __init__(self, tagger, viterbi_decoder):
        super(TritonFastNERTagger, self).__init__()
        self.tagger = tagger
        self.viterbi_decoder = viterbi_decoder

    def forward(self, sentences):
        self.tagger.embed(sentences)
        names = self.tagger.get_names()
        lengths = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)
        pre_allocated_zero_tensor = torch.zeros(
            self.tagger.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device="cuda:0",
        )
        all_embs = list()
        for sentence in sentences:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding(names)]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[: self.tagger.embedding_length *
                                              nb_padding_tokens]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.tagger.embedding_length,
            ]
        )
        sorted_lengths = torch.tensor(lengths, dtype=torch.long)
        return sorted_lengths, sentence_tensor


class TritonFastNERModel(torch.nn.Module):
    def __init__(self, model):
        super(TritonFastNERModel, self).__init__()
        self.model = model

    def forward(self, sorted_lengths, sentence_tensor):
        if self.model.use_dropout:
            sentence_tensor = self.model.dropout(sentence_tensor)
        if self.model.use_word_dropout:
            sentence_tensor = self.model.word_dropout(sentence_tensor)
        if self.model.use_locked_dropout:
            sentence_tensor = self.model.locked_dropout(sentence_tensor)
        if self.model.reproject_embeddings:
            sentence_tensor = self.model.embedding2nn(sentence_tensor)
        if self.model.use_rnn:
            packed = pack_padded_sequence(
                sentence_tensor,
                sorted_lengths,
                batch_first=True,
                enforce_sorted=False)
            rnn_output, hidden = self.model.rnn(packed)
            sentence_tensor, output_lengths = pad_packed_sequence(
                rnn_output, batch_first=True)

        if self.model.use_dropout:
            sentence_tensor = self.model.dropout(sentence_tensor)
        if self.model.use_locked_dropout:
            sentence_tensor = self.model.locked_dropout(sentence_tensor)

        # linear map to tag space
        features = self.model.linear(sentence_tensor)

        # Depending on whether we are using CRF or a linear layer, scores is either:
        # -- A tensor of shape (batch size, sequence length, tagset size, tagset size) for CRF
        # -- A tensor of shape (aggregated sequence length for all sentences in batch, tagset size) for linear layer
        if self.model.use_crf:
            features = self.model.crf(features)
        return features, sorted_lengths, self.model.crf.transitions


if __name__ == '__main__':

    model = SequenceTagger.load("flair/ner-english-fast")
    viterbi_decoder = model.viterbi_decoder
    tagger = model.embeddings
    tagger = TritonFastNERTagger(tagger, viterbi_decoder)
    sentences = [Sentence(string) for string in string_list]
    sorted_lengths, sentence_tensor = tagger.forward(sentences)

    print("INPUT__0 dims: {}".format(sorted_lengths.shape))
    print("INPUT__0 dtype: {}\n".format(sorted_lengths.dtype))

    print("INPUT__1 dims: {}".format(sentence_tensor.shape))
    print("INPUT__1 dtype: {}\n".format(sentence_tensor.dtype))

    model.__delattr__("embeddings")
    traced_model = TritonFastNERModel(model)
    features, sorted_lengths, transitions = traced_model.forward(
        sorted_lengths, sentence_tensor)

    print("OUTPUT__0 dims: {}".format(features.shape))
    print("OUTPUT__0 dtype: {}\n".format(features.dtype))

    print("OUTPUT__1 dims: {}".format(sorted_lengths.shape))
    print("OUTPUT__1 dtype: {}\n".format(sorted_lengths.dtype))

    print("OUTPUT__2 dims: {}".format(transitions.shape))
    print("OUTPUT__2 dtype: {}\n".format(transitions.dtype))

    traced_model = torch.jit.trace(
        traced_model, [
            sorted_lengths, sentence_tensor])
    torch.jit.save(traced_model, save_as)
    torch.save(tagger, "tagger.bin")
    print("saved TorchScript model as {}".format(save_as))
    print("Saved client tagger to tagger.bin")
