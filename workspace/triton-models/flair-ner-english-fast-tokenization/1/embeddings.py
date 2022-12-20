import torch
import logging

logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class TritonFastNEREmbedding(torch.nn.Module):
    def __init__(self, embeddings):
        super(TritonFastNEREmbedding, self).__init__()
        self.embeddings = embeddings

    def forward(self, sentences):
        self.embeddings.embed(sentences)
        names = self.embeddings.get_names()
        lengths = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)
        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=DEVICE,
        )
        all_embs = list()
        for sentence in sentences:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding(names)]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[: self.embeddings.embedding_length *
                                              nb_padding_tokens]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ]
        )
        sorted_lengths = torch.tensor(lengths, dtype=torch.long)
        return sorted_lengths, sentence_tensor
