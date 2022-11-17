import torch
from transformers import BertTokenizer, BertModel


text_0 = "Who was Jim Henson?"
text_1 = "Jim Henson was a puppeteer"
save_as = "/workspace/triton-models/bert-base-uncased/1/model.pt"


class TritonBertModel(torch.nn.Module):
    def __init__(self):
        super(TritonBertModel, self).__init__()
        self.model = BertModel.from_pretrained(
            "bert-base-uncased",
            torchscript=True,
            output_attentions=True)

    def forward(self, tokens_tensor, segments_tensors):
        return self.model(tokens_tensor, segments_tensors)[0]


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", torchscript=True)
    tokenized_tensor_0 = tokenizer(
        text_0,
        add_special_tokens=True,
        return_tensors="pt")
    tokenized_tensor_1 = tokenizer(
        text_1,
        add_special_tokens=True,
        return_tensors="pt")
    tokens_tensor = torch.concat(
        (tokenized_tensor_0["input_ids"],
         tokenized_tensor_1["input_ids"]),
        axis=1)
    segments_tensors = torch.concat(
        (tokenized_tensor_0["token_type_ids"],
         tokenized_tensor_1["attention_mask"]),
        axis=1)
    model = TritonBertModel().eval()

    print("INPUT__0 dims: {}".format(tokens_tensor.shape))
    print("INPUT__0 dtype: {}\n".format(tokens_tensor.dtype))

    print("INPUT__1 dims: {}".format(segments_tensors.shape))
    print("INPUT__1 dtype: {}\n".format(segments_tensors.dtype))

    embedding_oputput = model.forward(tokens_tensor, segments_tensors)
    print("OUTPUT__0 dims: {}".format(embedding_oputput.shape))
    print("OUTPUT__0 dtype: {}\n".format(embedding_oputput.dtype))

    traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
    torch.jit.save(traced_model, save_as)
    print("saved TorchScript model as {}".format(save_as))
