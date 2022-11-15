import torch
from transformers import BertModel, BertTokenizer


class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.model = BertModel.from_pretrained(
            "bert-base-uncased",
            torchscript=True,
            output_attentions=True).to("cuda:0")
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", torchscript=True)

    def dummy_input(self):
        text_1 = "Who was Jim Henson ?"
        text_2 = "Jim Henson was a puppeteer"
        INPUT__0 = torch.ByteTensor(list(bytes(text_1, "utf8"))).to("cuda:0")
        INPUT__1 = torch.ByteTensor(list(bytes(text_2, "utf8"))).to("cuda:0")
        return INPUT__0, INPUT__1

    def forward(self, INPUT__0, INPUT__1):
        text_0 = bytes(INPUT__0).decode()
        tokenized_text_0 = self.tokenizer.encode(
            text=text_0, add_special_tokens=True, return_tensors="pt")
        segment_0 = torch.zeros(tokenized_text_0.shape, dtype=torch.int64)

        text_1 = bytes(INPUT__1).decode()
        tokenized_text_1 = self.tokenizer.encode(
            text=text_1, add_special_tokens=True, return_tensors="pt")
        segment_1 = torch.ones(tokenized_text_1.shape, dtype=torch.int64)

        tokens_tensor = torch.concat(
            (tokenized_text_0, tokenized_text_1), axis=1).to("cuda:0")
        segments_tensors = torch.concat(
            (segment_0, segment_1), axis=1).to("cuda:0")
        return self.model(
            tokens_tensor,
            token_type_ids=segments_tensors)[0].squeeze()


model = CustomModel().eval()
INPUT__0, INPUT__1 = model.dummy_input()
OUTPUT__0 = model.forward(INPUT__0, INPUT__1)
traced_model = torch.jit.trace(model, example_inputs=[INPUT__0, INPUT__1])
torch.jit.save(
    traced_model, "/workspace/triton-models/bert-base-uncased/1/model.pt")

print("saved TorchScript model as /workspace/triton-models/bert-base-uncased/1/model.pt")
