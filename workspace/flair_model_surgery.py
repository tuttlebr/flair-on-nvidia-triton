import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

from flair.data import Sentence
from flair.models import SequenceTagger
import torch


class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.model = SequenceTagger.load("flair/ner-english-fast")

    def dummy_input(self):
        INPUT__0 = torch.ByteTensor(
            list(
                bytes(
                    "NVIDIA was founded by Jensen Huang, Chris Malachowsky, and Curtis Priem in 1993.",
                    "utf8",
                )))
        return INPUT__0

    def forward(self, INPUT__0):
        string = bytes(INPUT__0).decode()
        sentence = Sentence(string)
        self.model.predict(sentence)
        return torch.ByteTensor(list(bytes(str(sentence), "utf8")))


custom_model = CustomModel().eval()
INPUT__0 = custom_model.dummy_input()
OUTPUT__0 = custom_model.forward(INPUT__0)

traced_model = torch.jit.trace(func=custom_model, example_inputs=INPUT__0)

traced_model.save("/workspace/triton-models/flair-ner-english-fast/1/model.pt")

print("saved TorchScript model as /workspace/triton-models/flair-ner-english-fast/1/model.pt")
