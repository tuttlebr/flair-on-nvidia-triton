from flair.data import Sentence
import triton_python_backend_utils as pb_utils
import logging
import json
import torch
import numpy as np

from viterbi_decoder import TritonFastNERViterbi, DEVICE

logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def string_to_bytes(string, encoding="utf-8"):
    return np.array(list(bytes(str(string), encoding)))


def bytes_to_string(byte_list):
    return bytes(byte_list).decode()


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # Parse model_config
        self.model_config = json.loads(args["model_config"])

        # Get output configs
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "tagged_sentences"
        )

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

        # Load the flair model
        self.viterbi_decoder = torch.load(
            "/models/flair-ner-english-fast-viterbi-decoder/1/viterbi_decoder.bin")

        self.transitions = torch.load(
            "/models/flair-ner-english-fast-viterbi-decoder/1/crf_transitions.bin").detach()

        self.model = TritonFastNERViterbi(
            self.viterbi_decoder, self.transitions)

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        responses = []

        for request in requests:
            # Get input
            sentence_bytes = pb_utils.get_input_tensor_by_name(
                request, "sentence_bytes").as_numpy().tolist()
            sentence_string = bytes_to_string(sentence_bytes)

            sentences = [Sentence(sentence_string)]

            features = torch.tensor(pb_utils.get_input_tensor_by_name(
                request, "features").as_numpy()).to(DEVICE)

            sorted_lengths = torch.tensor(pb_utils.get_input_tensor_by_name(
                request, "sorted_lengths").as_numpy()).to(DEVICE)

            tagged_sentences_dict = self.model.forward(
                sentences, features, sorted_lengths)

            tagged_sentences = string_to_bytes(tagged_sentences_dict)

            out_tensor_0 = pb_utils.Tensor(
                "tagged_sentences",
                tagged_sentences.astype(output0_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    out_tensor_0,
                ]
            )
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("TritonFastNERViterbi cleaning up...")
