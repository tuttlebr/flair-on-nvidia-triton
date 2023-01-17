from flair.data import Sentence
from flair.models import SequenceTagger
import triton_python_backend_utils as pb_utils
import logging
import json
import numpy as np

logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


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
            self.model_config, "OUTPUT_0"
        )

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

        # Load the flair model
        self.model = SequenceTagger.load(
            "/models/ner-english-fast/1/model.bin")

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
            sentences = pb_utils.get_input_tensor_by_name(
                request, "INPUT_0").as_numpy().astype(object).squeeze()

            sentences = [Sentence(str(sentence)[2:]) for sentence in sentences]

            self.model.predict(sentences)

            tagged_sentences_dict = {}
            for sentence in sentences:
                sentence_list = []
                for entity in sentence.get_spans("ner"):
                    sentence_list.append(
                        {
                            "entity_group": entity.tag,
                            "start": entity.start_position,
                            "word": entity.text,
                            "end": entity.end_position,
                            "score": int(entity.score * 100),
                        }
                    )
                tagged_sentences_dict[sentence.text] = sentence_list

            out_tensor_0 = pb_utils.Tensor(
                "OUTPUT_0", np.array(tagged_sentences_dict).astype(object))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    out_tensor_0,
                ]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Triton Flair NER English Fast model cleaning up...")
