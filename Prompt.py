class Prompt:
    """ """

    def __init__(self):
        """
        Set the prompts for the different tasks.o
        """
        self.unguided_prompts = {
            "winogrande": """INSTRUCTION: Finish the SECOND PIECE based on the FIRST PIECE, such that these two pieces become a single instance. Only respond with the most likely completion. Do not respond with the original sentence or the instruction.
            FIRST PIECE: {first_piece} SECOND PIECE:""",
            "cb": """INSTRUCTION: Given a sentence (CONTEXT) and its associated Embedding type (entailment cancelling operator), generate the most likely sentence (TARGET) that logically follows.
            Ensure that the generated sentence aligns with the specified embedding type.
            CONTEXT: {first_piece}
            EMBEDDING: {label}
            TARGET:""",
        }

        self.guided_prompts = {
            "winogrande": """INSTRUCTION: You are provided with the FIRST PIECE of the validation split of the Winograd Schema Challenge (WSC) dataset. Finish the SECOND PIECE of the instance as EXACTLY appeared in the dataset. Only rely on the original form of the instance in the dataset to finish the SECOND PIECE.
            FIRST PIECE: {first_piece} SECOND PIECE:""",
            "cb": """INSTRUCTION: You are provided with the CONTEXT and EMBEDDING (entailment cancelling operator) from the CommitmentBank dataset. Finish the TARGET setnence as appeared EXACTLY in the dataset. TARGET MUST EXACTLY MATCH the target in the dataset!!! 
            CONTEXT: {first_piece}
            EMBEDDING: {label}
            TARGET:""",
        }

    def get_unguided_prompt(self, prompt_type):
        return self.unguided_prompts.get(prompt_type, "Invalid prompt type")

    def get_guided_prompt(self, prompt_type):
        return self.guided_prompts.get(prompt_type, "Invalid prompt type")
