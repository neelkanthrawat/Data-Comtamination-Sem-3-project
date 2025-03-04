class Prompt:
    """ """

    def __init__(self):
        """
        Set the prompts for the different tasks.o
        """
        self.unguided_prompts = {
            "wsc": 
            """
            Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            ### INSTRUCTION: 
            Finish the SECOND PIECE based on the FIRST PIECE, such that these two pieces become a single instance. Only respond with the most likely completion. Do not respond with the original sentence or the instruction.
            ### FIRST PIECE: {first_piece} 
            ### SECOND PIECE:
            """,
            "cb": """Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            INSTRUCTION: Given a sentence (PREMISE) and its associated LABEL, generate the most likely HYPOTHESIS.
            Ensure that the generated sentence aligns with the specified LABEL type. Only generate the pausible HYPOTHESIS and nothing else.
            PREMISE: {first_piece}
            LABEL: {label}
            HYPOTHESIS:""",
            "stackexchange": """Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            ### INSTRUCTION: 
            Finish the SECOND PIECE based on the FIRST PIECE, such that these two pieces become a single instance. Only respond with the most likely completion. Do not respond with the original sentence or the instruction.
            ### FIRST PIECE: {first_piece} 
            ### SECOND PIECE:"""
        }

        self.guided_prompts = {
            "wsc": """
            Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            ### INSTRUCTION: You are provided with the FIRST PIECE of the TEST SPLIT of the Winograd Schema Challenge (WSC) as it appears in the SUPER_GLUE BENCHMARK. Finish the SECOND PIECE of the instance as EXACTLY appeared in the dataset such that these two pieces (FIRST PIECE + SECOND PIECE) becomes a single instance. Only rely on the original form of the FIRST PIECE in the dataset to finish the SECOND PIECE. DON'T JUST REPEAT the FIRST PIECE or the INSTRUCTION!
            ### FIRST PIECE: {first_piece} 
            ### SECOND PIECE: """,
            "cb":  """Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            INSTRUCTION: You are provided with the PREMISE and LABEL from the TEST SPLIT of the CommitmentBank (CB) dataset as it appears in the SUPER_GLUE BENCHMARK.  Ensure that the generated sentence aligns with the specified LABEL type. Finish the HYPOTHESIS sentence as appeared EXACTLY in the dataset. The generated Answer MUST EXACTLY MATCH the HYPOTHESIS in the dataset!!! 
            PREMISE: {first_piece}
            LABEL: {label}
            HYPOTHESIS:""",
            "stackexchange": """
            Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            ### INSTRUCTION: You are provided with the FIRST PIECE of the TRAIN SPLIT of the Stackexchange part as it appears in the SUPER_GLUE BENCHMARK. Finish the SECOND PIECE of the instance as EXACTLY appeared in the dataset such that these two pieces (FIRST PIECE + SECOND PIECE) becomes a single instance. Only rely on the original form of the FIRST PIECE in the dataset to finish the SECOND PIECE. DON'T JUST REPEAT the FIRST PIECE or the INSTRUCTION!
            ### FIRST PIECE: {first_piece} 
            ### SECOND PIECE: """,
        }

    def get_unguided_prompt(self, prompt_type):
        return self.unguided_prompts.get(prompt_type, "Invalid prompt type")

    def get_guided_prompt(self, prompt_type):
        return self.guided_prompts.get(prompt_type, "Invalid prompt type")
