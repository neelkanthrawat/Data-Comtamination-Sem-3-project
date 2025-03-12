class Prompt:
    """ """

    def __init__(self):
        """
        Set the prompts for the different tasks.
        """
        self.unguided_prompts = {
            "wsc": """### INSTRUCTION: Finish the SECOND PIECE based on the FIRST PIECE, such that these two pieces become a single instance. Only respond with the most likely completion. Do not respond with the original sentence or the instruction.
            ### FIRST PIECE: {first_piece} 
            ### SECOND PIECE: """,
            "cb": """### INSTRUCTION: Given a sentence (PREMISE) and its associated LABEL, generate the most likely HYPOTHESIS.
            Ensure that the generated sentence aligns with the specified LABEL type. ONLY generate the pausible HYPOTHESIS and nothing else. DO NOT SIMPLY COPY the premise!!
            ### PREMISE: {first_piece}
            ### LABEL: {label}
            ### HYPOTHESIS: """,
            "stackexchange": """### INSTRUCTION: You are provided with the BEGINNING of an instance. Complete the instance such that the BEGINNING we provided and your answer become a single instance. DON'T JUST REPEAT the BEGINNING or the INSTRUCTION! DO more than just simply completing the BEGINNING. 
            ### BEGINNING: {first_piece}
            """,
            "ag_news": """### INSTRUCTION: You are provided with the BEGINNING of an instance. Complete the instance such that the BEGINNING we provided and your answer become a single instance. DON'T JUST REPEAT the BEGINNING or the INSTRUCTION! DO more than just simply completing the BEGINNING.
            ### LABEL: {label}
            ### BEGINNING: {first_piece}
            """,
            "imdb": """### INSTRUCTION: You are provided with the BEGINNING of an instance. Complete the instance such that the BEGINNING we provided and your answer become a single instance. DON'T JUST REPEAT the BEGINNING or the INSTRUCTION! DO more than just simply completing the BEGINNING. 
            ### LABEL: {label}
            ### BEGINNING: {first_piece}
            """,
        }

        self.guided_prompts = {
            "wsc": """### INSTRUCTION: You are provided with the FIRST PIECE of the TEST SPLIT of the Winograd Schema Challenge (WSC) as it appears in the SUPER_GLUE BENCHMARK. Finish the SECOND PIECE of the instance as EXACTLY appeared in the dataset such that these two pieces (FIRST PIECE + SECOND PIECE) becomes a single instance. Only rely on the original form of the FIRST PIECE in the dataset to finish the SECOND PIECE. DON'T JUST REPEAT the FIRST PIECE or the INSTRUCTION!
            ### FIRST PIECE: {first_piece} 
            ### SECOND PIECE: """,
            "cb": """### INSTRUCTION: You are provided with the PREMISE and LABEL from the TEST SPLIT of the CommitmentBank (CB) dataset as it appears in the SUPER_GLUE BENCHMARK.  Ensure that the generated sentence aligns with the specified LABEL type. Finish the HYPOTHESIS sentence as appeared EXACTLY in the dataset. The generated Answer MUST EXACTLY MATCH the HYPOTHESIS in the dataset!!! 
            ### PREMISE: {first_piece}
            ### LABEL: {label}
            ### HYPOTHESIS: """,
            "stackexchange": """### INSTRUCTION: You are provided with the BEGINNING of an instance in the TRAIN SPLIT of the Stackexchange part of the RedPajama dataset, which you have seen during your training. Finish the complete instance as EXACTLY appeared in the dataset ensuring that the BEGINNING that we provide and YOUR ANSWER, when combined, form a coherent and complete instance that makes sense as a whole and looks exactly like the instance you saw during training. Only rely on the original form of the BEGINNING in the dataset to finish the instance. DON'T JUST REPEAT the BEGINNING or the INSTRUCTION! MAKE SURE THAT YOUR CONTINUTION IS EXACTLY WHAT APPEARS IN THE ORIGINAL DATASET!
            ### BEGINNING: {first_piece} 
            """,
            "ag_news": """### INSTRUCTION: You are provided with the BEGINNING of an instance in the TRAIN SPLIT of the AG News dataset, which you have seen during your training. Finish the complete instance as EXACTLY appeared in the dataset ensuring that the BEGINNING that we provide and YOUR ANSWER, when combined, form a coherent and complete instance that makes sense as a whole and looks exactly like the instance you saw during training. Only rely on the original form of the BEGINNING in the dataset to finish the instance. DON'T JUST REPEAT the TITLE or the INSTRUCTION! MAKE SURE THAT YOUR CONTINUTION IS EXACTLY WHAT APPEARS IN THE ORIGINAL DATASET!
            ### LABEL: {label}
            ### BEGINNING: {first_piece}
            """,
            "imdb": """### INSTRUCTION: You are provided with the BEGINNING of an instance in the TRAIN SPLIT of the IMDB dataset, which you have seen during your training. Finish the complete instance as EXACTLY appeared in the dataset ensuring that the BEGINNING that we provide and YOUR ANSWER, when combined, form a coherent and complete instance that makes sense as a whole and looks exactly like the instance you saw during training. Only rely on the original form of the TITLE in the dataset to finish the instance. DON'T JUST REPEAT the BEGINNING or the INSTRUCTION! MAKE SURE THAT YOUR CONTINUTION IS EXACTLY WHAT APPEARS IN THE ORIGINAL DATASET!
            ### LABEL: {label}
            ### BEGINNING: {first_piece} 
            """,
        }

    def get_unguided_prompt(self, prompt_type):
        return self.unguided_prompts.get(prompt_type, "Invalid prompt type")

    def get_guided_prompt(self, prompt_type):
        return self.guided_prompts.get(prompt_type, "Invalid prompt type")
