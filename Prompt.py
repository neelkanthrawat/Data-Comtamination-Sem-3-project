class Prompt:
    """ """

    def __init__(self):
        """
        Set the prompts for the different tasks.o
        """
        self.unguided_prompts = {
            "winogrande": """INSTRUCTION:
            Finish the SECOND PIECE based on the FIRST PIECE, such that these two pieces become a single instance.

            FIRST PIECE: {first_piece}

            SECOND PIECE:
            """,
            "cb": """INSTRUCTION:
            Given a sentence (CONTEXT) and its associated Embedding type (entailment cancelling operator), generate the most likely sentence (TARGET) that logically follows.
            Ensure that the generated sentence aligns with the specified embedding type.

            EMBEDDING: 
            {label}

            CONTEXT: 
            {first_piece}

            TARGET:
           
            AN EXAMPLE:
            EMBEDDING: 
            modal

            CONTEXT:
            Doug emerges from the cabin looking blue with cold. He smiles weakly at everyone.

            TARGET:
            Although he clearly wants to walk off the boat the ambulance crew scramble on board and clamp him onto a stretcher so firmly that I wonder if they think he might punch them.
            """,
        }

        self.guided_prompts = {
            "winogrande": """INSTRUCTION:
            You are provided with the FIRST PIECE of a summary from the {split_name} split of the {dataset_name} dataset.
            Finish the SECOND PIECE of the summary as EXACTLY appeared in the dataset.
            ONLY rely on the original form of the summary in the dataset to finish the SECOND PIECE.

            FIRST PIECE:
            {first_piece}

            SECOND PIECE:
            """,
            "cb": """INSTRUCTION:
            Finish the TARGET based on the CONTEXT, such that the fllowing EMBEDDING shows the logical relationship between TARGET and CONTEXT.

            TARGET:
            {first_piece}

            EMBEDDING: {label}

            SENTENCE 2:
            """,
        }

    def get_unguided_prompt(self, prompt_type):
        return self.unguided_prompts.get(prompt_type, "Invalid prompt type")

    def get_guided_prompt(self, prompt_type):
        return self.guided_prompts.get(prompt_type, "Invalid prompt type")
