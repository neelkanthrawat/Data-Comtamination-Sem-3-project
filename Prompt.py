class Prompt:
    """ """

    def init(self):
        """
        Set the prompts for the different tasks.
        """

        self.unguided_prompts = {
            "winogrande": """INSTRUCTION:
            Finish the SECOND PIECE based on the FIRST PIECE, such that these two pieces become a single instance.

            FIRST PIECE:
            {first_piece}

            SECOND PIECE:
            """,
            "cb": """INSTRUCTION:
            Finish the TARGET based on the CONTEXT, such that the following EMBEDDING shows the logical relationship between TARGET and CONTEXT.

            TARGET:
            {first_piece}

            EMBEDDING: {label}

            SENTENCE 2:
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
            Finish the TARGET based on the CONTEXT, such that the following EMBEDDING shows the logical relationship between TARGET and CONTEXT.

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
