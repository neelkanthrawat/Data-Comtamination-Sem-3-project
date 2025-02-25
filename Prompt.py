class Prompt:
    """ """

    def init(self):
        """
        Set the prompts for the different tasks.
        """

        self.prompts = {
            "winogrande": """INSTRUCTION:
            Finish the SECOND PIECE based on the FIRST PIECE, such that these two pieces become a single instance with the following LABEL.

            LABEL: {label}

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

    def get_prompt(self, prompt_type):
        return self.prompts.get(prompt_type, "Invalid prompt type")
