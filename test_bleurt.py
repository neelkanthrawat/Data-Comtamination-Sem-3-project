import evaluate


def main():

    predictions = ["hello there", "general kenobi"]
    references = ["hello there", "general kenobi"]

    bleurt = evaluate.load("bleurt", module_type="metric")
    # rouge = evaluate.load("rouge")

    bleurt_score = bleurt.compute(predictions=predictions, references=references)
    print(f"BLEURT score: {bleurt_score}")

    # rouge_score = rouge.compute(predictions=predictions, references=references)
    # print(f"ROUGE score: {rouge_score}")


if __name__ == "__main__":
    main()
