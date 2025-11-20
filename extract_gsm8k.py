from datasets import load_dataset
import os

def main():
    try:
        dataset = load_dataset("gsm8k", "main", split="test")
        questions = dataset[:20]['question']
    except Exception as e:
        print(f"Error loading gsm8k: {e}")
        print("Using dummy questions.")
        questions = [
            "Janet has 5 apples. She buys 3 more. How many apples does she have?",
            "A train travels at 60 mph. How far does it go in 2.5 hours?"
        ] * 10

    with open("gsm8k_questions.txt", "w") as f:
        for q in questions:
            f.write(q.replace("\n", " ") + "\n")
    
    print("Saved gsm8k_questions.txt")

if __name__ == "__main__":
    main()

