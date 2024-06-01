from datasets import load_dataset
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key'
import openai

def generate_response(prompt):
    response = openai.Completion.create(
        model="gpt-4o",
        prompt=prompt,
        max_tokens=50,
        temperature=0,
        top_p=0.9,
        n=1,
        stop=None
    )
    return response.choices[0].text.strip()

def main() :
    test_dataset = load_dataset("vfleaking/DirectHarm4")
    test_dataset = test_dataset["test"]
    test_dataset = [example for example in test_dataset if example["category"]=="disinfo"]

    results = []
    for data in test_dataset :
        prompt = data["instruction"]
        prompt = "Please answer the following questions in 15 to 30 words.\n" + prompt
        # gpt4 generate propmt
        generated_response = generate_response(prompt)
        print(generated_response)
        exit(0)
        results.append({
            "instruction": prompt,
            "response": generated_response,
        })
        
    save_dir = "../dataset/DirectHarm4/harmful_anchor.jsonl"
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    # with open(save_dir, 'w') as f:
        # for result in results:
        #     f.write(json.dumps(result) + '\n')    


if __name__ == "__main__" :
    main()