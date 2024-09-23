from datasets import load_dataset
import pickle
from tqdm import tqdm
import sys
import re
import gc
from llama import Dialog, Llama


def correctAnswerPattern(results):
    if len(results) == 0:
        return False
    for result in results:
        answ = result["generation"]["content"]
        match = re.search(ANSWER_PATTERN_MULTICHOICE, answ)
        if not match:
            return False
    return True

#Steps:
#1. Iterate through all questions in the dataset
#2. For each question, save the embedding
#3. Answer each question 5 times, with temperate of 1
#4. If all answers are correct, set known for this question to yes. 
#5. For each question, save a dict with the following fields: question, knwon, embedding, 

mytopics= ["clinical_knowledge","medical_genetics"]
# mytopics= ["medical_genetics"]
# mytopics = ['high_school_european_history', 'business_ethics','high_school_us_history', 'high_school_physics', 'high_school_world_history', 'virology', 'high_school_microeconomics', 'econometrics', 'college_computer_science', 'high_school_biology', 'abstract_algebra', 'professional_accounting', 'philosophy', 'professional_medicine', 'nutrition', 'global_facts', 'machine_learning', 'security_studies', 'public_relations', 'professional_psychology', 'prehistory', 'anatomy', 'human_sexuality', 'college_medicine', 'high_school_government_and_politics', 'college_chemistry', 'logical_fallacies', 'high_school_geography', 'elementary_mathematics', 'human_aging', 'college_mathematics', 'high_school_psychology', 'formal_logic', 'high_school_statistics', 'international_law', 'high_school_mathematics', 'high_school_computer_science', 'conceptual_physics', 'miscellaneous', 'high_school_chemistry', 'marketing', 'professional_law', 'management', 'college_physics', 'jurisprudence', 'world_religions', 'sociology', 'us_foreign_policy', 'high_school_macroeconomics', 'computer_security', 'moral_scenarios', 'moral_disputes', 'electrical_engineering', 'astronomy', 'college_biology']

alltopics = ['high_school_european_history', 'business_ethics','high_school_us_history', "clinical_knowledge","medical_genetics", 'high_school_physics', 'high_school_world_history', 'virology', 'high_school_microeconomics', 'econometrics', 'college_computer_science', 'high_school_biology', 'abstract_algebra', 'professional_accounting', 'philosophy', 'professional_medicine', 'nutrition', 'global_facts', 'machine_learning', 'security_studies', 'public_relations', 'professional_psychology', 'prehistory', 'anatomy', 'human_sexuality', 'college_medicine', 'high_school_government_and_politics', 'college_chemistry', 'logical_fallacies', 'high_school_geography', 'elementary_mathematics', 'human_aging', 'college_mathematics', 'high_school_psychology', 'formal_logic', 'high_school_statistics', 'international_law', 'high_school_mathematics', 'high_school_computer_science', 'conceptual_physics', 'miscellaneous', 'high_school_chemistry', 'marketing', 'professional_law', 'management', 'college_physics', 'jurisprudence', 'world_religions', 'sociology', 'us_foreign_policy', 'high_school_macroeconomics', 'computer_security', 'moral_scenarios', 'moral_disputes', 'electrical_engineering', 'astronomy', 'college_biology']

mytopics = alltopics 

ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"
nqs = 4

ckpt_dir="Meta-Llama-3-8B-Instruct"
tokenizer_path="Meta-Llama-3-8B-Instruct/tokenizer.model"
max_seq_len=1024
generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=nqs,
)
print("Model loaded")

n_skipped=  0 
for topic in mytopics:
    outdicts = []
    # Load the MMLU dataset
    dataset = load_dataset("lukaemon/mmlu", topic, trust_remote_code=True)
    testset = dataset['test']
    print(f"Loaded {topic} dataset with {len(testset)} questions.")

    # sys.exit()

    # n_data = 25
    for i,elem in tqdm(enumerate(testset)):
        out = {}
        completequestion = f"{elem['input']}\n A) {elem['A']} B) {elem['B']} C) {elem['C']} D) {elem['D']}"
        emb = generator.getSentenceEmbeddings([[{f"role": "user", "content": completequestion}]])
        mc_question = f"""
        Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

        {elem['input']}

        A) {elem["A"]}
        B) {elem["B"]}
        C) {elem["C"]}
        D) {elem["D"]}
        """.strip()
        if len(mc_question) > max_seq_len:
            print(f"Skipping question {i} because it is too long")
            continue

        qbatch = [[{f"role": "user", "content": mc_question}] for _ in range(nqs)]

        goodPattern = False
        for j in range(5):
            results = generator.chat_completion(
                qbatch,
                max_gen_len=2048,
                temperature=0.6,
                top_p=0.9,
                logprobs=True
            )
            if correctAnswerPattern(results):
                goodPattern = True
                break
            else:
                print(f"Retrying question {i} for the {j+1}th time because of wrong answer patterns")
        if not goodPattern:
            print("Skipping question because of wrong answer patterns")
            n_skipped += 1
            continue

        answers = []
        known = True
        for j, result in enumerate(results):
            # print(result["generation"]["content"])       
            # print("\n\n ############################## \n\n") 
            answ = result["generation"]["content"]
            match = re.search(ANSWER_PATTERN_MULTICHOICE, answ)
            extracted_answer = match.group(1) if match else None
            known = False if extracted_answer != elem['target'] else known
            answers.append(answ)
        firstres = results[0]
        out = {"question": completequestion, 
               "mc_question": mc_question, 
               "known":known, 
               "embedding":emb, 
               "answers":answers, 
               "correctAnswer":elem['target'],
               "n_intokens":firstres['n_intokens'],
               "n_outtokens":firstres['n_outtokens'],
               "top_logprobs":firstres['top_logprobs'],
               "probs":firstres['probs'],
               }
        outdicts.append(out)

        #comment this out to run the whole dataset
        # if i >= n_data:
            # break
    print(f"Skipped {n_skipped} questions overall")
    #save outdicts using pickle
    # print(f"saving to datasets/{topic}_test.pkl")
    # with open(f'datasets/{topic}_test.pkl', 'wb') as f:
    #     pickle.dump(outdicts, f)





    #6. Use the embeddings as input and known as label to train a model. 
    #7. Test this model on some test data. 
    #elems in dict: question, known, embedding, givenAnswer, correctAnswer


    #content missing in the dict: in_tokens, out_tokens, top k token probs!