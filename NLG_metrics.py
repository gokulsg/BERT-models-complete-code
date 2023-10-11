import evaluate

def NLG_metrics(predictions, references):
    ## metrics ##
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore") 

    results_bleu = bleu.compute(predictions=predictions, references=references)
    results_rouge = rouge.compute(predictions=predictions, references=references)
    results_meteor = meteor.compute(predictions=predictions, references=references)
    results_bertscore = bertscore.compute(predictions=predictions, references=references, lang="en")

    return {"bleu":results_bleu, "rouge":results_rouge, "meteor":results_meteor, "bertscore": results_bertscore}

## Example ##
predictions = ["The cat is on the mat", "I am currently out of the office"]
references = ["There is a cat on the mat", "I am currently not in the office"]

results = NLG_metrics(predictions, references)
print(results)
