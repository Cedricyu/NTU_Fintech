import json

# Load data from JSON files
with open('ground_truths_example.json', 'r', encoding='utf-8') as gt_file:
    ground_truths_data = json.load(gt_file)

with open('pred_retrieve.json', 'r', encoding='utf-8') as pred_file:
    pred_retrieve_data = json.load(pred_file)

# Convert ground truths to dictionary for easier access
ground_truths = {item['qid']: item['retrieve'] for item in ground_truths_data['ground_truths']}
answers = {item['qid']: item['retrieve'] for item in pred_retrieve_data['answers']}

# Calculate scores
scores = {}
total_score = 0

for qid, ground_truth in ground_truths.items():
    if qid in answers:
        retrieved_value = answers[qid]
        if retrieved_value == ground_truth:
            scores[qid] = 1  # Award 1 point for a correct match
            total_score += 1
        else:
            scores[qid] = 0  # No points for incorrect match
    else:
        scores[qid] = 0  # No points if qid is not present in answers

# Output results
print("Scores:", scores)
print("Total Score:", total_score)

# Calculate and print accuracy
accuracy = total_score / len(ground_truths) if ground_truths else 0  # Avoid division by zero
print("Accuracy: {:.2f}%".format(accuracy * 100))  # Format as percentage
