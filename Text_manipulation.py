
import json
import random
from random import seed
from random import randint
import os
import glob
import nltk
import string

os.listdir()

os.chdir(os.getcwd() + '/Downloads')

print(glob.glob('*.json'))

# https://www.freecodecamp.org/news/python-parse-json-how-to-read-a-json-file/

with open('v2_OpenEnded_mscoco_val2014_questions.json', 'r') as json_file:
     data = json.load(json_file)

len(data)
type(data)

data.keys()

[data[list(data.keys())[i]] for i in range(5)]

len(data['questions'])
type(data['questions'])
type(data['questions'][0])

########################### Letter level swapping


#Swap_character function accepts a string and swap to the left or right in a position.
# if position = None and adjacent = None, function swap the letter randomly in a random position

def swap_characters(input_word, position, adjacent):
    temp_word = list(input_word)
    l1 = len(input_word) - 1
    
    if position is not None and adjacent is not None:
        
        if adjacent == 'left' and 0 <= position < l1 - 1:
            temp_word[position - 1] , temp_word[position] = temp_word[position] , temp_word[position - 1]
        elif adjacent == 'right' and  0 <= position < l1 - 1:
            temp_word[position] , temp_word[position + 1] = temp_word[position + 1] , temp_word[position]

        return ''.join(temp_word)
    
    else:
        
        idx = randint(0, l1)
        addition = [-1,1][randint(0, 1)]
        
        if idx == l1 and addition == 1:
            temp_word = temp_word
        elif idx == 0 and addition == -1:
            temp_word = temp_word
        else:
            temp_word[idx] , temp_word[idx + addition] = temp_word[idx + addition] , temp_word[idx]
            
        return ''.join(temp_word)

swap_characters("hasanmoradkhani", None , None)



######################## Word level swapping
# swap_words swap one word randomly in a random position

nltk.download('punkt')

def swap_words(sentence):
    q_tokenized = nltk.word_tokenize(sentence)
    l1 = len(q_tokenized) - 1 
    
    idx1 = randint(0 , l1)
    idx2 = randint(0 , l1)
    q_tokenized[idx1] , q_tokenized[idx2] =  q_tokenized[idx2] , q_tokenized[idx1]
    return ' '.join(q_tokenized ,  )


swap_words('Jamshid va gholi be madreseh miravand')




####### inserting punctuations in random places

type(data['questions'])
type(data['questions'][0])

string.punctuation

def insert_punctuation(paragraph, num_insertions=3):
    # Define a list of common punctuation marks
    punctuation_marks = string.punctuation

    # Split the paragraph into words
    words = paragraph.split()

    # Insert punctuation marks at random positions
    for _ in range(num_insertions):
        position = random.randint(0, len(words) - 1)
        punctuation = random.choice(punctuation_marks)
        words.insert(position, punctuation)

    # Join the words back into a paragraph
    modified_paragraph = ' '.join(words)

    return modified_paragraph

# Example usage
original_paragraph = "This is a sample paragraph without any punctuation"
modified_paragraph = insert_punctuation(original_paragraph)
print("Original paragraph:", original_paragraph)
print("Modified paragraph:", modified_paragraph)


############# Using constructed functions on V2_question data #############

swap_words(q['question'])
insert_punctuation(q['question'])
swap_characters(q['question'] , position= None , adjacent=None)
swap_characters(q['question'] , position= 10 , adjacent='left')

small_data = data['questions'][0:100]
type(small_data)

i = 0
for q in small_data:
    q['swap_words'] = swap_words(q['question'])
    q['insert_punctuations'] = insert_punctuation(q['question'])
    q['swap_characters1'] = swap_characters(q['question'] , position= None , adjacent=None)
    q['swap_characters2'] = swap_characters(q['question'] , position= 10 , adjacent='left')
    small_data[i] = q
    i += 1

# Writing to sample.json
with open("sample.json", "w") as outfile:
    outfile.write(json_object)



############# Model evaluation #############

def calculate_accuracy(predictions, ground_truth):
    correct = 0
    total = len(predictions)

    for pred, gt in zip(predictions, ground_truth):
        if pred == gt:
            correct += 1

    accuracy = correct / total
    return accuracy


##############################################

torchrun --nproc_per_node 1 example_chat_completion.py --ckpt_dir /Users/mpir0002/Downloads/llama/llama-2-7b/ --tokenizer_path /Users/mpir0002/Downloads/llama/tokenizer.model --max_seq_len 512 --max_batch_size 6











