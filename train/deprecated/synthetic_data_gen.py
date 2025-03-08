import json
from openai import OpenAI
import re
import os
import random

# Set up your OpenAI API key
client = OpenAI(api_key='doesntmatter', base_url='http://localhost:1234/v1')
#openai.api_key = "YOUR_OPENAI_API_KEY"

#def generate_story(prompt):
#    response = client.create_completion(model="text-davinci-003", prompt=prompt, max_tokens=2048)
#    return response["choices"][0]["text"]

def query_ai(question):
    response = client.chat.completions.create(
        model="local-model",  # Or use your local-model if available
        messages=[
            {"role": "system", "content": "You are a helpful assistant. You are an expert at dissecting language and stories to break them down into their constituent pieces or ingredients. You are an expert at identifing people, places, concepts, ideas, history and mythology, etc in a body of text. You will follow the user's request precisely. You will never create json data with keys that are meaningless things like pronouns. all json keys will be SINGLE WORD KEYS in any thing you generate that contains json. all json you create should be valid, well-formed json. JSON keys should NEVER have abbreviations or contractions or apostrophes or any special character in them. You will NEVER EVER provide to the user anything other than a strict JSON representation, not even commentary to explain the JSON. If you encounter a situation where there is no data to provide for the keys in question, just provide an empty json array as the value (rather than null)."},
            {"role": "user", "content": f"{question}"}
            ],
        temperature=0.7,
    )
    return response.choices[0].message

def process_file(file_path):
#    with open("synthetic_data.json", "r") as file:
#        synthetic_data = json.load(file) if os.path.exists("synthetic_data.json") else []

    with open(file_path, "r") as file:
        text = file.read()

    # Split the text into paragraphs using regex pattern
    #@paragraphs = re.split('(?<=\.|\?) ([^\.\?!])\1', text)
    paragraphs = text.split('\n')
    
    for paragraph in paragraphs:
        if paragraph != "" or paragraph.strip('\n') != "":  # Skip empty lines
            print("\n\nHit the code i wanted to hit")
            print("currently working on: ", paragraph)
            input_ = f"Given the following input: `{paragraph}`, please provide me with the key concepts, people, ideas, places, history, 'factual information', etc, from the text. Also, make sure any value you use for a json key is an appropriate json key, eg hyphenated or single string with now spaces. Please provide me with a JSON representation of the breakdown you generate, and respond only with the json object, NOTHING ELSE, not even the tickmarks around the json object. It's extraordinarily important that you ensure that EVERY json key in your struct has no more than ONE word as the key. Make sure you also provide a 'summary' of the input text, too as one of your json keys and values. None of your json values should be booleans. All json should at least include the keys 'summary', 'concepts', 'people', 'places', 'ideas', and 'facts'. All of these types are arrays of strings, except 'summary', which is a string. The JSON you create will NEVER have any keys other than those specified, and you should fit all classifications and analysis into those json keys as appropriate. ."
            output = query_ai(input_).content
            output = output.strip('```json')
            output = output.strip('```')
            output = output.strip('`')
            output = output.strip(' ')
            output = output.strip('\n')
            synthetic_data = output

            with open("synthetic_data3.json", "a") as fi:
                #data = json.dumps(synthetic_data).removeprefix('\\n')
                # js = json.loads(data)
                print("synth data was: ", synthetic_data)
                writejs = {
                    "sentence": f"{paragraph}",
                    "labels": json.loads(synthetic_data)
                }
                fi.write(json.dumps(writejs) + ',\n')
        else:
            print("\n\nhit the code i DIDNT want to hit.")

if __name__ == "__main__":
    file_path = "data/taa.txt"
    process_file(file_path)
