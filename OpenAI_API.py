pip install --upgrade openai

# Import the 'openai' library for interacting with the OpenAI GPT API.
from openai import OpenAI

# Replace 'YOUR_API_KEY' with your actual OpenAI GPT API key.
api_key_openai = "Your APIKey here "

# Set the OpenAI GPT model you want to use (e.g., 'text-davinci-002').
gpt_model = "text-davinci-002" # one of the following ["text-davinci-002", "curie", "babbage", "gpt-3.5-turbo"]

# Specify the number of tokens to generate (maximum 2048 tokens for 'text-davinci-002').
max_tokens = 100

# Initialize the OpenAI API client with your API key.
client = OpenAI(api_key=api_key_openai)

# Set the text prompt to generate content.
prompt_story = "Once upon a time,"

# Make a request to the OpenAI GPT API to generate text based on the prompt.
response = client.completions.create(
  model="gpt-3.5-turbo-instruct",  # Or another model name
  prompt = prompt_story,
  max_tokens = max_tokens
  #max_tokens = max_tokens
)

# Retrieve and print the generated text.
print(response.choices[0].text.strip())
