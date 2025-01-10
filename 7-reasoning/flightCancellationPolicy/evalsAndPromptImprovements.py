# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: openai
#     language: python
#     name: python3
# ---

# +

# ## Evals and Metaprompting

# ### Initial Setup
# We'll start with importing a few dependencies and setting up some visuals

# +


get_ipython().run_line_magic('pip', 'install openai')
get_ipython().run_line_magic('pip', 'install pandas')
get_ipython().run_line_magic('pip', 'install ipython')


# +


from openai import OpenAI
from IPython.display import display, Markdown
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functionDefinitions import TOOLS
import csv
import json
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
client = OpenAI()
MODEL = 'o1-mini'

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# ## Generating a Routine
# We'll take the Flight Cancellation Policy that we have created and convert it to an LLM-based routine with the following prompt

# +


with open('originalPolicy/flightCancellationsPolicy.md', 'r') as file:
    flight_cancellation_policy = file.read()

# Takes human readable policy and converts it to a AI routine


# +


CONVERSION_PROMPT = """
You are a helpful assistant tasked with taking an external facing help center article and converting it into a internal-facing programmatically executable routine optimized for an LLM. 
The LLM using this routine will be tasked with reading the policy, answering incoming questions from customers, and helping drive the case toward resolution.

Please follow these instructions:
1. **Review the customer service policy carefully** to ensure every step is accounted for. It is crucial not to skip any steps or policies.
2. **Organize the instructions into a logical, step-by-step order**, using the specified format.
3. **Use the following format**:
   - **Main actions are numbered** (e.g., 1, 2, 3).
   - **Sub-actions are lettered** under their relevant main actions (e.g., 1a, 1b).
      **Sub-actions should start on new lines**
   - **Specify conditions using clear 'if...then...else' statements** (e.g., 'If the product was purchased within 30 days, then...').
   - **For instructions that require more information from the customer**, provide polite and professional prompts to ask for additional information.
   - **For actions that require data from external systems**, write a step to call a function using backticks for the function name (e.g., `call the check_delivery_date function`).
      - **If a step requires the customer service agent to take an action** (e.g., process a refund), generate a function call for this action (e.g., `call the process_refund function`).
      - **Define any new functions** by providing a brief description of their purpose and required parameters.
   - **If there is an action an assistant can performon behalf of the user**, include a function call for this action (e.g., `call the change_email_address function`), and ensure the function is defined with its purpose and required parameters.
      - This action may not be explicitly defined in the help center article, but can be done to help the user resolve their inquiry faster
   - **The step prior to case resolution should always be to ask if there is anything more you can assist with**.
   - **End with a final action for case resolution**: calling the `case_resolution` function should always be the final step.
4. **Ensure compliance** by making sure all steps adhere to company policies, privacy regulations, and legal requirements.
5. **Handle exceptions or escalations** by specifying steps for scenarios that fall outside the standard policy.

**Important**: If at any point you are uncertain, respond with "I don't know."

Please convert the customer service policy into the formatted routine, ensuring it is easy to follow and execute programmatically.

"""


# +


def generate_routine(policy):
    try:
        messages = [
            {
                "role": "user",
                "content": f"""
                    {CONVERSION_PROMPT}

                    POLICY:
                    {policy}
                """
            }
        ]

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        

        return response.choices[0].message.content 
    except Exception as e:
        print(f"An error occurred: {e}")


# +


flight_cancellation_routine = generate_routine(flight_cancellation_policy)


# +


display(Markdown(flight_cancellation_routine))


# ## Evaluating Accuracy
# 
# Now that we have a routine generated with o1, we can run it against our evaluation suite and measure its accuracy.
# 
# We'll start by creating an agent that is equipped with the policy and a list of tools. It will be given messages from an existing conversation and will be tasked with determining the next best action to take

# +


def agent_response(transcript, policy, model):
    try:
        messages = [
            {
                "role": "system",
                "content": f"""
You are a customer service agent that is responsible for handling airline related issues. Below is the exact policy that you must follow to address the customer's issue

POLICY:
{policy}
                """
            }
        ]

        messages.extend(transcript)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS
        )
        
        return response.choices[0].message 
    except Exception as e:
        print(f"An error occurred: {e}")


# We will process each row in parallel to reduce runtime and compare the function call + inputs that the model selects against our expected function + parameters.

# +


def process_row(row_number, row, policy, model):
    try:
        # Extract values from the current row
        conversation_str = row['conversation']
        expected_function = row['expected_function']
        expected_inputs_str = row['expected_inputs']

        # Parse the conversation JSON
        try:
            conversation = json.loads(conversation_str)
        except json.JSONDecodeError as e:
            print(f"Error parsing 'conversation' in row {row_number}: {e}")
            conversation = None

        # Parse the expected_inputs JSON
        try:
            expected_inputs = json.loads(expected_inputs_str)
            # If expected_inputs is a string (double-encoded), parse it again
            if isinstance(expected_inputs, str):
                expected_inputs = json.loads(expected_inputs)
        except json.JSONDecodeError as e:
            print(f"Error parsing 'expected_inputs' in row {row_number}: {e}")
            expected_inputs = None

        # Extract the last assistant's message content if it exists
        response = agent_response(conversation, policy, model)
        assistant_message_content = response.content if response else None
        tool_calls = response.tool_calls

        # If the most recent response does not contain a tool call and just a message from the assistant, we rerun it once more to get our tool call.
        if not tool_calls:
            if assistant_message_content:
                # Append the assistant's message content to the conversation
                conversation.append({"role": "assistant", "content": assistant_message_content})
                # Make another call to agent_response
                response = agent_response(conversation, policy, model)
                tool_calls = response.tool_calls

        if not tool_calls:
            actual_function = None
            actual_inputs = None
            is_correct = False
        else:
            tool_call = tool_calls[0]  # Assuming we're only interested in the first tool call
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            is_correct = (function_name == expected_function) and (arguments == expected_inputs)
            actual_function = function_name
            actual_inputs = arguments

        return {
            'expected_function': expected_function,
            'expected_inputs': expected_inputs,
            'actual_function': actual_function,
            'actual_inputs': actual_inputs,
            'is_correct': is_correct,
            'assistant_message_content': assistant_message_content
        }

    except Exception as e:
        print(f"Error processing row {row_number}: {e}")
        return {
            'expected_function': row.get('expected_function'),
            'expected_inputs': row.get('expected_inputs'),
            'actual_function': None,
            'actual_inputs': None,
            'is_correct': False,
            'assistant_message_content': None
        }

def evaluate_function_calls(file_path, policy, model):
    records = []

    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            # Initialize the CSV reader with pipe as delimiter
            reader = csv.DictReader(csvfile, delimiter='|', quotechar='"', escapechar='\\')

            # Use ThreadPoolExecutor to process rows in parallel
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(process_row, row_number, row, policy, model): row_number for row_number, row in enumerate(reader, start=2)}
                for future in futures:
                    record = future.result()
                    records.append(record)

    except Exception as e:
        print(f"An unexpected error occurred while reading the CSV file: {e}")
        return

    df = pd.DataFrame(records)
    total_accuracy = df['is_correct'].mean()
    return df, total_accuracy


# Let's take a look at our results.

# +


# Assuming the CSV file is located at 'evals/functionCallingEval.csv'
df, accuracy = evaluate_function_calls('evals/functionCallingEval.csv', flight_cancellation_routine, 'gpt-4o-mini-2024-07-18')

# Display the accuracy as a mini header
display(Markdown(f"### Accuracy: {accuracy:.2%}"))

display(df)


# ## Metaprompting
# 
# Let's now leverage o1 again to add in a metaprompting loop to see if we can improve the quality of our evals.
# 
# We'll take the following multi-step approach:
# - We'll pass in the current routine + eval results to o1 and ask it analyze the results and update the routine accordingly
# - Since o1 does not currently support structured outputs, we'll chain with output with a 4o to enforce a schema we can parse
# - Finally, we take the new routine and run it back through our eval to generate new results
# 
# We'll run this loop a fixed number of times and see what improvements we can make

# +


def metaprompt(messages):
    try:
        response = client.chat.completions.create(
            model='o1-preview',
            messages=messages,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")


# +


def enforce_schema(updated_prompt):
    try:
        messages = [
            {
                "role": "system",
                "content": f"""
You will be given a response from an LLM that just generated a policy for flight cancellations.
Your task is to take just the policy exactly as it is written and return it in the defined json. Remove all parts from the LLM's answer that are not part of the policy.

LLM RESPONSE:
{updated_prompt}
                """
            }
        ]

        response = client.chat.completions.create(
            model='gpt-4o-2024-08-06',
            messages=messages,
            response_format= {
                "type": "json_schema",
                "json_schema": {
                    "name": "policy_output",
                    "schema": {
                    "type": "object",
                    "properties": {
                        "final_answer": { "type": "string" }
                    },
                    "required": ["final_answer"],
                    "additionalProperties": False
                    },
                    "strict": True
                }
            }

        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}") 


# +


updated_policy = flight_cancellation_routine
messages = [
    {
        "role": "user",
        "content": f"""
You are an agent that is responsible for improving the quality of routine instructions that are provided to a customer service LLM agent.

I am going to give you the policy for the customer service agent that contains detailed instructions on how to handle flight cancellations and changes.

You will also be provided with the results from an eval set that include the following:
    - conversation history: This is the conversation that we present to the LLM along with the system prompt
    - expected_function: This is the function we expect the LLM to call
    - expected_input: This is the input we expect the LLM to provide to the function
    - actual_function: This is the actual function the LLM called
    - actual_input: This is the actual input the LLM provided
    - assistant_message_content: This is the message the LLM generated when it returned its response
    - is_correct: True/False value depending on if the model responded correctly

Carefully analyze the instructions provided as well as the results of the eval. Get a firm understanding of the failures in the policy.

Return an updated policy that will perform better against the dataset.

Here is the current policy:
{flight_cancellation_policy}
"""
    }
]

for _ in range(5):
    # Evaluate the function calls with the current policy
    df, accuracy = evaluate_function_calls('evals/functionCallingEval.csv', updated_policy, 'gpt-4o-mini-2024-07-18')
    
    # Display the accuracy as a mini header
    display(Markdown(f"### Accuracy: {accuracy:.2%}"))
    display(df)
    results_json = df.to_json(orient='records')

    messages.append({
        "role": "user",
        "content": f"""
Here are the results based on the current policy:
{results_json}
"""
    })
    # Use the metaprompt function to get an updated policy
    temp_policy_json = enforce_schema(metaprompt(messages))
    temp_policy_str = temp_policy_json.strip("json```").strip("```")
    temp_policy = json.loads(temp_policy_str)["final_answer"]
    print(f"Corrected Policy: {temp_policy}")

    messages.append({
        "role": "assistant",
        "content": f"""
{temp_policy}
"""
    })

    # Update the policy for the next iteration
    updated_policy = temp_policy


# ## Distilling Down to a smaller model

# Each time we release a new snapshot of a model, it is always a challenge to ensure that your existing prompt works for the new snapshot.
# 
# In this example, we'll simulate that work by trying to get the routine to work for our older GPT 3.5 Turbo model.

# +


messages = [
    {
        "role": "user",
        "content": f"""
You are an agent that is responsible for improving the quality of routine instructions that are provided to a customer service LLM agent.

I am going to give you the policy for the customer service agent that contains detailed instructions on how to handle flight cancellations and changes.

You will also be provided with the results from an eval set that include the following:
    - conversation history: This is the conversation that we present to the LLM along with the system prompt
    - expected_function: This is the function we expect the LLM to call
    - expected_input: This is the input we expect the LLM to provide to the function
    - actual_function: This is the actual function the LLM called
    - actual_input: This is the actual input the LLM provided
    - assistant_message_content: This is the message the LLM generated when it returned its response
    - is_correct: True/False value depending on if the model responded correctly

Carefully analyze the instructions provided as well as the results of the eval. Get a firm understanding of the failures in the policy.

Return an updated policy that will perform better against the dataset.

Here is the current policy:
{updated_policy}
"""
    }
]

for _ in range(5):
    # Evaluate the function calls with the current policy
    df, accuracy = evaluate_function_calls('evals/functionCallingEval.csv', updated_policy, 'gpt-3.5-turbo-0125')
    
    # Display the accuracy as a mini header
    display(Markdown(f"### Accuracy: {accuracy:.2%}"))
    display(df)

    results_json = df.to_json(orient='records')

    messages.append({
        "role": "user",
        "content": f"""
Here are the results based on the current policy:
{results_json}
"""
    })
    # Use the metaprompt function to get an updated policy
    temp_policy_json = enforce_schema(metaprompt(messages))
    temp_policy_str = temp_policy_json.strip("json```").strip("```")
    temp_policy = json.loads(temp_policy_str)["final_answer"]
    print(f"Corrected Policy: {temp_policy}")

    messages.append({
        "role": "assistant",
        "content": f"""
{temp_policy}
"""
    })

    # Update the policy for the next iteration
    updated_policy = temp_policy

