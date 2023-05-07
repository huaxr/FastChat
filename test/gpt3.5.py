import openai
import gradio as gr

openai.api_key = "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"

messages = [
    {"role": "system", "content": "You are an AI specialized in Oman Arab Bank. Please ask me anything related to Oman Arab Bank."},
]

# Define a list of Oman Arab Bank-related keywords
bank_keywords = ['oman arab bank', 'oab', 'banking', 'finance', 'financial', 'account', 'credit', 'loan']

def is_bank_related(input):
    # Check if any of the Oman Arab Bank-related keywords are present in the input
    # for keyword in bank_keywords:
    #     if keyword in input.lower():
    #         return True
    # return False
    return True

def chatbot(input):
    if input:
        # Only send the user's input to the OpenAI API if it is related to Oman Arab Bank
        if is_bank_related(input):
            messages.append({"role": "user", "content": input})
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )
            reply = chat.choices[0].message.content
            messages.append({"role": "assistant", "content": reply})
        else:
            # If the input is not related to Oman Arab Bank, return a message informing the user
            reply = "I'm sorry, I can only provide information related to Oman Arab Bank. Please ask me something about Oman Arab Bank. Try using the words for better queries 'oman arab bank', 'oab', 'banking', 'finance', 'financial', 'account', 'credit', 'loan"
        return reply

inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")
outputs = gr.outputs.Textbox(label="Reply")
# Add bank-related keywords as the interface's description
description = "Please ask a question related to Omana Arab Bank or any of its services. Keywords: " + ", ".join(bank_keywords)

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="Oman Arab Bank Chatbot",
             description="Ask anything you want about Oman Arab Bank",
             theme="dark").launch(share=True)