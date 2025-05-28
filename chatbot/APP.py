import json
import random
import tkinter as tk
from tkinter import scrolledtext, ttk

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Load the model and intents
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "YAM"

# Create the main window
root = tk.Tk()
root.title("Chatbot")
root.geometry("1280x700")
root.configure(bg="#075E54")

# Define styles
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=6)
style.configure("TEntry", font=("Helvetica", 12), padding=10)
style.configure("TLabel", font=("Helvetica", 12), background="#075E54", foreground="#ffffff")

# Create a frame for the chat window
chat_frame = tk.Frame(root, bg="#ECE5DD")
chat_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Create a scrolled text widget
chat_window = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state='disabled', bg="#DCF8C6", fg="#000000", font=("Helvetica", 12))
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create a frame for the entry and buttons
entry_frame = tk.Frame(root, bg="#075E54", pady=10)
entry_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

# Create an entry widget with rounded corners
entry = tk.Entry(entry_frame, font=("Helvetica", 14), width=50, bd=0, bg="#FFFFFF")
entry.grid(row=0, column=0, padx=(10, 0), pady=5, sticky="ew")
entry_frame.grid_columnconfigure(0, weight=1)

def send_message(event=None):
    user_input = entry.get()
    if user_input.strip() == "":
        return
    
    # Display the user's message in the chat window
    chat_window.config(state='normal')
    chat_window.insert(tk.END, f"You: {user_input}\n", ("user",))
    chat_window.config(state='disabled')
    chat_window.yview(tk.END)
    
    entry.delete(0, tk.END)
    
    # Generate bot response
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                bot_response = random.choice(intent['responses'])
                break
    else:
        bot_response = "I do not understand..."
    
    # Display the bot's response in the chat window
    chat_window.config(state='normal')
    chat_window.insert(tk.END, f"{bot_name}: {bot_response}\n", ("bot",))
    chat_window.config(state='disabled')
    chat_window.yview(tk.END)

# Tag configurations for user and bot messages
chat_window.tag_config("user", foreground="#25D366", font=("Helvetica", 12, "bold"))
chat_window.tag_config("bot", foreground="#075E54", font=("Helvetica", 12, "italic"))

# Create a send button
send_button = tk.Button(entry_frame, text="Send", bd=0, command=send_message, bg="#128C7E", fg="#ffffff", activebackground="#25D366", font=("Helvetica", 12))
send_button.grid(row=0, column=1, padx=(5, 10), pady=5)

# Bind the enter key to the send_message function
root.bind('<Return>', send_message)

# Make the chat frame expandable
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Run the Tkinter event loop
root.mainloop()
