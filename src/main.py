# # src/main.py
# import argparse
# import sys
# from src.cosmos_db import create_cosmos_container
# from src.ui import chatbot_ui

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--setup-db", action="store_true", help="Set up Cosmos DB container with vector indexing.")
#     parser.add_argument("--run-chat", action="store_true", help="Launch the chatbot UI.")

#     args = parser.parse_args()
#     if not any(vars(args).values()):
#         parser.print_help()
#         sys.exit(1)

#     if args.setup_db:
#         create_cosmos_container()

#     if args.run_chat:
#         demo = chatbot_ui()
#         demo.launch(share=True)

# if __name__ == "__main__":
#     main()

import sys
from src.chatbot import generate_response

def run_terminal_chat():
    print("Welcome to the Terminal Chatbot!")
    print("Type your message and press Enter (type 'exit' or 'quit' to end the session).\n")
    while True:
        user_query = input("You: ")
        if user_query.strip().lower() in ["exit", "quit"]:
            print("Exiting chat. Goodbye!")
            break
        # Generate a response using your backend
        answer = generate_response(user_query)
        print("Bot:", answer)
        print("-" * 40)

if __name__ == "__main__":
    run_terminal_chat()
