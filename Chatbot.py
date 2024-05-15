import re
def chatbot_response(user_input):
    if re.search(r'\b(hello|hi)\b',user_input, re.IGNORECASE):
        return "Hello! How can I assist you today?"
    elif re.search(r'\b(how are you)\b', user_input, re.IGNORECASE):
        return "I'm just a chatbot, but thanks for asking!"
    elif re.search(r'\b(goodbye|bye)\b', user_input, re.IGNORECASE):
        return "Goodbye! Have a great day!"
    else:
        return "Sorry, I didn't understand that."

def main():
    print("Welcome to the Rule-Based Chatbot!")
    print("You can start chatting. Type 'quit' to exit.")
    while True:
        user_input=input("You: ")
        if user_input.lower()=="quit":
            print("Chatbot: Goodbye!")
            break
        else:
            response=chatbot_response(user_input)
            print("Chatbot:",response)

if __name__=="__main__":
    main()
    