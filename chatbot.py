import openai
import speech_recognition as sr
import pyttsx3

# Set up GPT-3.5 API access
openai.api_key = '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'

# Function to interact with GPT-3.5 API
def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Function for speech-to-text conversion
def convert_speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak something:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None

# Function for text-to-speech conversion
def convert_text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Function to get user input
def get_user_input():
    try:
        user_input = input("You: ")
        return user_input
    except KeyboardInterrupt:
        print("\nGoodbye!")
        exit()

# Main function
def main():
    print("Hello! I'm your voice assistant.")

    while True:
        user_input = get_user_input()

        # Speech-to-Text
        if user_input.lower() == 'speech':
            user_input_text = convert_speech_to_text()
        else:
            user_input_text = user_input

        # Generate response from GPT-3.5-turbo API
        chatbot_response = generate_response(user_input_text)

        print("ChatGPT:", chatbot_response)

        # Text-to-Speech
        convert_text_to_speech(chatbot_response)

if __name__ == "__main__":
    main()  