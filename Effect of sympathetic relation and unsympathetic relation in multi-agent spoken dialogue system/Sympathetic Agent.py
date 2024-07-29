import speech_recognition as sr
import openai
import os
import keyboard
from gtts import gTTS
import pygame
import random

print("Welcome to the Multi-Conversation AI! Press Space to start the conversation. State 'stop' to end the conversation.")
conversation1 = []
conversation2 = []

def getAudio():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    return recognizer.recognize_google(audio)

openai.api_key = "YOUR_API_KEY"
openai.api_base = 'https://api.openai.com/v1/chat'
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

conversation1.append({ 'role': 'system', 'content': open_file(os.path.join(os.getcwd(), "Prompt Chat1.txt")) })
conversation2.append({ 'role': 'system', 'content': open_file(os.path.join(os.getcwd(), "Prompt Chat2.txt")) })

def gpt3Agent1(messages, engine='gpt-3.5-turbo', temp=0.9, tokens=100, freq_pen=2.0, pres_pen=2.0, stop=['DOGGIEBRO:', 'CHATTER:']):
    response = openai.Completion.create(
        model=engine,
        messages=messages,
        temperature=temp,
        max_tokens=tokens,
        frequency_penalty=freq_pen,
        presence_penalty=pres_pen,
        stop=stop)
    text = response['choices'][0]['message']['content'].strip()
    return text

def gpt3Agent2(messages, engine='gpt-3.5-turbo', temp=0.9, tokens=100, freq_pen=2.0, pres_pen=2.0, stop=['DOGGIEBRO:', 'CHATTER:']):
    response = openai.Completion.create(
        model=engine,
        messages=messages,
        temperature=temp,
        max_tokens=tokens,
        frequency_penalty=freq_pen,
        presence_penalty=pres_pen,
        stop=stop)
    text = response['choices'][0]['message']['content'].strip()
    return text

def ttsAgent1(response):
    tts = gTTS(text=response, lang='en')
    tts.save("response.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()

def ttsAgent2(response):
    tts = gTTS(text=response, lang='en-GB')
    tts.save("response.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()

while True:
    if keyboard.is_pressed('space'):
        print("Start talking!")
        agentSelected = random.randint(1, 2)
        user_input = getAudio()
        print("User:", user_input)
        conversation1.append({ 'role': 'user', 'content': user_input })
        conversation2.append({ 'role': 'user', 'content': user_input })
        while True:
            if user_input == "stop":
                break
            if(agentSelected == 1):
                response = gpt3Agent1(conversation1)
                print("Firebolt:", response)
                conversation1.append({ 'role': 'system', 'content': response })
                conversation2.append({ 'role': 'user', 'content': "Firebolt said \"" + response + "\""})
                ttsAgent1(response)
            else:
                response = gpt3Agent2(conversation2)
                print("English:", response)
                conversation1.append({ 'role': 'user', 'content': "English said \"" + response + "\""})
                conversation2.append({ 'role': 'system', 'content': response })
                ttsAgent2(response)
            userSpeaks = random.randint(1, 2)
            if(userSpeaks == 1):
                if(agentSelected == 1):
                    agentSelected = 2
                else:
                    agentSelected = 1
            else:
                agentSelected = random.randint(1, 2)
                print("Start talking!")
                user_input = getAudio()
                print("User: ", user_input)
                conversation1.append({ 'role': 'user', 'content': user_input })
                conversation2.append({ 'role': 'user', 'content': user_input })
        break