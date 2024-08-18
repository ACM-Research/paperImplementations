import speech_recognition as sr
import openai
import os
from gtts import gTTS

conversation = []

def getAudio():
	recognizer = sr.Recognizer()
	microphone = sr.Microphone()
	with microphone as source:
		recognizer.adjust_for_ambient_noise(source)
		audio = recognizer.listen(source)
	return recognizer.recognize_google(audio)

openai.api_key = "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"
openai.api_base = 'https://api.openai.com/v1/chat'
def open_file(filepath):
	with open(filepath, 'r', encoding='utf-8') as infile:
		return infile.read()

conversation.append({ 'role': 'system', 'content': open_file(os.path.join(os.getcwd(), "Assets\Prompt Chat.txt")) })

def gpt3Agent(messages, engine='gpt-3.5-turbo', temp=0.9, tokens=100, freq_pen=2.0, pres_pen=2.0, stop=['DOGGIEBRO:', 'CHATTER:']):
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

def ttsAgent(response):
	tts = gTTS(text=response, lang='en')
	tts.save("Assets\\response.mp3")
	with open("Assets\Captions.txt", 'a') as captions:
		captions.write("\n|\n")

while True:
	captions = open("Assets\Captions.txt", 'r')
	if captions.read() == "~":
		with open("Assets\Captions.txt", 'w') as captions:
			captions.write("Start Speaking!")
		user_input = getAudio()
		conversation.append({ 'role': 'user', 'content': user_input })
		with open("Assets\Captions.txt", 'w') as captions:
			captions.write(f"User: {user_input}\n")
		response = gpt3Agent(conversation)
		conversation.append({ 'role': 'system', 'content': response })
		with open("Assets\Captions.txt", 'a') as captions:
			captions.write(f"\nVirtul Agent: {response}\n")
		ttsAgent(response)