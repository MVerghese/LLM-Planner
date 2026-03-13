from openai import OpenAI
import numpy as np
from typing import List
import requests
import json
from utils import encode_image, load_images

class VLLM_Interface:
	def __init__(self, engine, client=None):
		self.engine = engine
		if not client:
			self.client = OpenAI(
				base_url="http://localhost:8000/v1",
				api_key="token-abc123",
			)
		else:
			self.client = client
		
	
	def get_tokenization(self, prompt):
		api_url = "http://localhost:8000/tokenize"
		headers = {
			"accept": "application/json",
			"Content-Type": "application/json"
		}
		data = {
			"prompt": prompt
		}

		response = requests.post(api_url, headers=headers, data=json.dumps(data))
		if response.status_code == 200:
			tokenization = response.json()
			return tokenization["tokens"]
		else:
			raise Exception(f"Tokenization API request failed with status code {response.status_code}: {response.text}")


	
	def generate(self, prompt, max_tokens=300, temperature=0.0, images=None, n = 1, return_logprobs=False, num_logprobs=5, logit_bias = {}, add_generation_prompt = True, continue_final_message = False):
		if temperature == 0.0 and n > 1:
			print("Warning: Requesting multiple outputs with temperature set to 0.0 will result in identical outputs. Setting n to 1 but this is probably not what you want")
			n = 1
		if 'gpt-4o' in self.engine:
			if not add_generation_prompt:
				print("Warning GPT models do not support omitting the generation prompt. Ignoring this option")
			if continue_final_message:
				print("Warning GPT models do not support continuing the final message. Ignoring this option")
			# Create the base message content
			message_content = []
			
			# Add image if provided
			if images and len(images) > 0:
				message_content.append({
					"type": "image_url",
					"image_url": {
						"url": f"data:image/jpeg;base64,{images[0]}"
					}
				})
			
			# Add text prompt
			message_content.append({
				"type": "text",
				"text": prompt,
			})
			
			# Make the API call
			response = self.client.chat.completions.create(
				model=self.engine,
				messages=[
					{
						"role": "user",
						"content": message_content,
					}
				],
				max_tokens=max_tokens,
				temperature=temperature,
				n=n
			)
		elif "Qwen3-VL" in self.engine:
			 # Create the base message content
			message_content = []
			

			# Add image if provided
			if images and len(images) > 0:
				for image in images:
					message_content.append({
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{image}"
						}
					})
			
			# Add text prompt
			message_content.append({
				"type": "text",
				"text": prompt,
			})
			extra_body = {
				"add_generation_prompt": add_generation_prompt,
				"continue_final_message": continue_final_message
			}
			
			# Make the API call
			response = self.client.chat.completions.create(
				model=self.engine,
				messages=[
					{
						"role": "user",
						"content": message_content,
					}
				],
				extra_body=extra_body,
				max_tokens=max_tokens,
				temperature=temperature,
				n=n,
				logprobs=return_logprobs,
				top_logprobs=num_logprobs if return_logprobs else 0,
				logit_bias = logit_bias

			)

			

		else:
			raise ValueError(f"{self.engine} is not supported!")

		generated = [choice.message.content for choice in response.choices]
		if return_logprobs:
			logprobs = [choice.logprobs.content for choice in response.choices]
			return generated, logprobs
		else:
			return generated

	def get_token_logprobs(self, prompt, tokens, images=None, requested_logprobs=10):
		if 'gpt-4o' in self.engine:
			message_content = []
			
			if images and len(images) > 0:
				message_content.append({
					"type": "image_url",
					"image_url": {
						"url": f"data:image/jpeg;base64,{images[0]}"
					}
				})
			
			message_content.append({
				"type": "text",
				"text": prompt,
			})
			
			response = self.client.chat.completions.create(
				model=self.engine,
				messages=[
					{
						"role": "user",
						"content": message_content,
					}
				],
				max_tokens=1,
				logprobs=True,
				top_logprobs=requested_logprobs
			)
			
		elif "Qwen3-VL" in self.engine:
			message_content = []
			
			if images and len(images) > 0:
				for image in images:
					message_content.append({
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{image}"
						}
					})
			# else:
				# print("No images provided for Qwen3-VL model, proceeding with text only input.")
			
			message_content.append({
				"type": "text",
				"text": prompt,
			})
			
			response = self.client.chat.completions.create(
				model=self.engine,
				messages=[
					{
						"role": "user",
						"content": message_content,
					}
				],
				max_tokens=1,
				logprobs=True,
				top_logprobs=requested_logprobs
			)
		response_logprobs = response.choices[0].logprobs.content[0].top_logprobs
		# print(response_logprobs)
		logprobs = {token.token: token.logprob for token in response_logprobs if token.token in tokens}
		assert len(logprobs) == len(tokens), f"Expected logprobs for all tokens, but got {len(logprobs)} out of {len(tokens)}, please increase the 'requested_logprobs' parameter if needed."
		return logprobs
			# print("Response from Qwen3-VL model:", response.choices[0].logprobs.content[0].top_logprobs)
			# # import pdb; pdb.set_trace()
			# # return response.choices[0].message.logprobs
	
	def yes_no_probabilities(self, prompt, images=None):
		logprobs_dict = self.get_token_logprobs(prompt, tokens=["Yes","No"], images=images, requested_logprobs=500)
		logprobs = np.array([logprobs_dict["Yes"], logprobs_dict["No"]])
		probs = np.exp(logprobs)
		probs /= np.sum(probs)  # Normalize to get probabilities
		return probs

	def MCQ_probabilities(self, prompt, images=None, options=["A","B","C","D","E"]):
		logprobs_dict = self.get_token_logprobs(prompt, tokens=options, images=images, requested_logprobs=500)
		logprobs = np.array([logprobs_dict[option] for option in options])
		probs = np.exp(logprobs)
		probs /= np.sum(probs)  # Normalize to get probabilities
		return probs
	
	def VQA(self, question, images, num_candidate_images = 1, recency_bias = 0.5):
		"""
		Visual Question Answering (VQA) using the LLM.
		"""
		image_selection_prompt = f"Can the question: {question} be answered using information in this image?"
		image_scores = [self.yes_no_probabilities(image_selection_prompt, images=[image])[0] for image in images]  # Probability of "Yes"
		image_times = np.linspace(0,1,len(images))
		image_recency_weights = np.exp(recency_bias * (image_times-1))  # Exponential decay for older images
		image_scores_weighted = np.array(image_scores) * image_recency_weights
		# Select the top-k images based on their scores
		top_k_images = np.argsort(image_scores_weighted)[-num_candidate_images:][::-1]
		answer_prompt = f"Based on the information in the images, answer the following question in a single complete sentence:\n {question}{'?' if not question.endswith('?') else ''}"
		answer = self.generate(answer_prompt, images=[images[i] for i in top_k_images], n=1)[0]
		return answer

def main():
	engine = 'Qwen/Qwen3-VL-8B-Instruct'
	vllm_interface = VLLM_Interface(engine)

	# image_path = "/home/mverghese/Downloads/L_Big.jpg"
	# im = load_images([image_path])[0]
	# encoded_image = encode_image(im)
	# prompt = "Here is a description of a 3D printed part: **L-Shaped Geometry with a Vertical Post:** The part has a distinct L-shape, with a tall vertical post and a shorter horizontal base.\n\n2.  **Small, Thin Wing Feature on Top:** The most notable feature is the small, thin, wing-like structure extending from the top of the vertical post. \n    *   **Overhang:** This wing is a significant overhang (likely >45 degrees relative to the print bed).\n  I have an issue with the wing part curling. What print parameter should I change to fix this?"


	# out = vllm_interface.generate(prompt, images=[encoded_image], temperature=0.0)
	# print(out)

	# prompt = "There is a cup in the cabinet and the cabinet is closed. If I tried to pickup the cup, would I succeed?"
	# prompt = "There is a cup in the cabinet. I don't know if the cabinet is open or closed. If I tried to pickup the cup, would I succeed?"
	# tokenize = vllm_interface.get_tokenization(prompt)
	# print(tokenize)
	# # logprobs = vllm_interface.get_token_logprobs(prompt, tokens=["Yes","No"], requested_logprobs=20)
	prompt_file = "/home/mverghese/LLM_Grounding/LLM-Planner/e2e/prompt.txt"
	with open(prompt_file, 'r') as f:
		prompt = f.read()
	response = vllm_interface.generate(prompt, n=1, temperature=0.0, add_generation_prompt=True, continue_final_message=False)
	print(response)
	# import pdb; pdb.set_trace()
	# # prompt = "There is a cup in the cabinet and the cabinet is open. Can you see the cup?"
	# probs = vllm_interface.yes_no_probabilities(prompt)
	# print(f"Probabilities for 'Yes' and 'No': {probs}")
	
if __name__ == "__main__":
	main()