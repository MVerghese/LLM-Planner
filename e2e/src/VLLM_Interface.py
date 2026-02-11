from openai import OpenAI
import numpy as np

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
	
	def generate(self, prompt, max_tokens=300, temperature=1.0, images=None):
		if 'gpt-4o' in self.engine:
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
				temperature=temperature
			)
			return response.choices[0].message.content
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
				temperature=temperature
			)
			return response.choices[0].message.content

		else:
			raise ValueError(f"{engine} is not supported!")
	
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
			else:
				print("No images provided for Qwen3-VL model, proceeding with text only input.")
			
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

def main():
	engine = 'Qwen/Qwen3-VL-8B-Instruct'
	vllm_interface = VLLM_Interface(engine)
	prompt = "There is a cup in the cabinet and the cabinet is closed. If I tried to pickup the cup, would I succeed?"
	prompt = "There is a cup in the cabinet. I don't know if the cabinet is open or closed. If I tried to pickup the cup, would I succeed?"
	# prompt = "There is a cup in the cabinet and the cabinet is open. Can you see the cup?"
	probs = vllm_interface.yes_no_probabilities(prompt)
	print(f"Probabilities for 'Yes' and 'No': {probs}")
	
if __name__ == "__main__":
	main()