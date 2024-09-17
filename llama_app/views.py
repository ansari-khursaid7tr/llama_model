from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Define a dictionary of available models
MODELS = {
    "model1": "your_repo_id_for_model1",
    "model2": "your_repo_id_for_model2",
    "model3": "your_repo_id_for_model3",
    "model4": "your_repo_id_for_model4",
}

API_TOKEN = "your_api_token_here"  # Replace with your actual API token

# Dictionary to store loaded models and tokenizers
loaded_models = {}

def load_model(repo_id):
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, use_auth_token=API_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(repo_id, use_auth_token=API_TOKEN)
        print(f"Successfully loaded model from {repo_id}")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model from {repo_id}: {str(e)}")
        return None, None

def home(request):
    return render(request, 'llama_app/templates/index.html', {'models': MODELS.keys()})

@csrf_exempt
def generate(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        input_text = data['input_text']
        selected_model = data['model']

        if selected_model not in MODELS:
            return JsonResponse({'error': 'Invalid model selection'}, status=400)

        repo_id = MODELS[selected_model]

        if repo_id not in loaded_models:
            model, tokenizer = load_model(repo_id)
            if model is None or tokenizer is None:
                return JsonResponse({'error': f'Failed to load model {selected_model}'}, status=500)
            loaded_models[repo_id] = (model, tokenizer)
        else:
            model, tokenizer = loaded_models[repo_id]

        try:
            # Tokenize input
            input_ids = tokenizer.encode(input_text, return_tensors='pt')

            # Generate text
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_length=1000,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    no_repeat_ngram_size=2
                )

            # Decode output and remove the input text
            full_output = tokenizer.decode(output[0], skip_special_tokens=True)
            generated_text = full_output[len(input_text):].strip()

            return JsonResponse({'generated_text': generated_text})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)