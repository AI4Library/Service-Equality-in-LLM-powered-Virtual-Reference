# llama2_library_service_equality

This framework is designed for conducting experiments to assess bias in AI-based virtual reference services, focusing
on responses to queries based on different ethnic and gender groups. 
The goal is to simulate user interactions with a virtual AI librarian, using a diverse dataset to uncover potential 
biases in response patterns related to the user's background.

## Getting Started

### Reproduction
Ensure all dependencies are installed by running:

```bash
pip install -r requirements.txt
```

### Running Experiments
The framework allows for running experiments based on ethnic or gender backgrounds. 
You can specify the number of runs and the model size to use.

### Example Command
To run an experiment analyzing ethnic and gender bias 10 times using the Llama 2-Chat-7b model:
```bash
python -m run --num_runs 10 --model_name 7b
```

### Model Configurations and Benchmarking
The experiments can be run using different configurations of the Llama 2 model:

- 7b: The base model, not quantized. Approximately 30 seconds per sample on an A40 GPU, totaling around 8.5 hours for 
1000 samples.
- 13b and 70b: These models use 4-bit quantization for reduced memory footprint and faster inference.

### Results
The experiment's results will be saved in the results directory, named according to the experiment type and model name
(e.g., ethnic-7b.json). The results include detailed information about the query, the simulated user's background, and 
the AI-generated response.

### License
0BSD

### Contact
hw56@indiana.edu






[//]: # (# AI4Library)

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (```bash)

[//]: # (# run ethnic expriment 10 times using llama-2-chat-7b)

[//]: # (python -m run --exp ethnic --num_runs 10 --model_name 7b)

[//]: # (```)

[//]: # ()
[//]: # (7b is not quantized.)

[//]: # (~30s/sample, on an A40 GPU. 8.5 hrs for 1000 samples)

[//]: # ()
[//]: # (13b and 70b are in 4-bit quantization.)

[//]: # (13b: 30s per sample)

[//]: # (70b: 55s per sample)