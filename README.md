# llama2_library_service_equality

This repo hosts scripts and results for the manuscript *Service Equality in AI-powered Virtual Reference*.
The goal is to simulate user interactions with a virtual AI librarian, using a diverse dataset to uncover potential 
biases in response patterns related to the user's background.

## Reproduction
Before running the experiment, ensure all dependencies are installed by running (for Linux or Mac):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduction
Before running the experiment, ensure all dependencies are installed by running (for Linux or Mac):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running Experiments

The experiments were conducted on Montana State University's Tempest cluster. See the sbatch files in `runs` for 
details. For local reproduction, you can specify the number of runs and the model size to use. 
The original experiment performed 2,000 runs.

To run an experiment simulating 10 user interactions with a virtual AI librarian using the Llama 2-Chat-7b model:

```bash
python -m run --num_runs 10 --model_name 7b
```

### Model Configurations and Benchmarking

The experiments can be run using different sizes of the Llama 2 model:

- Llama 2-Chat 7b: Approximately 30 seconds per sample on an A40 GPU.
- Llama 2-Chat 13b and 70b: These models use 4-bit quantization for a reduced memory footprint and faster inference. 
The former took 30 seconds per generation, and the latter took 55 seconds.

During the text generation phase, we utilized a temperature setting of 0.7 and applied nucleus sampling with a threshold
of 0.9.
 

### Results
The experiment's results will be saved in the `results` directory, named according to the model size (e.g., `7b.json`). 
The results include detailed information about the synthetic query (including a fixed system prompt), the simulated 
user's background, and the AI-generated response.

### Analysis
Please refer to the 
[Colab Notebook](https://colab.research.google.com/drive/1sNTrjcFFRAOjy7rIhauaQjpXWmwtPDVi?usp=sharing) for details. 


## License
0BSD

## Contact
hw56@indiana.edu

## Acknowledgements
Computational efforts were performed on the Tempest High Performance Computing System, operated and supported by 
University Information Technology Research Cyberinfrastructure at Montana State University.

