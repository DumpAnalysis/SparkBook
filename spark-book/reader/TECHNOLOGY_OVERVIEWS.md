# Technology overviews

Each entry explains a technology used in the WSL2 laboratory and gives an example of what to inspect. Select a topic below, or open an individual numbered guide to read all the background for that exercise. Page numbers refer to Version 59.

- [Adapting speech and image models](#topic-mediaadapter)
- [Agent loops, tools and action validation](#topic-agent)
- [Bash, Python and the working directory](#topic-shell)
- [BERT, WordPiece and masked language modeling](#topic-bert)
- [Bytes, memory budgets and unified memory](#topic-budgets)
- [Calibration and explicit quantize and dequantize operations](#topic-qdq)
- [Checking the telemetry itself](#topic-obschecks)
- [Checkpoints, adapters and merged weights](#topic-checkpoints)
- [Client workers, queues and concurrency](#topic-concurrency)
- [Clients, servers and a local HTTP request](#topic-http)
- [Compiling CUDA C++ for this GPU](#topic-build)
- [Compiling PyTorch with Dynamo, Inductor and Triton](#topic-compile)
- [Compute Sanitizer and CUDA correctness checks](#topic-sanitizer)
- [Connecting speech recognition, generation and synthesis](#topic-multimediapipe)
- [Container registries, manifests and platform selection](#topic-oci)
- [Contrastive training for sentence encoders](#topic-contrast)
- [CPU execution and partial GPU offload](#topic-placement)
- [The CUDA driver, toolkit and runtime](#topic-toolkit)
- [CUDA streams, events and graphs](#topic-streams)
- [CUDA-GDB, breakpoints and exception cores](#topic-debugger)
- [Custom PyTorch operators and gradients](#topic-operators)
- [Data parallelism, sharding and gradient averaging](#topic-parallel)
- [DataLoader, queues and prefetching](#topic-supply)
- [Datasets, labels and independent evaluation](#topic-splits)
- [Decision calibration, risk and coverage](#topic-calibration)
- [Device buffer exports, allocator snapshots and GPU core dumps](#topic-dumps)
- [Diagnostic observations, controls and written conclusions](#topic-reasoning)
- [Diagnostic patterns and evidence cards](#topic-casebook)
- [Diffusion, noise prediction and a sampling schedule](#topic-diffusion)
- [Distributed training on one physical laptop](#topic-dist)
- [Docker Compose and service networking](#topic-compose)
- [Docker images, containers and volumes](#topic-docker)
- [Dynamic batching and bounded admission](#topic-batching)
- [Fine-tuning, LoRA and trainable parameters](#topic-lora)
- [Floating-point precision and numerical tolerances](#topic-precision)
- [Git repositories, snapshots and saved run directories](#topic-identifiers)
- [GPU metrics, samples and missing observations](#topic-metrics)
- [GPU scheduling in a Kubernetes cluster](#topic-scheduling)
- [Gradients, optimizers and resuming an update](#topic-trainingstate)
- [Grafana dashboards and data sources](#topic-grafana)
- [Grounding a model in retrieved evidence](#topic-grounding)
- [Guest paging, Windows commit and vmstat](#topic-paging)
- [Hugging Face Hub, snapshots and dataset cards](#topic-hfhub)
- [Images, waveforms and model processors](#topic-media)
- [Inference APIs, readiness and service health](#topic-serving)
- [Jev decision models and typed questions](#topic-jev)
- [Kernel fusion and CUDA event timing](#topic-fusion)
- [Kubernetes, K3s, Pods and Services](#topic-kube)
- [LangChain agents and local model adapters](#topic-langchain)
- [LangGraph state, checkpoints and human review](#topic-langgraph)
- [LangSmith tracing and evaluation](#topic-langsmith)
- [Latency, throughput and repeatable timing](#topic-timing)
- [LM Studio, llmster and the lms command line](#topic-lmstudio)
- [Loading a model larger than the GPU budget](#topic-bigmodel)
- [Loading a model with llama.cpp](#topic-server)
- [Matrix layout, tiles and shared memory](#topic-tiles)
- [Matrix multiplication on the GPU](#topic-matmul)
- [Model caches, local files and offline execution](#topic-offline)
- [Model downloads, Git revisions and GGUF](#topic-files)
- [Model weights, tokens and the context window](#topic-model)
- [MusicGen and discrete audio tokens](#topic-music)
- [Nemotron, mixtures of experts and hybrid state](#topic-nemotron)
- [Nsight Compute and hardware counters](#topic-ncu)
- [Nsight Systems, NVTX and CUPTI timelines](#topic-nsys)
- [NumPy arrays, dataframes and data provenance](#topic-arrays)
- [The NVIDIA Container Toolkit and GPU visibility](#topic-containersgpu)
- [NVIDIA NeMo AutoModel and supervised training recipes](#topic-nemo)
- [NVIDIA NGC Catalog, resources and container images](#topic-ngc)
- [NVIDIA Triton Inference Server and model repositories](#topic-tritonserver)
- [Ollama, local model names and residency](#topic-ollama)
- [ONNX graphs and ONNX Runtime execution providers](#topic-onnx)
- [OpenTelemetry traces, spans and correlated logs](#topic-otel)
- [Packing INT4 values into bytes](#topic-packing)
- [PCA and a two-dimensional map of embeddings](#topic-pca)
- [Plotting GPU activity against a workload](#topic-plots)
- [Prometheus, exporters and time series](#topic-prometheus)
- [Prompts, generated answers and acceptance checks](#topic-prompt)
- [PromQL rates, histograms and alert rules](#topic-promql)
- [PTX, SASS and device registers](#topic-instructions)
- [PydanticAI and typed model outputs](#topic-pydantic)
- [Python environments, pip and binary wheels](#topic-pythonpackages)
- [PyTorch allocator snapshots and tensor ownership](#topic-allocator)
- [PyTorch Profiler and operators](#topic-profiler)
- [PyTorch tensors and synchronized CUDA execution](#topic-torch)
- [Quantization and weight formats](#topic-quant)
- [RAPIDS Memory Manager and allocation statistics](#topic-rmm)
- [RAPIDS, cuDF and cuML](#topic-rapids)
- [Reading an exported trace with SQLite](#topic-sqlite)
- [Reducing a failing case and isolating a change](#topic-reductiontest)
- [Reductions, histograms and atomic updates](#topic-reductions)
- [Retrieval, relevance labels and rank metrics](#topic-ranking)
- [RMS normalization, softmax and numerical stability](#topic-normalization)
- [Roofline models of kernel performance](#topic-roofline)
- [Sentence embeddings and semantic similarity](#topic-embeddings)
- [Sharing a GPU and owning a model](#topic-ownership)
- [Speech recognition with Whisper](#topic-asr)
- [Stable Diffusion XL, Diffusers and the VAE](#topic-sdxl)
- [Stalled workers and stack traces](#topic-stalls)
- [Streaming text and the first content event](#topic-streaming)
- [t-SNE and neighbourhoods in a plot](#topic-tsne)
- [TensorRT engines and optimization profiles](#topic-tensorrt)
- [Text-to-speech with MMS and VITS](#topic-tts)
- [Token vectors, attention masks and pooling](#topic-pooling)
- [Transformer attention and cached decoding](#topic-attention)
- [Transformers, tokenizers and the Hugging Face libraries](#topic-hfstack)
- [UMAP and the neighbourhood graph](#topic-umap)
- [Unsloth, QLoRA and a training smoke test](#topic-unsloth)
- [Vector indexes and model identity](#topic-vectorindex)
- [Verifying a CUDA allocation above guest RAM](#topic-retained)
- [Vision-language models and visual grounding](#topic-vlm)
- [vLLM and serving language models](#topic-vllm)
- [Warps, Tensor Cores, cuBLAS and CUTLASS](#topic-warps)
- [Waveforms, spectrograms and audio levels](#topic-spectrum)
- [Windows, WSL2 and the mapped GPU](#topic-wsl)
- [Word error rate and transcript alignment](#topic-wer)
- [Writing a CUDA kernel with threads and blocks](#topic-kernels)

<a id="topic-mediaadapter"></a>

## Adapting speech and image models

[Exercise 200](exercises/200.txt) · book page 640

A speech recognizer can learn from recordings paired with verified transcripts. An image adapter can learn from images paired with descriptions. In both cases the examples define what the training objective rewards. Two recordings of the same utterance or nearly identical image captions can make a held-out comparison misleading. Exercise 200 updates existing parameters in Whisper’s final decoder block. Exercise 201 instead adds trainable low-rank factors to the image model.

For speech, inspect the audio and its actual reference words before training. For images, hold the base, prompt, seed and sampling settings fixed when enabling or disabling the adapter. Score the requested change and what should be preserved. A changed waveform or image fingerprint establishes a different output.

Further reading

- [Hu, E. J., et al. LoRA: Low-Rank Adaptation of Large Language Models. 2021](https://arxiv.org/abs/2106.09685)
- [Hugging Face. Whisper Transformers 4.57.1. Consulted 27 October 2026](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/whisper)
- [Hugging Face Evaluate. Word error rate metric. Consulted 27 October 2026](https://huggingface.co/spaces/evaluate-metric/wer/blob/main/README.md)
- [Hugging Face. LoRA training with Diffusers 0.35.1. Consulted 27 October 2026](https://huggingface.co/docs/diffusers/v0.35.1/en/training/lora)
- [Hugging Face. SDXL LoRA training script at v0.35.1. Consulted 27 October 2026](https://github.com/huggingface/diffusers/blob/v0.35.1/examples/text_to_image/train_text_to_image_lora_sdxl.py)

<a id="topic-agent"></a>

## Agent loops, tools and action validation

[Exercise 17](exercises/017.txt) · book page 94

An agent combines a decision step with actions the application can perform. A model may propose reading a file or calling a tool. A controller checks the tool name and arguments, executes an allowed action and gives the resulting observation back to the decision step. The loop needs a stop condition.

A replay supplies predetermined proposals, so we can test the controller with a known sequence. A live planner uses the model’s own proposals. Read proposal, validation, action and result in order. Only the action record establishes which tool actually ran.

Further reading

- [Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y. ReAct Synergizing Reasoning and Acting in Language Models. ICLR 2023](https://arxiv.org/abs/2210.03629)
- [LangChain. Agents. Consulted 28 October 2026](https://docs.langchain.com/oss/python/langchain/agents)
- [Pydantic. PydanticAI output validation. Consulted 28 October 2026](https://pydantic.dev/docs/ai/core-concepts/output/)

<a id="topic-shell"></a>

## Bash, Python and the working directory

[Exercise 1](exercises/001.txt) · book page 22

Bash reads a command, expands variables and starts the named program. In bash inventory.sh, Bash runs a shell script. In python gpu_probe.py, the selected Python interpreter runs a Python file. Relative filenames start at the working directory. The tilde in ~/work denotes your home directory, and $LAB_RUN expands to the location of your current run.

A Python virtual environment gives the project its own interpreter and installed packages. Activating it affects the current shell. The command python -c "import sys; print(sys.executable)" shows which interpreter would run the next Python exercise. A new terminal can select a different one.

Further reading

- [Microsoft. Working across Windows and Linux file systems](https://learn.microsoft.com/en-us/windows/wsl/filesystems)
- [Python Software Foundation. Virtual environments and packages](https://docs.python.org/3/tutorial/venv.html)
- [GNU Project. Bash shell parameters](https://www.gnu.org/software/bash/manual/html_node/Shell-Parameters.html)

<a id="topic-bert"></a>

## BERT, WordPiece and masked language modeling

[Exercise 83](exercises/083.txt) · book page 298

BERT is a transformer encoder that builds a representation of a token using surrounding input on both sides. A WordPiece tokenizer splits text into vocabulary entries, sometimes parts of words, and maps them to integer IDs. A masked-language-model objective hides selected positions and trains the encoder to predict their original tokens.

In this exercise we train a small vocabulary and a small BERT on teaching text. Try a known phrase with one masked word and inspect its token IDs and candidate predictions. Save the tokenizer with the model. Reassigning an ID would cause the same number to name a different token.

Further reading

- [Devlin, J., et al. BERT. 2018](https://arxiv.org/abs/1810.04805)
- [Hugging Face. BERT API, Transformers 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/bert)
- [Hugging Face. Tokenizers quicktour](https://huggingface.co/docs/tokenizers/en/quicktour)
- [Hugging Face. Masked language modeling, Transformers 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/tasks/masked_language_modeling)

<a id="topic-budgets"></a>

## Bytes, memory budgets and unified memory

[Exercise 2](exercises/002.txt) · book page 27

One GiB is 1,073,741,824 bytes. The installed module in this machine is 64 GiB. Windows reports about 32.23 GiB visible to the OS and NVML reports a dedicated segment of 31,232 MiB, or 30.50 GiB (32.75 decimal GB). The two figures sum to about 62.73 GiB and have different accounting scopes. Guest MemTotal is a separate limit.

Record the API, units, time and process beside each value. CUDA reported 45.97 GiB across its dedicated and shareable budget. Adding the NVML and CUDA totals counts overlapping memory. Exercise 248 and the memory investigation explain the measured scopes. For an allocation decision query current free bytes and allow for working storage.

Further reading

- [NVIDIA. CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [NVIDIA. NVIDIA System Management Interface documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
- [PyTorch. CUDA semantics and memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
- [Linux Kernel. The proc filesystem and memory accounting](https://docs.kernel.org/filesystems/proc.html)
- [PyTorch. torch.cuda.memory.mem_get_info](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.mem_get_info.html)
- [NVIDIA. CUDA Programming Guide, unified and virtual memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)

<a id="topic-qdq"></a>

## Calibration and explicit quantize and dequantize operations

[Exercise 164](exercises/164.txt) · book page 517

A quantizer maps a range of real values to a limited set of codes using a scale and sometimes a zero point. Calibration estimates suitable ranges from example inputs. A rare large value can stretch a scale, leaving fewer useful codes for the many small values. Per-channel scales allow different ranges for different channels.

An ONNX QDQ graph marks where values are quantized and dequantized. Measure reconstruction error on separate evaluation rows. Then inspect the engine and its selected implementations before attributing a speed change to integer kernels.

Further reading

- [ONNX Runtime. Quantize ONNX models. Consulted 27 October 2026](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ONNX. QuantizeLinear operator. Consulted 27 October 2026](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html)

<a id="topic-obschecks"></a>

## Checking the telemetry itself

[Exercise 117](exercises/117.txt) · book page 391

A monitoring pipeline can lose or distort evidence. A duplicate event can double a count, a wrong clock unit can stretch a duration a thousandfold, and a missing span can make a request look shorter. First check identities, timestamps, units and the relationship between related records.

The exercise damages copies of a known trace and compares them with the unaltered directory. For aggregated latency, preserve both sum and count. A one-request group and a hundred-request group should contribute in proportion to their request counts when computing a combined mean.

Further reading

- [Python. time and monotonic clocks](https://docs.python.org/3/library/time.html)
- [OpenTelemetry. Python manual instrumentation](https://opentelemetry.io/docs/languages/python/instrumentation/)
- [Prometheus. Histograms and summaries. Consulted 28 October 2026](https://prometheus.io/docs/practices/histograms/)

<a id="topic-checkpoints"></a>

## Checkpoints, adapters and merged weights

[Exercise 27](exercises/027.txt) · book page 128

A checkpoint is saved state that a later process can open. An adapter may contain only changes to named layers and still require the exact base model. A merged checkpoint incorporates those changes into the weights intended for deployment. A file format such as safetensors stores arrays and their types.

A successful save and load proves that a representation could be read. To test numerical fidelity, use identical inputs before and after saving and compare the resulting scores, called logits, or the generated tokens. To resume training, also account for optimizer and other training state in Exercise 69.

Further reading

- [Hugging Face. PEFT LoRA documentation](https://huggingface.co/docs/peft/en/developer_guides/lora)
- [PyTorch. Saving and Loading a General Checkpoint](https://docs.pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
- [Hugging Face. Transformers Trainer and checkpoint continuation. Consulted 28 October 2026](https://huggingface.co/docs/transformers/main_classes/trainer)

<a id="topic-concurrency"></a>

## Client workers, queues and concurrency

[Exercise 21](exercises/021.txt) · book page 105

A worker handles a unit of work. Two clients can have requests in flight at the same time while one server processes them sequentially, interleaves them, or groups them. An open connection is therefore a different observation from GPU execution. More clients can increase memory use and time spent waiting.

Compare the same number of requests first with one worker and then with two. Record when each request enters, receives its first content and finishes. Changes in total throughput and in the latency experienced by an individual reader can point in different directions.

Further reading

- [ggml-org. llama.cpp HTTP server documentation, pinned tag b11898](https://github.com/ggml-org/llama.cpp/blob/b11898/tools/server/README.md)
- [NVIDIA. Triton dynamic batching. Consulted 27 October 2026](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
- [Python. ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor)

<a id="topic-http"></a>

## Clients, servers and a local HTTP request

[Exercise 3](exercises/003.txt) · book page 31

A server waits for requests. A client sends one to an address and a port. The address 127.0.0.1 means this machine’s loopback interface. An HTTP request names an operation and may carry a body. Here, JSON expresses that body as named fields, such as a model name and a prompt.

Exercise 3 uses a fixture, a small server with a known reply, to test the client before loading any weights. First read the HTTP status, then the body. A successful exchange tells us the two programs communicated. Grading a generated answer requires a separate check of its contents.

Further reading

- [ggml-org. llama.cpp HTTP server documentation, pinned tag b11898](https://github.com/ggml-org/llama.cpp/blob/b11898/tools/server/README.md)
- [Microsoft. Accessing network applications with WSL](https://learn.microsoft.com/en-us/windows/wsl/networking)
- [MDN Web Docs. Overview of HTTP](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview)

<a id="topic-build"></a>

## Compiling CUDA C++ for this GPU

[Exercise 32](exercises/032.txt) · book page 154

A CUDA C++ source file can contain code for the host CPU and functions called kernels for the GPU. nvcc coordinates the compilation steps. A build target tells the compiler which device architecture it should generate code for. The linker brings the compiled pieces and their libraries together as a program.

A debug build retains information that helps connect execution with source lines. Optimizations can change instructions and timing. Give the debug and optimized executables different names and run the same numerical input through both. Keep the compiler command with the result.

Further reading

- [NVIDIA. CUDA Compiler Driver NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
- [NVIDIA. CUDA Programming Guide. Intro to CUDA C++](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)
- [NVIDIA. CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

<a id="topic-compile"></a>

## Compiling PyTorch with Dynamo, Inductor and Triton

[Exercise 101](exercises/101.txt) · book page 352

In eager mode Python issues framework operations as execution reaches them. With torch.compile, TorchDynamo can capture compatible operations into a graph, and a backend such as TorchInductor can optimize and generate executable code. On GPUs that code may include kernels written using the Triton programming language. NVIDIA Triton Inference Server, used in Exercise 150, serves models over a network interface.

Read five calls over three input sizes. Inspect the compiler’s guards, recompilation and graph-break messages. Keep the initial compilation time separate from the timings of later calls. A graph break can return part of the computation to eager execution. A graph break ends a captured graph. Unsupported work can execute eagerly before capture resumes. A leading size of zero or one can change broadcasting and stride behavior, so a dynamic graph can still carry a lower-bound guard of two.

Further reading

- [PyTorch. Compiler observability and TORCH_LOGS, 2.14 guide](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.observability.html)
- [PyTorch. PyTorch compiler](https://docs.pytorch.org/docs/stable/torch.compiler.html)
- [PyTorch. Dynamic shapes and specialization of zero and one](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html)

<a id="topic-sanitizer"></a>

## Compute Sanitizer and CUDA correctness checks

[Exercise 37](exercises/037.txt) · book page 170

Compute Sanitizer observes a GPU program while it runs. Its tools check different properties. Memcheck examines memory accesses, racecheck detects supported shared-memory hazards, initcheck looks for uninitialized global-memory reads and synccheck examines synchronization use. Choose the tool that matches the question.

Compare an intentionally defective teaching case with the otherwise equivalent valid control. Record whether instrumentation started, the reported location and the process exit. A numerical pass with the tool unavailable still leaves the requested instrumentation check open.

Further reading

- [NVIDIA. Compute Sanitizer User Guide](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)

<a id="topic-multimediapipe"></a>

## Connecting speech recognition, generation and synthesis

[Exercise 202](exercises/202.txt) · book page 653

A speech assistant can contain three independent model calls. The recognizer produces text, the language model answers it, and the synthesizer produces speech from that answer. A plausible final recording can hide a transcription error or a changed instruction in an intermediate step.

Carry one request ID through all three stages. Retain the input audio identity, transcript, generated reply and output audio identity. Listen to the input before blaming the recognizer, and compare the reply with the text passed to synthesis. This lets you locate the first change that explains the final symptom.

Further reading

- [Hugging Face. MusicGen Transformers 4.57.1. Consulted 27 October 2026](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/musicgen)
- [Hugging Face. VITS Transformers 4.57.1. Consulted 27 October 2026](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/vits)
- [Hugging Face. Whisper Transformers 4.57.1. Consulted 27 October 2026](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/whisper)

<a id="topic-oci"></a>

## Container registries, manifests and platform selection

[Exercise 206](exercises/206.txt) · book page 670

An image tag can point to an index of builds for several platforms. An entry in that index identifies an operating system, an architecture and a manifest digest. The selected manifest in turn identifies the image’s contents. The ARM64 laptop needs the appropriate Linux ARM64 entry.

Keep the index and the selected manifest together. A digest identifies particular bytes even if a tag is moved later. Exercise 207 deliberately changes copies of these records. A missing platform, duplicate selection or mismatching digest can be diagnosed before pulling layers or launching a GPU.

Further reading

- [NVIDIA. NGC catalog user guide. Consulted 28 October 2026](https://docs.nvidia.com/ngc/latest/ngc-catalog-user-guide.html)
- [NVIDIA. PyTorch release 26.09. Consulted 28 October 2026](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-09.html)
- [Docker. Buildx imagetools inspect. Consulted 28 October 2026](https://docs.docker.com/reference/cli/docker/buildx/imagetools/inspect/)

<a id="topic-contrast"></a>

## Contrastive training for sentence encoders

[Exercise 86](exercises/086.txt) · book page 306

A pretrained token encoder can produce sentence vectors without having learned which incident descriptions should match a query. Contrastive training supplies that relationship. A positive pair consists of a query and its relevant passage. Other passages in the batch can act as negatives.

Rotating passages among unrelated incident families creates a control with deliberately wrong labels. Compare the training losses, then inspect the ranks on identical held-out queries. A small test set can reward a corrupted training run by chance. The checkpoint reload check asks a different question about preserving the selected vectors.

Further reading

- [Reimers, N., and Gurevych, I. Sentence-BERT. 2019](https://arxiv.org/abs/1908.10084)
- [Sentence Transformers. Loss functions](https://sbert.net/docs/package_reference/sentence_transformer/losses.html)
- [Sentence Transformers. Batch samplers](https://sbert.net/docs/package_reference/sentence_transformer/sampler.html)

<a id="topic-placement"></a>

## CPU execution and partial GPU offload

[Exercise 12](exercises/012.txt) · book page 69

A model is an ordered computation built from layers. A runtime can execute some layers on the GPU and others on the CPU. The boundary may require moving intermediate results. Decoder state, work buffers and the operating system also need memory while a request runs.

A CPU control runs the workload with GPU layers disabled. Compare it with a run that requests GPU layers. In both cases inspect the loader’s actual layer and buffer report, then submit the same prompt. The requested layer count alone cannot tell us what placement the runtime achieved.

Further reading

- [ggml-org. llama.cpp HTTP server documentation, pinned tag b11898](https://github.com/ggml-org/llama.cpp/blob/b11898/tools/server/README.md)
- [Hugging Face. Accelerate: Loading big models into memory](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference)
- [Qwen. Qwen2.5-32B-Instruct-GGUF model card and files](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF)

<a id="topic-toolkit"></a>

## The CUDA driver, toolkit and runtime

[Exercise 31](exercises/031.txt) · book page 151

CUDA is NVIDIA’s platform for running computations on a GPU. The driver exposes the device. The toolkit supplies development tools such as the nvcc compiler. A runtime library lets an application allocate device memory and launch work. A framework or container may carry its own user-space libraries.

Version labels at these layers can differ. Record the mapped driver query, nvcc version, PyTorch build and device compute capability in their own fields. A missing compiler affects building source. A working prebuilt application can still use the GPU.

Further reading

- [NVIDIA. CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [NVIDIA. CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [NVIDIA. CUDA Programming Guide: The CUDA platform](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html)
- [NVIDIA. CUDA Compiler Driver NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)

<a id="topic-streams"></a>

## CUDA streams, events and graphs

[Exercise 58](exercises/058.txt) · book page 227

A stream is an ordered queue of GPU work. Two streams may run work concurrently if dependencies and resources allow it. An event recorded after a producer lets a consumer stream wait for that producer. Double buffering lets one buffer accept new data while another is being processed.

A CUDA Graph records a set of operations and their dependencies for replay with lower submission overhead. Keep the test’s input shapes and storage addresses consistent with its capture. Verify the resulting values. Actual overlap between streams requires device intervals that overlap in time.

Further reading

- [NVIDIA. CUDA Programming Guide: Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [NVIDIA. CUDA Programming Guide. CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

<a id="topic-debugger"></a>

## CUDA-GDB, breakpoints and exception cores

[Exercise 35](exercises/035.txt) · book page 164

A debugger can stop an executing program at a breakpoint so you can inspect its current state. For CUDA, selecting a kernel, block and thread identifies whose registers and variables you are examining. Symbols connect source names and line information with a particular build.

An exception core permits later inspection of state that the platform captured at a fault. Preserve the matching executable and symbols with it. On this WSL2 configuration, building the program, attaching the debugger and producing a usable core are independent checks. Record the first operation that actually succeeded or stopped.

Further reading

- [NVIDIA. CUDA-GDB manual, WSL and GPU core dump sections](https://docs.nvidia.com/cuda/cuda-gdb/index.html#known-issues)

<a id="topic-operators"></a>

## Custom PyTorch operators and gradients

[Exercise 57](exercises/057.txt) · book page 225

A custom operator exposes compiled code as a named PyTorch operation. Its registration describes inputs and outputs. The dispatcher selects the appropriate implementation for a device. If the operation is to take part in training, PyTorch also needs a rule for propagating gradients backwards.

First call the operator from Python on a small tensor and compare its output against a familiar operation. Check registration and gradient behavior separately. Then run on the caller’s selected stream. An operation can return the right value on the default stream while mishandling ordering elsewhere.

Further reading

- [PyTorch. Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [PyTorch. torch.library, version 2.14](https://docs.pytorch.org/docs/2.14/library.html)
- [PyTorch. torch.utils.cpp_extension, version 2.14](https://docs.pytorch.org/docs/2.14/cpp_extension.html)

<a id="topic-parallel"></a>

## Data parallelism, sharding and gradient averaging

[Exercise 124](exercises/124.txt) · book page 408

In data parallelism, each worker owns a model replica and processes part of the batch. An all-reduce combines contributions across ranks. PyTorch DistributedDataParallel synchronizes gradients so replicas can take consistent updates. Unequal local batch sizes require care when relating those updates to a full-batch reference. DistributedDataParallel (DDP) averages gradients across participating ranks. Fully Sharded Data Parallel (FSDP) partitions training state. With unequal example counts, an average of rank means needs sample-count weighting to recover the global example mean.

Sharding divides storage or computation, such as assigning different columns of a matrix to different workers. Fully Sharded Data Parallel uses sharding for training state. Compare a small partitioned calculation and its gradients with an unpartitioned reference. That mathematical check can run on CPU before any multi-GPU communication test.

Further reading

- [PyTorch. Distributed communication package and debugging, version 2.14](https://docs.pytorch.org/docs/2.14/distributed.html)
- [PyTorch. DistributedDataParallel, version 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch. FSDP2 fully_shard, version 2.14](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html)
- [NVIDIA. NCCL Creating a Communicator, consulted 27 October 2026](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)

<a id="topic-supply"></a>

## DataLoader, queues and prefetching

[Exercise 103](exercises/103.txt) · book page 359

Training often needs the CPU to read, decode and assemble examples before the GPU can use them. PyTorch’s DataLoader makes batches and can use worker processes. Prefetching lets the producer prepare future batches while the consumer processes the current one. Buffer space and memory place a limit on how much work can wait.

Introduce a known delay before a batch is ready. Measure time waiting for input separately from computation. Then change the queue depth while keeping the work fixed. Low GPU utilization alone cannot tell input starvation from a small or intermittent workload.

Further reading

- [PyTorch. DataLoader and data processing, version 2.14](https://docs.pytorch.org/docs/2.14/data.html)
- [PyTorch. Multiprocessing best practices, version 2.14](https://docs.pytorch.org/docs/2.14/notes/multiprocessing.html)
- [PyTorch data loading with worker processes. See [85]](https://docs.pytorch.org/docs/2.14/data.html)

<a id="topic-splits"></a>

## Datasets, labels and independent evaluation

[Exercise 24](exercises/024.txt) · book page 119

A dataset contains examples. In supervised training an example pairs an input with a desired target. The training set updates the model. A validation or development set helps select settings or checkpoints. A held-out test set estimates how the selected model behaves on unseen examples.

If two prompts paraphrase the same incident, putting one in training and the other in test can make generalization look better than it is. Split by incident family before generating variants. JSON Lines stores one JSON record per line, so you can inspect the family identifier, input and target together.

Further reading

- [Hugging Face. Dataset cards. Consulted 28 October 2026](https://huggingface.co/docs/hub/main/datasets-cards)
- [Hugging Face. lhoestq demo1 dataset at the selected revision. Consulted 28 October 2026](https://huggingface.co/datasets/lhoestq/demo1/tree/87ecf163bedca9d80598b528940a9c4f99e14c11)

<a id="topic-calibration"></a>

## Decision calibration, risk and coverage

[Exercise 233](exercises/233.txt) · book page 750

A probability forecast can be checked against labeled outcomes. For a binary event with outcome one and predicted probability 0.8, the squared error is 0.04. Averaging squared probability errors gives the binary Brier score. A single confident prediction cannot establish that a model is calibrated.

A routing system can abstain below a confidence threshold. Coverage is the fraction of cases it accepts. Selective risk describes the error among accepted cases. Count both accepted cases and mistakes at each threshold. If none were accepted, the error fraction among them has no denominator.

Further reading

- [TypeSafe. Confidence definitions. Consulted 28 October 2026](https://docs.typesafe.ai/confidence)

<a id="topic-dumps"></a>

## Device buffer exports, allocator snapshots and GPU core dumps

[Exercise 34](exercises/034.txt) · book page 160

Each of these artifacts answers a different question. A buffer dump copies bytes from a selected allocation after the program finishes writing it. To decode the bytes you need their element type, shape and byte order. A framework allocator snapshot describes allocations and their histories. A GPU exception core records supported debugging state when execution faults.

Begin with a buffer initialized to a known pattern. Synchronize, export just that allocation, decode a few elements and compare them with the pattern. A matching payload checks this buffer at that point in the program.

Further reading

- [NVIDIA. CUDA Runtime API: Memory Management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html)
- [NVIDIA. CUDA-GDB manual, WSL and GPU core dump sections](https://docs.nvidia.com/cuda/cuda-gdb/index.html#known-issues)
- [PyTorch. Understanding CUDA Memory Usage, version 2.14](https://docs.pytorch.org/docs/2.14/torch_cuda_memory.html)

<a id="topic-reasoning"></a>

## Diagnostic observations, controls and written conclusions

[Exercise 13](exercises/013.txt) · book page 75

A diagnostic observation is something a reader can check in an artifact, such as a response status or a timestamp. A hypothesis proposes an explanation. A control changes or removes a suspected factor so that competing explanations predict different observations. An oracle supplies an expected result against which a computation can be checked.

For the fictional R17 incident, start with one observed event. Name two explanations consistent with it, then say what you would measure to tell them apart. The remaining uncertainty is part of the conclusion. This same method applies later to kernels, generated images and stalled workers.

<a id="topic-casebook"></a>

## Diagnostic patterns and evidence cards

[Exercise 70](exercises/070.txt) · book page 259

A diagnostic pattern names a recurring structure in evidence, such as a worker repeatedly waiting on an empty queue. The same visible symptom can have several causes. A pattern suggests a question and a discriminating check. It becomes useful when you can point to the artifact that supports its application.

For an evidence card, identify the observed state first and write your competing explanation before revealing the worked answer. A deliberately injected delay can test how your method responds to a known fault. Keep that distinction when you later investigate a real stall with an unknown cause.

<a id="topic-diffusion"></a>

## Diffusion, noise prediction and a sampling schedule

[Exercise 180](exercises/180.txt) · book page 578

In the epsilon-prediction control, clean x₀ and sampled noise ε are mixed at time t as xₜ = √ᾱₜ x₀ + √(1 − ᾱₜ) ε. Alpha-bar is the cumulative product of the schedule’s retained-signal factors. A model learns to estimate ε. With the exact noise, recover x₀ by subtracting √(1 − ᾱₜ) ε from xₜ and dividing by √ᾱₜ.

The tiny point experiment deliberately uses known clean values and known noise. Substitute that same noise into the reconstruction formula and compare the recovered points with the originals. Replacing it with wrong noise should disturb the answer. This tests the arithmetic before any learned denoiser or expensive image generation is involved.

Further reading

- [Hugging Face. Stable Diffusion XL Diffusers 0.35.1. Consulted 27 October 2026](https://huggingface.co/docs/diffusers/v0.35.1/en/using-diffusers/sdxl)
- [Jonathan Ho Ajay Jain and Pieter Abbeel. Denoising Diffusion Probabilistic Models 2020. Consulted 27 October 2026](https://arxiv.org/abs/2006.11239)

<a id="topic-dist"></a>

## Distributed training on one physical laptop

[Exercise 123](exercises/123.txt) · book page 407

Distributed training uses several processes that coordinate their work. Each process has a rank, its identifier within a group. World size is the number of participating processes. A rendezvous brings those ranks together. Gloo and NVIDIA’s NCCL are communication backends used in different configurations.

On one laptop, two CPU processes can act as two ranks over loopback and expose coordination, delays or failures. They still share one physical machine. A one-rank NCCL check can exercise the available GPU. Two-device training requires two independently addressable CUDA devices.

Further reading

- [PyTorch. Distributed communication package and debugging, version 2.14](https://docs.pytorch.org/docs/2.14/distributed.html)
- [PyTorch. torchrun Elastic Launch, version 2.14](https://docs.pytorch.org/docs/2.14/elastic/run.html)
- [PyTorch. DistributedDataParallel, version 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)
- [NVIDIA. NCCL Creating a Communicator, consulted 27 October 2026](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)

<a id="topic-compose"></a>

## Docker Compose and service networking

[Exercise 141](exercises/141.txt) · book page 451

Docker Compose describes related containers, their configuration, networks and mounts in a YAML file. One service can refer to another by its service name on a shared Compose network. Inside a container, 127.0.0.1 identifies that container’s own loopback interface in the usual network configuration.

Follow one request from its caller to the declared port and destination service. Check the effective configuration and lifecycle events if the service exits. Record whether a model file should come from the image or a mount. A missing mount can make a healthy executable fail before inference starts.

Further reading

- [Docker. Host networking driver. Consulted 28 October 2026](https://docs.docker.com/engine/network/drivers/host/)
- [Docker. The Compose application model](https://docs.docker.com/compose/intro/compose-application-model/)

<a id="topic-docker"></a>

## Docker images, containers and volumes

[Exercise 140](exercises/140.txt) · book page 447

An image packages a program and its user-space files. A Dockerfile describes how to build that image. A container is a running instance with its own process and filesystem view, using the host kernel. A volume or bind mount makes selected files, such as model weights, available inside that view.

Building an image checks the build instructions. Starting a container also tests its entry command, permissions and configuration. Publish the intended port, send one small request and retain the container’s logs and exit status. A tag is a convenient name. A digest identifies particular image content. Here the container host shares the WSL guest’s kernel. A container memory limit above the measured 15.70 GiB guest RAM cannot grant additional physical guest RAM. Swap policy and current availability also matter.

Further reading

- [Docker. Install Docker Engine on Ubuntu. Consulted 27 October 2026](https://docs.docker.com/engine/install/ubuntu/)
- [ggml-org. llama.cpp Docker documentation. Consulted 27 October 2026](https://github.com/ggml-org/llama.cpp/blob/b11898/docs/docker.md)
- [Docker. What is a container](https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-a-container/)

<a id="topic-batching"></a>

## Dynamic batching and bounded admission

[Exercise 132](exercises/132.txt) · book page 429

A batch groups compatible inputs into one model operation. Dynamic batching collects requests that arrive separately and can briefly wait for more work before dispatch. This can improve throughput while increasing the wait for an individual request. A bounded queue limits how much pending work the server admits.

Give every request its own ID. On overload, follow it from submission to rejection, cancellation or exactly one completed response. Compare queue time with model execution time. A request that timed out at the client may still occupy the server until cancellation reaches the owner.

Further reading

- [vLLM. Production metrics. Consulted 27 October 2026](https://github.com/vllm-project/vllm/blob/main/docs/usage/metrics.md)
- [NVIDIA. Triton dynamic batching. Consulted 27 October 2026](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)

<a id="topic-lora"></a>

## Fine-tuning, LoRA and trainable parameters

[Exercise 25](exercises/025.txt) · book page 121

In a training step the model makes a prediction. A loss measures its disagreement with the target. Backpropagation computes gradients, and an optimizer uses them to adjust trainable parameters. Fine-tuning starts from a pretrained model. The Transformers library loads the model and tokenizer, while PEFT provides parameter-efficient training methods.

LoRA freezes the pretrained weights in this exercise. For each selected matrix, it learns two new factors with inner dimension r. With input width d and output width k, the two factors train r × (d + k) parameters. The update is scaled by alpha/r, here 16/8 = 2. The 28 layers’ query and value projections account for 1,089,536 trainable parameters. Keep the factors, tokenizer identity and exact base revision together.

Further reading

- [Hu, E. J., et al. LoRA: Low-Rank Adaptation of Large Language Models. 2021](https://arxiv.org/abs/2106.09685)
- [Dettmers, T., et al. QLoRA: Efficient Finetuning of Quantized LLMs. 2023](https://arxiv.org/abs/2305.14314)
- [Hugging Face. PEFT LoRA documentation](https://huggingface.co/docs/peft/en/developer_guides/lora)

<a id="topic-precision"></a>

## Floating-point precision and numerical tolerances

[Exercise 99](exercises/099.txt) · book page 347

Floating-point formats allocate bits to sign, exponent and fraction. FP16 has five exponent and ten fraction bits, with largest finite value 65,504. BF16 (bfloat16) has eight exponent and seven fraction bits. FP32 has eight exponent and 23 fraction bits. BF16 covers a range similar to FP32 with coarser spacing between adjacent values.

An absolute tolerance bounds a difference in value. A relative tolerance scales with the reference magnitude. Near zero, the absolute term is especially important. Compare intermediate values on matched inputs to find the first divergence. A final predicted class can still match after an earlier numerical error.

Further reading

- [NVIDIA. Floating Point and IEEE 754](https://docs.nvidia.com/cuda/floating-point/index.html)
- [PyTorch. Numerical accuracy, version 2.14](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
- [PyTorch. Tensor data types](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch-dtype)

<a id="topic-identifiers"></a>

## Git repositories, snapshots and saved run directories

[Exercise 4](exercises/004.txt) · book page 34

Git records versions of a project. A commit identifies one saved version. GitHub hosts the companion repository. A clone gives you its files and Git history. A release ZIP gives you a snapshot you can use without learning Git commands. The companion supplies the small programs and teaching inputs. Large downloaded model weights have their own identities.

Keep the staging copy apart from your working lab. A fresh run directory holds the command, results and logs from one attempt. When comparing two trials, keep both directories and note the software revision used for each.

Further reading

- [Microsoft. Working across Windows and Linux file systems](https://learn.microsoft.com/en-us/windows/wsl/filesystems)
- [Hugging Face. Hub overview. Consulted 28 October 2026](https://huggingface.co/docs/hub/index)
- [Scott Chacon and Ben Straub. Pro Git, Getting Started with Git](https://git-scm.com/book/en/v2/Getting-Started-What-is-Git%3F)

<a id="topic-metrics"></a>

## GPU metrics, samples and missing observations

[Exercise 91](exercises/091.txt) · book page 323

The NVIDIA Management Library (NVML) reports supported sampled device fields. GPU utilization is the fraction of the device sample period during which one or more kernels ran. Memory utilization measures the fraction of that period in which device memory was read or written. A small continuously active kernel can register high busy time. Occupancy, throughput and bytes per second require their own measurements.

Start with the idle baseline and note the field name, unit, timestamp, sample interval and provider. Preserve an unavailable field as missing. Zero is a measured value. The distinction matters when the dashboard shows a flat line during an otherwise successful request.

Further reading

- [NVIDIA. NVIDIA System Management Interface documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
- [NVIDIA. Useful nvidia-smi Queries](https://nvidia.custhelp.com/app/answers/detail/a_id/3751)
- [NVIDIA. CUDA on WSL User Guide and NVML limitations](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [NVIDIA. NVML utilization field definitions](https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html)

<a id="topic-scheduling"></a>

## GPU scheduling in a Kubernetes cluster

[Exercise 146](exercises/146.txt) · book page 463

Kubernetes can schedule a GPU request only when a node advertises the corresponding resource. An NVIDIA device plugin reports supported devices to Kubernetes. The scheduler selects a suitable node, and the container runtime must then expose the assigned device to the process.

Start with the node’s allocatable resource, then read the Pod’s resource request and scheduling events. Once it starts, verify CUDA from inside that Pod. Replicating a Deployment creates additional Pods. It cannot create additional physical GPUs on a one-device laptop.

Further reading

- [NVIDIA. Container Toolkit installation guide. Consulted 27 October 2026](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Kubernetes. Schedule GPUs. Consulted 27 October 2026](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [NVIDIA. Kubernetes device plugin and releases. Consulted 27 October 2026](https://github.com/NVIDIA/k8s-device-plugin)

<a id="topic-trainingstate"></a>

## Gradients, optimizers and resuming an update

[Exercise 68](exercises/068.txt) · book page 254

A gradient tells an optimizer how the loss changes as a trainable value changes. AdamW also carries running estimates derived from earlier gradients. A training process may keep random-number state, a batch position and a scaler for limited-precision arithmetic. Live tensors retained by Python can change the memory picture.

To test recovery, save at a known step, resume in a new process and compare the next batch, loss and updated parameters with an uninterrupted run. Restoring only the visible model weights can produce the same immediate prediction yet a different next update.

Further reading

- [PyTorch. torch.utils.checkpoint, version 2.14](https://docs.pytorch.org/docs/2.14/checkpoint.html)
- [PyTorch. Automatic Mixed Precision package, version 2.14](https://docs.pytorch.org/docs/2.14/amp.html)
- [PyTorch. Saving and Loading a General Checkpoint](https://docs.pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
- [PyTorch. Automatic differentiation with torch.autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)

<a id="topic-grafana"></a>

## Grafana dashboards and data sources

[Exercise 215](exercises/215.txt) · book page 697

Grafana displays results queried from a data source, here Prometheus. A dashboard contains panels. Each panel applies a query, a time range and display settings such as units. Provisioning describes these connections and panels in files that another installation can reproduce.

Choose one GPU memory panel. Read its data source, expression and unit, run that expression directly in Prometheus and compare the same time window. An empty panel can result from a label mismatch or an unsuitable time range even when the exporter has working data.

Further reading

- [Grafana Labs. Grafana OSS 13.3.0 download. Consulted 28 October 2026](https://grafana.com/grafana/download/13.3.0?edition=oss&platform=linux)
- [Grafana Labs. Provision Grafana data sources and dashboards. Consulted 28 October 2026](https://grafana.com/docs/grafana/latest/administration/provisioning/)
- [Grafana Labs. HTTP API and health endpoint. Consulted 28 October 2026](https://grafana.com/docs/grafana/latest/developer-resources/api-reference/http-api/api-legacy/other/)

<a id="topic-grounding"></a>

## Grounding a model in retrieved evidence

[Exercise 15](exercises/015.txt) · book page 89

A language model can answer from the text supplied in its prompt. Grounding means requiring its factual claims to agree with identifiable evidence. A retrieval step can select passages before generation. Together these operations are often called retrieval-augmented generation, or RAG. A small lab can select evidence directly from local files.

Take one claim in the answer and find the exact log line that supports it. If there is no such line, describe the claim as a hypothesis. This works even before a vector search engine is introduced.

Further reading

- [Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y. ReAct Synergizing Reasoning and Acting in Language Models. ICLR 2023](https://arxiv.org/abs/2210.03629)
- [Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

<a id="topic-paging"></a>

## Guest paging, Windows commit and vmstat

[Exercise 30](exercises/030.txt) · book page 143

When memory is under pressure the operating system may move pages between RAM and backing storage. In WSL2, vmstat reports guest activity. Its si and so columns show swap-in and swap-out rates. The first rate row normally summarizes time since boot. Later rows cover sampling intervals.

Windows available physical memory and Windows commit headroom answer other questions. Commit headroom is the difference between the commit limit and committed bytes. Match timestamps from both systems to the actual request interval. Used swap by itself cannot establish that a slow request was paging.

Further reading

- [Linux Kernel. The proc filesystem and memory accounting](https://docs.kernel.org/filesystems/proc.html)
- [PyTorch. torch.cuda.memory.mem_get_info](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.mem_get_info.html)
- [procps-ng. free(1)](https://man7.org/linux/man-pages/man1/free.1.html)
- [procps-ng. vmstat(8)](https://man7.org/linux/man-pages/man8/vmstat.8.html)
- [Microsoft. PERFORMANCE_INFORMATION structure and system commit counters](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-performance_information)

<a id="topic-hfhub"></a>

## Hugging Face Hub, snapshots and dataset cards

[Exercise 208](exercises/208.txt) · book page 679

Hugging Face Hub hosts versioned repositories for models, datasets and other artifacts. A model card documents a model’s purpose and usage. A dataset card describes data and its intended scope. A snapshot collects files at one revision. A moving branch name can later select different bytes.

Inspect a revision and its file list before downloading. A tiny encoder may need configuration, a tokenizer and weights. For a dataset, check the actual rows and split membership as well as the card. If access is gated, authorization is a separate prerequisite from the ability to load the files.

Further reading

- [Hugging Face. Hub overview. Consulted 28 October 2026](https://huggingface.co/docs/hub/index)
- [Hugging Face. Model cards. Consulted 28 October 2026](https://huggingface.co/docs/hub/en/model-cards)
- [Hugging Face. Dataset cards. Consulted 28 October 2026](https://huggingface.co/docs/hub/main/datasets-cards)
- [Hugging Face. Hub download API 0.36.0. Consulted 28 October 2026](https://huggingface.co/docs/huggingface_hub/v0.36.0/package_reference/file_download)
- [Hugging Face. Hub cache behavior 0.36.0. Consulted 28 October 2026](https://huggingface.co/docs/huggingface_hub/v0.36.0/en/guides/manage-cache)

<a id="topic-media"></a>

## Images, waveforms and model processors

[Exercise 172](exercises/172.txt) · book page 556

An RGB image can be represented as a grid with three color values at each position. A waveform is a sequence of sample values taken at a known rate. At 16,000 samples per second, 32,000 mono samples represent two seconds. Image dimensions, audio rate and channel count are part of the input.

Pillow reads image files into pixel arrays, while SciPy supplies tools for audio signals. A model processor converts such files into the tensors the model expects. Its resizing, normalization or resampling settings affect those tensors. Use a known shape image or short tone to check the file, decoded values and processing settings before interpreting a generated caption or transcript.

Further reading

- [Hugging Face. Qwen2.5 VL Transformers 4.57.1. Consulted 27 October 2026](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_vl)
- [Hugging Face. VITS Transformers 4.57.1. Consulted 27 October 2026](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/vits)
- [Hugging Face. Whisper Transformers 4.57.1. Consulted 27 October 2026](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/whisper)
- [SciPy. Polyphase resampling. Consulted 27 October 2026](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)

<a id="topic-serving"></a>

## Inference APIs, readiness and service health

[Exercise 136](exercises/136.txt) · book page 439

An inference service turns a model call into a request that another program can make. Its contract specifies the input fields, supported shapes, response structure and error behavior. A vector endpoint, for example, should return one vector of the expected length for each accepted input.

A liveness check asks whether the process is functioning. A readiness check asks whether it can serve the intended workload now. Draining stops accepting new work while allowing in-flight requests to finish. Try a known request and an invalid one, then check which process, status and response correspond to each.

Further reading

- [ggml-org. llama.cpp HTTP server documentation, pinned tag b11898](https://github.com/ggml-org/llama.cpp/blob/b11898/tools/server/README.md)
- [Kubernetes. Liveness readiness and startup probes. Consulted 27 October 2026](https://kubernetes.io/docs/concepts/workloads/pods/probes/)
- [MDN Web Docs. Overview of HTTP](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview)

<a id="topic-jev"></a>

## Jev decision models and typed questions

[Exercise 232](exercises/232.txt) · book page 748

Jev takes supplied state and typed questions about it. Choice selects from named options. Score represents a result on an ordered rubric. Noul answers a yes-or-no question with a probability of yes. The response fields differ, so the caller must use the contract for the particular question type.

Imagine routing an incident to storage or networking. Compare the returned label with its distribution, then inspect the separate confidence field where the contract supplies one. The local fixtures exercise how the client checks these responses. The optional hosted-service procedure has its own execution status.

Further reading

- [TypeSafe. Jev models and version aliases. Consulted 28 October 2026](https://docs.typesafe.ai/models)
- [TypeSafe. Official OpenAPI request, response and header schema. Consulted 28 October 2026](https://api.typesafe.ai/openapi.json)
- [TypeSafe. Choice primitive. Consulted 28 October 2026](https://docs.typesafe.ai/primitives/choice)
- [TypeSafe. Score primitive. Consulted 28 October 2026](https://docs.typesafe.ai/primitives/score)
- [TypeSafe. Noul primitive. Consulted 28 October 2026](https://docs.typesafe.ai/primitives/noul)
- [TypeSafe. Confidence definitions. Consulted 28 October 2026](https://docs.typesafe.ai/confidence)

<a id="topic-fusion"></a>

## Kernel fusion and CUDA event timing

[Exercise 53](exercises/053.txt) · book page 212

Kernel fusion combines operations that would otherwise run as separate kernels. For example, an addition followed by an activation may reuse a value already in a register, avoiding a write to and a read from an intermediate array. It can also change resource use or arithmetic order.

Compare the final arrays first. For timing, record events in the stream that owns the work and wait for the ending event before reading elapsed time. A CPU clock around an asynchronous launch can measure little more than submission. Keep warm-up and measured iterations identifiable.

Further reading

- [NVIDIA. CUDA Programming Guide: Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [NVIDIA. CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA. Nsight Compute Profiling Guide, roofline charts](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

<a id="topic-kube"></a>

## Kubernetes, K3s, Pods and Services

[Exercise 144](exercises/144.txt) · book page 459

Kubernetes runs workloads from a declared desired state. K3s is the Kubernetes distribution used for the local cluster. A Pod runs one or more containers together. A Deployment manages replicas and replacements. A Service gives clients an address that can route requests to matching ready Pods. kubectl is the command-line client.

Follow the chain from Deployment to Pod to ready endpoint and, finally, one inference reply. If a rollout stalls, inspect the Pod’s events, logs and readiness check. A created Deployment confirms that Kubernetes accepted the description; the request tests whether the model can serve.

Further reading

- [K3s. Quick-start guide. Consulted 27 October 2026](https://docs.k3s.io/quick-start)
- [K3s. Advanced options and alternative container runtimes. Consulted 27 October 2026](https://docs.k3s.io/advanced)
- [Kubernetes. Liveness readiness and startup probes. Consulted 27 October 2026](https://kubernetes.io/docs/concepts/workloads/pods/probes/)
- [Kubernetes. Deployments and rollout strategies. Consulted 27 October 2026](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)

<a id="topic-langchain"></a>

## LangChain agents and local model adapters

[Exercise 220](exercises/220.txt) · book page 721

LangChain supplies interfaces for chat models and tools and can assemble them into an agent loop. A model adapter translates between the framework’s messages and a provider’s request and response format. The application still decides which tools are available and validates the arguments it receives.

Begin with a known tool proposal and watch the controller record its result. Then replace the test responder with the local model endpoint. An endpoint can be compatible with ordinary chat while handling tool calls differently. Inspect the actual structured call before attributing a working tool loop to that model.

Further reading

- [LangChain. Agents. Consulted 28 October 2026](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain. ChatOpenAI integration and compatible API scope. Consulted 28 October 2026](https://docs.langchain.com/oss/python/integrations/chat/openai)
- [vLLM. Tool calling and parser qualification. Consulted 28 October 2026](https://docs.vllm.ai/en/latest/features/tool_calling/)

<a id="topic-langgraph"></a>

## LangGraph state, checkpoints and human review

[Exercise 223](exercises/223.txt) · book page 729

LangGraph represents an agent workflow as a graph. A node performs a step, an edge selects the next step and shared state carries information between them. A checkpointer saves state for a particular thread identifier. An interrupt can pause execution until the application supplies a review decision.

In the exercise a new Python process opens the saved state and resumes the intended thread. Check the thread ID, pending action and state after resumption. A resumed node can re-execute earlier code, so an external action needs its own record to prevent doing it twice.

Further reading

- [LangChain. LangGraph persistence. Consulted 28 October 2026](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangChain. LangGraph interrupts and replay. Consulted 28 October 2026](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangChain. SqliteSaver.from_conn_string. Consulted 28 October 2026](https://reference.langchain.com/python/langgraph.checkpoint.sqlite/SqliteSaver/from_conn_string)

<a id="topic-langsmith"></a>

## LangSmith tracing and evaluation

[Exercise 226](exercises/226.txt) · book page 736

LangSmith provides tracing and evaluation for model applications. A trace can group a request, model call and tool use as related runs. An evaluator applies a defined test to an output, perhaps comparing it with a reference or checking that a required field is present. Preserve the inputs and criterion behind each score.

The companion has offline controls for nested records, errors and redaction. A local trace or a prepared upload is evidence about the client. Observing a hosted run additionally requires an authorized send and readback. Each exercise identifies which of those steps actually took place.

Further reading

- [LangChain. LangSmith custom instrumentation. Consulted 28 October 2026](https://docs.langchain.com/langsmith/annotate-code)
- [LangChain. LangSmith tracing_context API. Consulted 28 October 2026](https://reference.langchain.com/python/langsmith/run_helpers/tracing_context)
- [LangChain. LangSmith evaluate API, upload_results option. Consulted 28 October 2026](https://reference.langchain.com/python/langsmith/evaluation/_runner/evaluate)
- [LangChain. Self-hosted LangSmith requirements. Consulted 28 October 2026](https://docs.langchain.com/langsmith/self-hosted)

<a id="topic-timing"></a>

## Latency, throughput and repeatable timing

[Exercise 7](exercises/007.txt) · book page 49

Latency measures the elapsed time for one operation. Throughput measures completed work per unit of time. For an arithmetic example, 20 generated tokens in two seconds is 10 tokens per second for that measured interval. A client timing the whole request also includes waiting, prompt processing and transport.

Use a monotonic clock for durations. Its readings never go backwards and changing the wall clock does not shift its origin. Record wall-clock timestamps for correlation between records. For nearest-rank p95, sort n observations and select rank ceil(0.95 × n), counting from one. For five requests this selects the slowest. Report individual timings for short runs.

Further reading

- [ggml-org. llama.cpp HTTP server documentation, pinned tag b11898](https://github.com/ggml-org/llama.cpp/blob/b11898/tools/server/README.md)
- [Python. time and monotonic clocks](https://docs.python.org/3/library/time.html)
- [Python. Monotonic clocks and wall time](https://docs.python.org/3/library/time.html#time.monotonic)

<a id="topic-lmstudio"></a>

## LM Studio, llmster and the lms command line

[Exercise 243](exercises/243.txt) · book page 788

LM Studio can run a headless service called llmster. In this book that service and its lms command-line client run in WSL2. Importing a model makes its files available to the runtime. Loading creates an active model instance. The server then exposes an endpoint that a separate client can call.

Compare the catalog of downloaded models with the list of loaded instances before and after a request. A first request can include just-in-time loading. Record that separately from a later warm request and check that the reply identifies the model you intended.

Further reading

- [LM Studio. Headless service and llmster. Consulted 28 October 2026](https://lmstudio.ai/docs/developer/core/headless)
- [LM Studio. Runtime management CLI. Consulted 28 October 2026](https://lmstudio.ai/docs/cli/runtime/runtime)
- [LM Studio. Importing local model files. Consulted 28 October 2026](https://lmstudio.ai/docs/cli/local-models/import)
- [LM Studio. Local server bind and port. Consulted 28 October 2026](https://lmstudio.ai/docs/cli/serve/server-start)
- [LM Studio. OpenAI-compatible endpoints. Consulted 28 October 2026](https://lmstudio.ai/docs/developer/openai-compat)

<a id="topic-bigmodel"></a>

## Loading a model larger than the GPU budget

[Exercise 29](exercises/029.txt) · book page 135

The “32B” in this model’s name describes roughly 32 billion parameters. Storage depends on how each number is represented. The nine Q8 shards in this exercise total 32.43 GiB. Loading also requires working buffers and room for the request. The loader can keep some layers on the CPU while placing others on the GPU.

A guard samples available memory and can stop a load before it reaches readiness. Treat that stop as an outcome with a timestamp and a threshold. A useful successful run records the actual layer split, replies completed and memory observations over the same interval.

Further reading

- [ggml-org. llama.cpp HTTP server documentation, pinned tag b11898](https://github.com/ggml-org/llama.cpp/blob/b11898/tools/server/README.md)
- [Hugging Face. Accelerate: Loading big models into memory](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference)
- [Qwen. Qwen2.5-32B-Instruct-GGUF model card and files](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF)
- [NVIDIA. CUDA on WSL User Guide and NVML limitations](https://docs.nvidia.com/cuda/wsl-user-guide/)

<a id="topic-server"></a>

## Loading a model with llama.cpp

[Exercise 5](exercises/005.txt) · book page 40

llama.cpp is a native inference runtime. Its llama-server executable opens the model file, prepares memory for inference and listens for requests. The file on disk, the running process and the address clients call are three different things. A downloaded model alone cannot answer a prompt.

Follow the startup log in order. Find the model identity, the CPU or GPU placement of layers, and the readiness message. Then send the small request. Record the process ID with the log so a later restart or shutdown can be associated with this particular server. Here -b is the maximum logical token batch submitted to llama.cpp and -ub bounds the physical microbatch processed together. Training uses microbatch for the example group contributing one accumulated gradient.

Further reading

- [ggml-org. llama.cpp HTTP server documentation, pinned tag b11898](https://github.com/ggml-org/llama.cpp/blob/b11898/tools/server/README.md)
- [ggml-org. llama.cpp release b11898](https://github.com/ggml-org/llama.cpp/releases/tag/b11898)
- [ggml-org. llama.cpp option registration, pinned tag b11898](https://github.com/ggml-org/llama.cpp/blob/b11898/common/arg.cpp)

<a id="topic-tiles"></a>

## Matrix layout, tiles and shared memory

[Exercise 47](exercises/047.txt) · book page 198

A matrix has rows and columns but occupies a sequence of memory locations. In row-major storage, adjacent columns of one row sit beside each other. A transpose swaps rows with columns. For a rectangular input, reversing the shape as well as the indices helps expose a mistaken stride.

A tile is a small rectangular portion that a block processes together. Threads can stage a tile in on-chip shared memory and synchronize before consuming one another’s values. Check the last partial tile: threads without a valid output may still need to participate in the block’s barrier.

Further reading

- [NVIDIA. CUDA Programming Guide. Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
- [NVIDIA. CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

<a id="topic-matmul"></a>

## Matrix multiplication on the GPU

[Exercise 51](exercises/051.txt) · book page 207

Each entry of a matrix product is a dot product. Multiply elements from one row of the left matrix by the corresponding elements of one column of the right matrix and add them. An input with shape (2, 3) can multiply one with shape (3, 4) to produce a (2, 4) result.

A simple kernel assigns one output entry to each thread. A tiled kernel reuses input values across a block to reduce repeated global-memory reads. Start with an asymmetric rectangular example and a separate CPU calculation before timing the optimized variant.

Further reading

- [NVIDIA. CUDA Programming Guide. Intro to CUDA C++](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)
- [NVIDIA. CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA. cuBLAS Library documentation](https://docs.nvidia.com/cuda/cublas/index.html)

<a id="topic-offline"></a>

## Model caches, local files and offline execution

[Exercise 22](exercises/022.txt) · book page 110

Acquisition copies a model, tokenizer and configuration onto local storage. A cache keeps downloaded files for reuse. Inference reads those files and executes the model. Some libraries still check a remote registry while opening a cached model unless configured for local loading.

Use the pinned revision, record which files are present and state exactly which process or network access you constrained. A successful local load establishes that the chosen files satisfied that loading path. Testing a fully disconnected laptop requires an actual request under that wider condition.

Further reading

- [Hugging Face. Hub overview. Consulted 28 October 2026](https://huggingface.co/docs/hub/index)
- [Hugging Face. Hub download API 0.36.0. Consulted 28 October 2026](https://huggingface.co/docs/huggingface_hub/v0.36.0/package_reference/file_download)
- [Hugging Face. Hub cache behavior 0.36.0. Consulted 28 October 2026](https://huggingface.co/docs/huggingface_hub/v0.36.0/en/guides/manage-cache)
- [Hugging Face. Hub environment variables 0.36.0. Consulted 28 October 2026](https://huggingface.co/docs/huggingface_hub/v0.36.0/package_reference/environment_variables)

<a id="topic-files"></a>

## Model downloads, Git revisions and GGUF

[Exercise 4](exercises/004.txt) · book page 34

A model repository holds weights, configuration and a description called a model card. A revision identifies a particular state of that repository. A SHA-256 digest is a fingerprint of a file’s bytes. Compare it with the expected digest before loading. Two similarly named files may contain different weights.

The early exercises use Qwen models. Their GGUF files contain tensors and metadata in a format the llama.cpp runtime can read. A large model can be split into numbered shards. Keep the runtime identity, model revision and complete shard list together so another reader can reconstruct the same load.

Further reading

- [Qwen. Qwen2.5-7B-Instruct-GGUF model card and files](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)
- [ggml-org. llama.cpp release b11898](https://github.com/ggml-org/llama.cpp/releases/tag/b11898)
- [Hugging Face. Hub overview. Consulted 28 October 2026](https://huggingface.co/docs/hub/index)
- [Hugging Face. Model cards. Consulted 28 October 2026](https://huggingface.co/docs/hub/en/model-cards)
- [ggml-org. GGUF file format specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

<a id="topic-model"></a>

## Model weights, tokens and the context window

[Exercise 2](exercises/002.txt) · book page 26

A language model stores learned numbers called weights. A tokenizer splits text into pieces and assigns each piece an integer. A piece might be part of a word. During inference the model reads the prompt and predicts successive tokens using the existing weights. Training changes those weights.

The context window bounds how many tokens the model can use in a request. During generation a transformer saves attention keys and values for earlier positions in a KV cache. Increasing the context or serving additional sequences can require more memory even when the weight file stays exactly the same size.

Further reading

- [Qwen. Qwen2.5-7B-Instruct-GGUF model card and files](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)
- [Hugging Face. Transformers cache strategies](https://huggingface.co/docs/transformers/en/kv_cache)
- [Qwen. Qwen2.5-7B-Instruct configuration](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)

<a id="topic-music"></a>

## MusicGen and discrete audio tokens

[Exercise 188](exercises/188.txt) · book page 605

MusicGen generates a sequence of audio codes conditioned on text. A codec represents sound using entries from learned codebooks and decodes those codes into a waveform. Audio tokens index the codec representation. They have a different relationship to time from either text tokens or individual PCM samples in a WAV file. At 32,000 samples/s and 50 codec frames/s, one codec frame covers 640 waveform samples. The four-codebook delay pattern in this run uses three steps. For 256 generation steps the 253 frames yield 161,920 samples, or 5.06 s.

Generate one short clip and record the prompt, seed, number of generated steps, waveform shape and sample rate. Derive playback duration from the waveform. Listening answers questions about music or artifacts that a successful WAV write and the model’s timing numbers cannot settle.

Further reading

- [Hugging Face. MusicGen Transformers 4.57.1. Consulted 27 October 2026](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/musicgen)
- [Meta. MusicGen small model card. Consulted 27 October 2026](https://huggingface.co/facebook/musicgen-small/tree/4c8334b02c6ec4e8664a91979669a501ec497792)
- [Jade Copet and colleagues. Simple and Controllable Music Generation 2023. Consulted 27 October 2026](https://arxiv.org/abs/2306.05284)

<a id="topic-nemotron"></a>

## Nemotron, mixtures of experts and hybrid state

[Exercise 247](exercises/247.txt) · book page 799

Nemotron names a family of models. The Nemotron 3 Nano variant used here combines expert routing with different kinds of sequence-processing layers. A mixture of experts selects part of the model for each token. The active parameter count describes work selected per token. The full stored model can still be much larger.

Trace the GGUF conversion back to a particular upstream revision. Inspect actual file bytes, the available memory views and which layers the native runtime placed on CUDA. Attention caches and recurrent state may have different layouts, so use the model’s own logs when interpreting its buffers.

Further reading

- [NVIDIA. Nemotron 3 Nano 30B A3B model card. Consulted 28 October 2026](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [ggml-org. Pinned Nemotron GGUF conversion files. Consulted 28 October 2026](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3-Nano-30B-A3B-GGUF/tree/fe801509a580ff84a5728d4a9104e3f6f7e9b0dd)
- [Hugging Face. Nemotron conversion catalog metadata with blob identities. Consulted 28 October 2026](https://huggingface.co/api/models/ggml-org/NVIDIA-Nemotron-3-Nano-30B-A3B-GGUF?blobs=true)
- [Unsloth. Nemotron inference and fine-tuning guidance. Consulted 28 October 2026](https://unsloth.ai/docs/models/nemotron-3)

<a id="topic-ncu"></a>

## Nsight Compute and hardware counters

[Exercise 41](exercises/041.txt) · book page 183

Nsight Compute examines selected GPU kernels in detail. Hardware counters record events such as memory traffic or executed operations. A profiler can combine counters into measures of achieved throughput and may need multiple passes to collect them. This differs from measuring the duration of an entire request.

Pick one named kernel and a bounded input. First check its numerical result. Then look for a kernel report containing the requested metric and unit. A permissions error or an unavailable metric is a finding about measurement access and supplies no value for that counter.

Further reading

- [NVIDIA. Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [NVIDIA. ERR_NVGPUCTRPERM and performance-counter access](https://developer.nvidia.com/ERR_NVGPUCTRPERM)
- [NVIDIA. Nsight Compute Profiling Guide, roofline charts](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

<a id="topic-nsys"></a>

## Nsight Systems, NVTX and CUPTI timelines

[Exercise 39](exercises/039.txt) · book page 176

Nsight Systems records a timeline across CPU threads, CUDA API calls and supported GPU activity. NVTX lets the application name a region such as “iteration 2”. CUPTI is an NVIDIA interface that tools can use to collect activity records. A host range describes what the CPU was doing while it was open.

Start with one named iteration. Look for the CPU launch call, then the corresponding GPU work with usable start and end times. A launch returning to the CPU does not mark the end of device execution. Zero device timestamps leave the duration unresolved.

Further reading

- [NVIDIA. NVTX: NVIDIA Tools Extension Library](https://nvidia.github.io/NVTX/)
- [NVIDIA. CUPTI usage and collection interfaces](https://docs.nvidia.com/cupti/main/main.html)
- [NVIDIA. Nsight Systems User Guide, capture ranges, statistics, and export](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

<a id="topic-arrays"></a>

## NumPy arrays, dataframes and data provenance

[Exercise 152](exercises/152.txt) · book page 487

NumPy stores numeric arrays with a shape and an element type. A dataframe is a table with named columns that may have different types. pandas provides dataframes on the CPU. A CSV file stores rows as text, while Parquet stores typed columns. Loading the same values with a different type can change a comparison.

The ecosystem exercises reuse small arrays and tables as reference inputs. Record their shape, column names, row identifiers, types and file digests. When comparing a CPU and GPU operation, put rows in the same documented order before comparing values.

Further reading

- [scikit-learn. PCA API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [NVIDIA RAPIDS. Installation guide. Consulted 27 October 2026](https://docs.nvidia.com/datascience/install/)
- [NumPy. The absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [pandas. What kind of data does pandas handle](https://pandas.pydata.org/docs/getting_started/intro_tutorials/01_table_oriented.html)

<a id="topic-containersgpu"></a>

## The NVIDIA Container Toolkit and GPU visibility

[Exercise 142](exercises/142.txt) · book page 453

A GPU container needs two things at once. Its image supplies the application and compatible user-space libraries. The container runtime makes the required GPU device interface and driver components available to the running container. Under WSL2 the Windows host still owns the mapped display driver.

Run a small device check inside the intended container, then perform a known calculation and check its output. A successful query in the surrounding WSL shell describes that shell. A container that starts successfully may still be missing the GPU interface its model needs.

Further reading

- [NVIDIA. CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [NVIDIA. Container Toolkit installation guide. Consulted 27 October 2026](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Kubernetes. Schedule GPUs. Consulted 27 October 2026](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)

<a id="topic-nemo"></a>

## NVIDIA NeMo AutoModel and supervised training recipes

[Exercise 168](exercises/168.txt) · book page 529

NeMo AutoModel runs training from a recipe that brings together a model, tokenizer, data reader, optimizer and checkpoint settings. Changing one of these can change what the training run means. This exercise follows the data from a teaching example to the token positions that receive a training loss.

For prompt-and-answer training, the labels can exclude the prompt and padding so that only intended answer tokens contribute to the objective. Decode a few labeled positions and inspect one batch before training. Keep the recipe, model identity, trainable parameter list and checkpoint needed for the recovery test.

Further reading

- [NVIDIA NeMo AutoModel. Installation. Consulted 27 October 2026](https://docs.nvidia.com/nemo/automodel/latest/get-started/installation)
- [NVIDIA NeMo AutoModel. Supervised and parameter efficient fine tuning. Consulted 27 October 2026](https://docs.nvidia.com/nemo/automodel/latest/recipes-e2e-examples/sft-peft)
- [NVIDIA NeMo AutoModel. Checkpointing. Consulted 27 October 2026](https://docs.nvidia.com/nemo/automodel/latest/development/checkpointing)
- [NVIDIA NeMo AutoModel. Qwen3 0.6B HellaSwag PEFT recipe at v0.7.0. Consulted 27 October 2026](https://github.com/NVIDIA-NeMo/Automodel/blob/v0.7.0/examples/llm_finetune/qwen/qwen3_0p6b_hellaswag_peft.yaml)

<a id="topic-ngc"></a>

## NVIDIA NGC Catalog, resources and container images

[Exercise 204](exercises/204.txt) · book page 663

NGC Catalog is NVIDIA’s catalog of GPU software and other assets. A resource entry can describe a downloadable file or tool. A container entry identifies an image in a registry. Reading a catalog page, acquiring an asset and executing it are three distinct operations.

Start by recording the item type, owner, version, expected size and available files. For a native command-line tool, inspect the executable’s target architecture before running it in this ARM64 WSL2 guest. For a container, follow its platform-specific digest in Exercise 206.

Further reading

- [NVIDIA. NGC catalog user guide. Consulted 28 October 2026](https://docs.nvidia.com/ngc/latest/ngc-catalog-user-guide.html)
- [NVIDIA. NGC user guide and account access. Consulted 28 October 2026](https://docs.nvidia.com/ngc/latest/ngc-user-guide.html)
- [NVIDIA. NGC CLI setup with pinned 4.34.10 installation commands. Consulted 28 October 2026](https://catalog.ngc.nvidia.com/orgs/nvidia/ngc-apps/resources/ngc_cli/-/setup)
- [NVIDIA. NGC CLI registry resource info command, including --files. Consulted 28 October 2026](https://docs.ngc.nvidia.com/cli/cmd_registry.html#info_repeat11)

<a id="topic-tritonserver"></a>

## NVIDIA Triton Inference Server and model repositories

[Exercise 150](exercises/150.txt) · book page 476

NVIDIA Triton Inference Server serves models through named backends. Its model repository contains model names, numbered version directories and configuration describing input and output tensors. A Python backend can implement an encoder. Other backends can execute an ONNX graph or a TensorRT engine.

Follow a small request from its input name and shape to the returned embedding. Check the model version that actually answered. Dynamic batching can combine several requests into one backend execution, so request count and execution count need not match. The Triton kernel programming language introduced in Exercise 101 is a different project.

Further reading

- [NVIDIA. Triton Inference Server 26.09 release notes. Consulted 27 October 2026](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2730/release-notes/rel-26-09.html)
- [NVIDIA. Triton model repositories. Consulted 27 October 2026](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html)
- [NVIDIA. Triton Python backend. Consulted 27 October 2026](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html)
- [NVIDIA. Triton dynamic batching. Consulted 27 October 2026](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
- [NVIDIA. Triton model configuration. Consulted 27 October 2026](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)

<a id="topic-ollama"></a>

## Ollama, local model names and residency

[Exercise 235](exercises/235.txt) · book page 757

Ollama runs a local service that manages models and exposes generation and embedding endpoints. A local model name identifies what to request. Importing a GGUF file creates a usable model entry, while loading it creates a running model that consumes resources. A residency setting influences how long it can stay loaded between calls.

Read the installed model list, then inspect the running models around a bounded request. For streamed generation, read the sequence of JSON objects through the final completion record. For embeddings, inspect vector count, dimension and response model identity.

Further reading

- [Ollama. Linux installation. Consulted 28 October 2026](https://docs.ollama.com/linux)
- [Ollama. FAQ on local mode, residency and concurrency. Consulted 28 October 2026](https://docs.ollama.com/faq)
- [Ollama. API schema and model metadata. Consulted 28 October 2026](https://github.com/ollama/ollama/blob/main/docs/openapi.yaml)
- [Ollama. Generation API. Consulted 28 October 2026](https://docs.ollama.com/api/generate)
- [Ollama. Embedding API. Consulted 28 October 2026](https://docs.ollama.com/api/embed)
- [Ollama. Importing a model. Consulted 28 October 2026](https://docs.ollama.com/import)

<a id="topic-onnx"></a>

## ONNX graphs and ONNX Runtime execution providers

[Exercise 151](exercises/151.txt) · book page 480

ONNX describes a model as a graph of operations, named inputs and outputs, and stored parameters. An opset specifies the operator definitions the graph expects. ONNX Runtime executes a graph by assigning supported groups of operations to execution providers, for example a CPU provider or the CUDA provider.

Check the graph structure and run a known input against an independent numerical result. Then examine provider assignment for the relevant nodes. A listed CUDA provider identifies build support. Create a session, inspect its active providers and check the relevant node’s profile event. An assignment log establishes placement, while timed device activity requires a usable trace.

Further reading

- [ONNX. Model versioning. Consulted 27 October 2026](https://onnx.ai/onnx/repo-docs/Versioning.html)
- [ONNX. Checker API. Consulted 27 October 2026](https://onnx.ai/onnx/api/checker.html)
- [ONNX Runtime. Profiling tools. Consulted 27 October 2026](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)
- [ONNX Runtime. CUDA execution provider. Consulted 27 October 2026](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

<a id="topic-otel"></a>

## OpenTelemetry traces, spans and correlated logs

[Exercise 111](exercises/111.txt) · book page 379

A trace follows one request across operations. Each span records a named operation, its time interval and its relationship to a parent span. A trace ID groups the operations belonging to one request. A span ID lets a log entry identify the particular operation that emitted it. Trace sampling selects requests or spans to retain. Metric sampling polls a measured field at successive times. Preserve the trace policy and the metric interval separately.

Follow one synthetic request from its root span through retrieval and generation. Match a log to its span ID and look for a second attempt after a failure. Adding all span durations can double-count time when a parent includes a child or two operations overlap.

Further reading

- [OpenTelemetry. Python manual instrumentation](https://opentelemetry.io/docs/languages/python/instrumentation/)
- [OpenTelemetry. Traces](https://opentelemetry.io/docs/concepts/signals/traces/)

<a id="topic-packing"></a>

## Packing INT4 values into bytes

[Exercise 66](exercises/066.txt) · book page 249

A byte has eight bits, so it can hold two four-bit codes. An INT4 representation must specify how a signed integer maps to a code, which half-byte comes first, and what happens to the last value when the count is odd. To represent real-valued weights, a scale maps codes back to approximate numbers.

Write down a tiny input, its expected codes and the exact packed byte sequence. Check bytes, then unpack and compare the reconstructed numbers. One implementation can pack and unpack consistently while still using a layout incompatible with another runtime.

Further reading

- [ONNX Runtime. Quantize ONNX models. Consulted 27 October 2026](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ONNX. QuantizeLinear operator. Consulted 27 October 2026](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html)

<a id="topic-pca"></a>

## PCA and a two-dimensional map of embeddings

[Exercise 79](exercises/079.txt) · book page 288

The scikit-learn library provides a Python implementation of principal component analysis (PCA). PCA finds directions in which centered data vary most. Plotting the first two components assigns a point to each sentence and retains only part of the original variation. The explained-variance ratios report how much the displayed components capture in the fitted data.

Fit the axes on the training vectors and use that same fit to transform held-out vectors. Follow a pair of points back through their row IDs and compare their original-space similarity. PCA axes can reverse sign across implementations, so a reflected plot can still represent the same subspace.

Further reading

- [scikit-learn. PCA API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [scikit-learn. Trustworthiness API](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html)
- [NVIDIA cuML. PCA API. Consulted 27 October 2026](https://docs.nvidia.com/cuml/latest/api/generated/cuml.decomposition.PCA/)

<a id="topic-plots"></a>

## Plotting GPU activity against a workload

[Exercise 92](exercises/092.txt) · book page 327

Matplotlib is the Python plotting library used here. A time-series plot gives timestamps horizontal positions and measured values vertical positions. The plotting program reads the sampler’s records and draws lines or points for related quantities. Request start and finish markers show which part of the picture belongs to the workload.

Read the axis units before comparing curves. Bytes and percentages belong to different scales. Leave gaps where the source returned no sample and keep the original CSV or JSON alongside the figure. A rise in device activity during a request is a reason to inspect a trace for attribution.

Further reading

- [NVIDIA. NVIDIA System Management Interface and field definitions](https://docs.nvidia.com/deploy/nvidia-smi/)
- [Python. time and monotonic clocks](https://docs.python.org/3/library/time.html)
- [Matplotlib. Quick start guide](https://matplotlib.org/stable/users/explain/quick_start.html)

<a id="topic-prometheus"></a>

## Prometheus, exporters and time series

[Exercise 212](exercises/212.txt) · book page 690

Prometheus asks configured targets for metrics at intervals called scrapes. An exporter translates another system’s measurements into named values that Prometheus can read. Labels distinguish series, for example by device. A gauge can rise or fall, while a counter usually accumulates events until its process restarts.

Trace one value from the original provider through the exporter’s metrics response to a raw Prometheus query. The lab’s WSL2 exporter can indicate when a management field is unavailable. NVIDIA’s DCGM exporter is another possible data source where DCGM is supported. A successful scrape says the endpoint answered.

Further reading

- [NVIDIA. DCGM getting started and supported environments. Consulted 28 October 2026](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/getting-started.html)
- [NVIDIA. DCGM installation and exporter architecture. Consulted 28 October 2026](https://docs.nvidia.com/datacenter/dcgm/latest/installation/index.html)
- [Prometheus. Configuration and scrape jobs. Consulted 28 October 2026](https://prometheus.io/docs/prometheus/latest/configuration/configuration/)
- [Prometheus. Query functions including rate resets and histogram_quantile. Consulted 28 October 2026](https://prometheus.io/docs/prometheus/latest/querying/functions/)

<a id="topic-prompt"></a>

## Prompts, generated answers and acceptance checks

[Exercise 6](exercises/006.txt) · book page 44

A chat prompt can contain messages with different roles. The system message sets overall instructions, while the user message supplies the task and its evidence. The model generates a sequence of tokens. A finish reason records why generation ended, such as a normal stop or an output length limit.

An answer can be fluent and still miss the task. If a prompt asks for two observations, count them and check both against the supplied lines. Recording the prompt, the complete reply and its finish reason lets a second reader grade that same answer.

Further reading

- [Qwen. Qwen2.5-7B-Instruct-GGUF model card and files](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)
- [ggml-org. llama.cpp HTTP server documentation, pinned tag b11898](https://github.com/ggml-org/llama.cpp/blob/b11898/tools/server/README.md)

<a id="topic-promql"></a>

## PromQL rates, histograms and alert rules

[Exercise 216](exercises/216.txt) · book page 700

PromQL is Prometheus’s query language. A rate over a counter estimates the average increase per second after accounting for counter resets across a time window. Histogram buckets accumulate observations up to specified boundaries. From those counts we can estimate a percentile such as p95, with precision limited by the bucket boundaries and available events.

An alert rule evaluates an expression repeatedly and can require it to remain true for a specified duration. Follow a teaching sequence from pending to firing and recovery. Test absent samples as a separate case. A missing series might otherwise suppress the very alert you hoped would reveal a lost measurement.

Further reading

- [Prometheus. Query functions including rate resets and histogram_quantile. Consulted 28 October 2026](https://prometheus.io/docs/prometheus/latest/querying/functions/)
- [Prometheus. Histograms and summaries. Consulted 28 October 2026](https://prometheus.io/docs/practices/histograms/)
- [Prometheus. Alerting rules and hold duration. Consulted 28 October 2026](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)
- [Prometheus. Unit testing recording and alerting rules. Consulted 28 October 2026](https://prometheus.io/docs/prometheus/latest/configuration/unit_testing_rules/)

<a id="topic-instructions"></a>

## PTX, SASS and device registers

[Exercise 33](exercises/033.txt) · book page 157

CUDA can represent a kernel as PTX, a virtual instruction language, and as native machine code for a target GPU. A disassembler displays that native code in a readable form often called SASS. Registers hold intermediate values used by a thread. Compiler optimization can combine operations or remove a source variable altogether.

Use cuobjdump to identify which images an executable actually contains. If it contains a native image only, generate PTX separately for comparison. Static instructions show what was built. A debugger stop or a measured trace is needed to identify what happened during one execution.

Further reading

- [NVIDIA. CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)
- [NVIDIA. CUDA-GDB manual, WSL and GPU core dump sections](https://docs.nvidia.com/cuda/cuda-gdb/index.html#known-issues)
- [NVIDIA. Parallel Thread Execution ISA 9.4](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

<a id="topic-pydantic"></a>

## PydanticAI and typed model outputs

[Exercise 229](exercises/229.txt) · book page 742

Pydantic models describe fields, types and validation rules for data. PydanticAI uses such contracts in agents, including for a model’s final output. A structured finding might require a named observation and a reference to evidence. A plausible paragraph or a well-formed JSON object can still fail that contract.

Feed the validator one accepted example and one that violates a specific rule. Then trace what happens when the model repeatedly returns invalid output. Keep the retry budget and actual request count visible. Passing field validation does not establish that the evidence supports the reported conclusion.

Further reading

- [Pydantic. PydanticAI unit testing and FunctionModel. Consulted 28 October 2026](https://pydantic.dev/docs/ai/guides/testing/)
- [Pydantic. PydanticAI output validation. Consulted 28 October 2026](https://pydantic.dev/docs/ai/core-concepts/output/)
- [Pydantic. PydanticAI retry layers. Consulted 28 October 2026](https://pydantic.dev/docs/ai/core-concepts/retries/)
- [Pydantic. Other compatible APIs and OpenAICompatibleProvider. Consulted 28 October 2026](https://pydantic.dev/docs/ai/models/compatible-apis/)

<a id="topic-pythonpackages"></a>

## Python environments, pip and binary wheels

[Exercise 14](exercises/014.txt) · book page 85

Python packages can contain interpreted code and compiled libraries. A wheel is an installable package archive. Its platform and interpreter tags matter on an ARM64 laptop. A source build may also need a compiler and matching Python development headers. A virtual environment keeps project packages together and selects an interpreter for the current shell.

Use python -m pip with the same interpreter you use for the exercise. A successful dependency check, a successful import and a synchronized CUDA calculation each establish something different. When one fails, record its actual message before changing packages.

Further reading

- [PyTorch. Local installation guidance and platform selection](https://pytorch.org/get-started/locally/)
- [PyTorch. Official CUDA 13.0 torch wheel index, including aarch64 builds](https://download.pytorch.org/whl/cu130/torch/)
- [Python Software Foundation. Virtual environments and packages](https://docs.python.org/3/tutorial/venv.html)
- [Python Packaging Authority. Package formats](https://packaging.python.org/en/latest/discussions/package-formats/)

<a id="topic-allocator"></a>

## PyTorch allocator snapshots and tensor ownership

[Exercise 42](exercises/042.txt) · book page 186

When a program creates a CUDA tensor, PyTorch obtains storage for it. Its allocator can retain unused blocks so that the next tensor can reuse them. Allocated bytes describe storage currently used by tensors. Reserved bytes include blocks held by the allocator. A management tool has a broader view again.

An allocator snapshot records blocks and their state, and can include allocation history if recording was enabled. To inspect numerical content, explicitly copy a small selected tensor. When diagnosing growth, first release the object that owns it, then compare live and reserved bytes in the same process.

Further reading

- [PyTorch. CUDA semantics and memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
- [PyTorch. Understanding CUDA Memory Usage, version 2.14](https://docs.pytorch.org/docs/2.14/torch_cuda_memory.html)
- [PyTorch CUDA memory snapshots. See [39]](https://docs.pytorch.org/docs/2.14/torch_cuda_memory.html)

<a id="topic-profiler"></a>

## PyTorch Profiler and operators

[Exercise 43](exercises/043.txt) · book page 188

A framework operator is a named operation such as matrix multiplication. PyTorch Profiler can collect CPU-side operator events and, where supported, GPU activity. A trace viewer draws each event on the track of the thread or device that produced it. One Python operation can issue several kernels.

Choose a short, named operation and check both tracks. A CPU operator range can include launches and waiting. To report GPU execution time, find valid device intervals for the workload. If the native model runs in a separate process, identify that process before interpreting a client trace.

Further reading

- [NVIDIA. CUPTI usage and collection interfaces](https://docs.nvidia.com/cupti/main/main.html)
- [PyTorch. torch.profiler, version 2.14](https://docs.pytorch.org/docs/2.14/profiler.html)
- [NVIDIA. Nsight Systems User Guide, capture ranges, statistics, and export](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

<a id="topic-torch"></a>

## PyTorch tensors and synchronized CUDA execution

[Exercise 14](exercises/014.txt) · book page 85

A tensor is a rectangular collection of numbers described by its shape, numeric type and device. A vector can have shape (3,), a matrix (2, 3). PyTorch can create these objects on the CPU or through CUDA on the GPU. A wheel is an installable package archive containing this framework build.

A CUDA kernel is a function that runs on the GPU. PyTorch launches kernels for tensor operations. GPU work can continue after the Python call that launched it returns. The probe performs a small calculation, synchronizes and checks its result. Read the interpreter path, device name, tensor device and result together. They connect a particular Python environment with an operation that completed.

Further reading

- [PyTorch. Local installation guidance and platform selection](https://pytorch.org/get-started/locally/)
- [PyTorch. CUDA semantics and memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
- [NVIDIA. CUDA Runtime API: Memory Management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html)

<a id="topic-quant"></a>

## Quantization and weight formats

[Exercise 11](exercises/011.txt) · book page 63

Quantization represents numbers using fewer possible values. A weight group can share a scale so a small integer represents an approximate weight. Four bits can encode sixteen patterns. A real Q4 file also stores scales, metadata and sometimes tensors at other precisions, so its byte count need not be exactly half that of Q8.

The smaller file is easier to place in memory. Approximation can change the generated tokens. Compare the same prompt with the same context, output cap and runtime settings. Keep the finish reason alongside reply quality and measured latency. Nominal bits and a rounded parameter count give a lower-complexity estimate. Q8_0 stores 32 eight-bit codes and one 16-bit scale per block, or 8.5 bits per weight. Q4_K_M mixes tensor types. Use verified shard bytes for a storage budget and measure buffers separately after loading.

Further reading

- [Qwen. Qwen2.5-7B-Instruct-GGUF model card and files](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)
- [Dettmers, T., et al. QLoRA: Efficient Finetuning of Quantized LLMs. 2023](https://arxiv.org/abs/2305.14314)
- [ONNX Runtime. Quantize ONNX models. Consulted 27 October 2026](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ONNX. QuantizeLinear operator. Consulted 27 October 2026](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html)
- [ggml. Block quantization types and layouts](https://github.com/ggml-org/llama.cpp/blob/b11898/ggml/src/ggml-common.h)

<a id="topic-rmm"></a>

## RAPIDS Memory Manager and allocation statistics

[Exercise 155](exercises/155.txt) · book page 494

GPU dataframe and machine-learning operations allocate temporary as well as long-lived buffers. RAPIDS Memory Manager, or RMM, provides memory resources through which participating libraries obtain device storage. A pool can hold memory between operations and fulfill several small allocations from its reserve.

Record RMM’s statistics before, during and after the same operation. Compare peak and current bytes with the device-level free-memory reading. The statistics describe allocations that passed through the monitored resource. Other libraries and processes may allocate outside it.

Further reading

- [PyTorch. CUDA semantics and memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
- [RAPIDS RMM. Allocation statistics API. Consulted 27 October 2026](https://docs.rapids.ai/api/rmm/stable/python/statistics/)

<a id="topic-rapids"></a>

## RAPIDS, cuDF and cuML

[Exercise 153](exercises/153.txt) · book page 490

RAPIDS is a collection of libraries for data analysis on NVIDIA GPUs. cuDF provides dataframe operations, such as grouping rows by a category and summing a column. cuML supplies machine-learning algorithms, including PCA. A familiar operation name makes a CPU implementation a useful reference.

Begin with a tiny table whose groups you can sum by hand. Run the corresponding CPU and GPU operations, align the group keys and compare the sums. For PCA, compare the fitted subspace or its projector. Two mathematically equivalent bases can have different signs or orientations.

Further reading

- [NVIDIA RAPIDS. Installation guide. Consulted 27 October 2026](https://docs.nvidia.com/datascience/install/)
- [NVIDIA NGC. RAPIDS Base 26.10 CUDA 13 Python 3.13. Consulted 27 October 2026](https://catalog.ngc.nvidia.com/orgs/nvidia/rapidsai/containers/base/26.10-cuda13-py3.13)
- [NVIDIA cuML. PCA API. Consulted 27 October 2026](https://docs.nvidia.com/cuml/latest/api/generated/cuml.decomposition.PCA/)

<a id="topic-sqlite"></a>

## Reading an exported trace with SQLite

[Exercise 40](exercises/040.txt) · book page 180

SQLite stores a relational database in a file. Each table contains rows with named columns. A trace exporter can put API calls, kernels and text labels into different tables. SQL queries select rows or join tables through a recorded key, such as a correlation identifier.

Inspect the available table names and column definitions before applying a query from another installation. Check the time units and find one known iteration. An empty result can mean the requested activity was never captured, the table has a different layout, or the filter selected no matching rows.

Further reading

- [NVIDIA. Nsight Systems User Guide, capture ranges, statistics, and export](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [SQLite. About SQLite](https://www.sqlite.org/about.html)

<a id="topic-reductiontest"></a>

## Reducing a failing case and isolating a change

[Exercise 115](exercises/115.txt) · book page 388

A reproducer is an input and procedure that reliably expose a chosen failure. Its predicate answers whether a candidate still fails in the required way. Reduction removes parts of an input and runs the predicate again. A result is 1-minimal when removing any single remaining element stops that failure.

When comparing configurations, test the same predicate on the same input. Bisection can narrow a transition if the ordered candidates separate passing from failing consistently. Keep the actual observation at each step. An unrelated crash during reduction would otherwise count as success for the wrong reason.

Further reading

- [Zeller and Hildebrandt. Simplifying and Isolating Failure-Inducing Input](https://doi.org/10.1109/32.988498)

<a id="topic-reductions"></a>

## Reductions, histograms and atomic updates

[Exercise 49](exercises/049.txt) · book page 202

A reduction combines many values into fewer values, such as summing a vector into one number. A GPU implementation can first write a partial sum per block and then sum those partials. Addition order affects floating-point rounding, so preserve the input type and comparison tolerance.

A histogram instead counts how many inputs fall into each bin. Several threads may choose the same bin. An atomic addition makes each increment indivisible with respect to the other increments at that location. Check the individual bins and whether their total equals the number of counted inputs.

Further reading

- [NVIDIA. CUDA Programming Guide. Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
- [NVIDIA. CUDA Programming Guide. C/C++ Language Extensions, atomic functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)
- [NVIDIA. Floating Point and IEEE 754](https://docs.nvidia.com/cuda/floating-point/index.html)

<a id="topic-ranking"></a>

## Retrieval, relevance labels and rank metrics

[Exercise 78](exercises/078.txt) · book page 285

A retrieval system compares a query with a collection of documents and returns an ordered list. Relevance labels tell the evaluator which documents answer a particular query. The top match can be semantically similar yet irrelevant to the question, so retain query and document identifiers alongside their scores.

Recall at k measures the fraction of relevant documents found in the first k places. Reciprocal rank uses the position of the first relevant result. A first relevant result at rank four contributes 0.25. Mean reciprocal rank, or MRR, averages those contributions across queries. Inspect individual ranks when the test set is small.

Further reading

- [Reimers, N., and Gurevych, I. Sentence-BERT. 2019](https://arxiv.org/abs/1908.10084)
- [Sentence Transformers. SentenceTransformer API](https://sbert.net/docs/package_reference/sentence_transformer/model.html)
- [Sentence Transformers. InformationRetrievalEvaluator](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

<a id="topic-normalization"></a>

## RMS normalization, softmax and numerical stability

[Exercise 55](exercises/055.txt) · book page 218

RMS normalization scales a vector using the square root of its mean squared value and a small stabilizing constant. Softmax turns a row of scores into positive weights that sum to one. These operations appear inside neural networks and involve reductions across a row.

For softmax, subtract the largest score before exponentiating. This leaves the mathematical probabilities unchanged and reduces overflow risk. Inspect a row containing large scores, a row with repeated scores and a partial final row. Compare finite values and row sums with an independent reference.

Further reading

- [NVIDIA. Floating Point and IEEE 754](https://docs.nvidia.com/cuda/floating-point/index.html)
- [Zhang, B., and Sennrich, R. Root Mean Square Layer Normalization. 2019](https://arxiv.org/abs/1910.07467)
- [Milakov, M., and Gimelshein, N. Online normalizer calculation for softmax. 2018](https://arxiv.org/abs/1805.02867)

<a id="topic-roofline"></a>

## Roofline models of kernel performance

[Exercise 105](exercises/105.txt) · book page 365

A roofline relates work performed to data moved. Arithmetic intensity is operations divided by bytes transferred at a stated memory boundary. Multiplying that intensity by bandwidth gives a memory-based ceiling. The smaller of that number and the compute ceiling gives the model’s upper bound for that intensity. The ridge point is compute ceiling divided by bandwidth. With 10 TFLOP/s and 100 GB/s it is 100 operations per byte. The illustrative add is below that crossover, and the 1024-square ideal matrix product is above it.

For example, two operations per byte and an assumed 100 GB/s give a 200 billion operations/s memory ceiling. Those are teaching inputs. An achieved point for this laptop requires measured work, time and an appropriate account of memory traffic. Keep estimated and measured values labeled.

Further reading

- [NVIDIA. CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA. Nsight Compute Profiling Guide, roofline charts](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

<a id="topic-embeddings"></a>

## Sentence embeddings and semantic similarity

[Exercise 75](exercises/075.txt) · book page 277

An encoder reads text and returns numbers. A sentence embedding represents a whole passage by one fixed-length vector, which can be compared with vectors for other passages. Sentence Transformers supplies the model interface used here. MiniLM is the compact pretrained encoder selected for the first experiment. An input token embedding represents a vocabulary item; its position and surrounding tokens contribute to the contextual representation. For nonzero vectors, cosine similarity is their dot product divided by the product of their lengths.

For an intuition, the example vectors (1, 0) and (0, 1) point at right angles and have cosine similarity zero. Two identical unit vectors have similarity one. Real vectors have many dimensions. Encode a diagnostic query and two candidate descriptions, then inspect which description ranks first and whether it answers the query.

Further reading

- [Reimers, N., and Gurevych, I. Sentence-BERT. 2019](https://arxiv.org/abs/1908.10084)
- [Sentence Transformers. all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Sentence Transformers. SentenceTransformer API](https://sbert.net/docs/package_reference/sentence_transformer/model.html)

<a id="topic-ownership"></a>

## Sharing a GPU and owning a model

[Exercise 130](exercises/130.txt) · book page 425

Several requests can share one GPU through a single process that owns the loaded model. Other workers send it inputs and receive results. Independent model-owning processes can instead allocate their own copies of the weights and runtime state. Count resident owners as well as client workers. The Multi-Process Service (MPS) lets eligible work from several client processes use the GPU concurrently. Each client still owns its model parameters. This laptop’s serving checks use their recorded ownership policy.

NVIDIA Multi-Process Service, or MPS, is a separate GPU scheduling facility with platform requirements. It does not automatically merge separate model copies. The single-owner queue in this lab lets us study waiting, cancellations and sharing on the available laptop without assuming MPS support.

Further reading

- [PyTorch. CUDA semantics and memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
- [PyTorch. Multiprocessing best practices, version 2.14](https://docs.pytorch.org/docs/2.14/notes/multiprocessing.html)
- [NVIDIA. Multi-Process Service, consulted 27 October 2026](https://docs.nvidia.com/deploy/mps/latest/index.html)
- [NVIDIA. When to Use MPS and system limitations, consulted 27 October 2026](https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html)

<a id="topic-asr"></a>

## Speech recognition with Whisper

[Exercise 196](exercises/196.txt) · book page 628

Automatic speech recognition turns sound into text. Whisper processes acoustic features from audio, then decodes text tokens. The expected sampling rate, language and decoding settings affect the input or output. A fluent sentence is possible even when the audio contains little evidence for it.

Start with a known utterance and its reference wording. Save the original audio, hypothesis and settings together. Try a silent or non-speech control and inspect what was produced. For longer recordings, retain chunk start times so text can be traced back to the corresponding audio.

Further reading

- [Hugging Face. Whisper Transformers 4.57.1. Consulted 27 October 2026](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/whisper)
- [OpenAI. Whisper tiny model card. Consulted 27 October 2026](https://huggingface.co/openai/whisper-tiny/tree/169d4a4341b33bc18d8881c4b69c2e104e1cc0af)
- [SciPy. Polyphase resampling. Consulted 27 October 2026](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)

<a id="topic-sdxl"></a>

## Stable Diffusion XL, Diffusers and the VAE

[Exercise 184](exercises/184.txt) · book page 586

Stable Diffusion XL generates an image in stages. Text encoders represent the prompt. A denoiser and scheduler iteratively update a smaller numerical image representation called a latent. A variational autoencoder, or VAE, decodes the result into pixels. The Diffusers library assembles those components into a pipeline.

Keep prompt, seed, resolution and step count fixed while changing placement or VAE tiling. Offload can move components between CPU and GPU, while tiling changes how part of the image is decoded. Inspect the final image as well as peak memory. A changed pixel alone tells little about the requested visual change. In the pinned VAE, enabling tiling leaves a full 1024-pixel frame at the default threshold. Exercise 186 keeps that size, sets a smaller tile edge and counts the actual decoder calls.

Further reading

- [Hugging Face. Stable Diffusion XL Diffusers 0.35.1. Consulted 27 October 2026](https://huggingface.co/docs/diffusers/v0.35.1/en/using-diffusers/sdxl)
- [Hugging Face. Reducing diffusion memory use. Consulted 27 October 2026](https://huggingface.co/docs/diffusers/v0.35.1/en/optimization/memory)
- [Hugging Face. LoRA training with Diffusers 0.35.1. Consulted 27 October 2026](https://huggingface.co/docs/diffusers/v0.35.1/en/training/lora)
- [Diffusers. AutoencoderKL decoder and tile limits, v0.35.1](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/autoencoders/autoencoder_kl.py)

<a id="topic-stalls"></a>

## Stalled workers and stack traces

[Exercise 109](exercises/109.txt) · book page 375

A process can be alive while it makes no progress. A stack trace lists the active chain of function calls in a thread. A heartbeat records progress or liveness at intervals, depending on what it measures. A supervisor can impose a deadline and collect a stack from a deliberately stalled teaching worker.

Read the last completed step, stack and child exit together. If several samples show the worker waiting at the same point, inspect what would release that wait. The supervisor can complete successfully after recording a child failure, so retain both exits. A heartbeat from a separate monitor shows that monitor is alive. A completed-step counter emitted by the worker tests its progress.

Further reading

- [Python. faulthandler traceback collection](https://docs.python.org/3/library/faulthandler.html)

<a id="topic-streaming"></a>

## Streaming text and the first content event

[Exercise 9](exercises/009.txt) · book page 53

A streaming server sends pieces of a reply while it is still generating. In Server-Sent Events, a single HTTP connection carries a sequence of events. An event can announce a role or carry metadata before any text appears. Network chunks and model tokens also have different boundaries.

For the first-content latency, start the clock when the client sends the request and stop it at the first event containing text. Keep reading through the protocol’s completion marker. This gives two useful observations, time to visible content and time to a complete reply.

Further reading

- [ggml-org. llama.cpp HTTP server documentation, pinned tag b11898](https://github.com/ggml-org/llama.cpp/blob/b11898/tools/server/README.md)
- [MDN Web Docs. Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)

<a id="topic-tsne"></a>

## t-SNE and neighbourhoods in a plot

[Exercise 81](exercises/081.txt) · book page 293

t-SNE builds a low-dimensional arrangement that tries to keep similar points near each other. Its perplexity parameter controls the effective neighbourhood size, and an optimization procedure finds the plotted coordinates. Both the parameter and the random initialization can affect the picture.

Choose one query and label its plotted neighbours by ID. Check those IDs against the neighbours in the original embedding space. Repeat with another suitable perplexity or seed. A broad empty region between two groups on a t-SNE map is insufficient evidence for a corresponding gap in the original space.

Further reading

- [scikit-learn. t-SNE API and parameter interpretation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [scikit-learn. Trustworthiness API](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html)

<a id="topic-tensorrt"></a>

## TensorRT engines and optimization profiles

[Exercise 160](exercises/160.txt) · book page 505

TensorRT optimizes a model for inference and builds an engine for a target environment. An execution context runs that engine. With dynamic inputs, an optimization profile gives permitted minimum and maximum shapes and an optimum shape used in optimization. An engine’s portability depends on the build and target configuration.

Build from the small known graph, keep its profile and test a shape inside the allowed range. Compare output with the independent reference. Measure build time, transfers and execution separately. A prebuilt engine repeated many times answers a different timing question from building one before every request.

Further reading

- [NVIDIA TensorRT. 11.3.0 release notes. Consulted 27 October 2026](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.3.0.html)
- [NVIDIA TensorRT. Support matrix and engine portability. Consulted 27 October 2026](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
- [NVIDIA TensorRT. Migrating trtexec from 10 to 11. Consulted 27 October 2026](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-trtexec.html)
- [NVIDIA TensorRT. Python API documentation. Consulted 27 October 2026](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html)
- [NVIDIA TensorRT. Performance benchmarking. Consulted 27 October 2026](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)

<a id="topic-tts"></a>

## Text-to-speech with MMS and VITS

[Exercise 192](exercises/192.txt) · book page 615

A text-to-speech system turns written input into a speech waveform. The English MMS model in the lab uses VITS, a neural synthesis architecture. The text-processing step determines how the model receives numbers, abbreviations and punctuation. Those choices can affect pronunciation, pauses and length.

Keep the exact written sentence and the model’s sample rate with the WAV output. Compare a number written as digits with its spelled-out form. A nonempty, finite waveform establishes that audio was produced. For a pronunciation claim, listen to the specific word and record what you heard. The pinned English MMS vocabulary contains digits zero through six and omits seven, eight and nine. Its normalizer drops characters outside its vocabulary. Inspect token IDs for the digit string and for the equivalent words before judging the sound.

Further reading

- [Hugging Face. VITS Transformers 4.57.1. Consulted 27 October 2026](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/vits)
- [Meta. MMS English text to speech model card. Consulted 27 October 2026](https://huggingface.co/facebook/mms-tts-eng/tree/c71de0fe7204c83f1c10820a7d696d0b450048ba)
- [Hugging Face. MMS English speech model vocabulary](https://huggingface.co/facebook/mms-tts-eng/blob/main/vocab.json)

<a id="topic-pooling"></a>

## Token vectors, attention masks and pooling

[Exercise 76](exercises/076.txt) · book page 281

A token encoder produces one vector for each input position. To process sentences of different lengths in the same batch, the tokenizer can add padding. An attention mask identifies positions containing real input. In masked mean pooling we sum the real token vectors and divide by the number of real positions.

Imagine a two-token sentence padded to four positions. The denominator for its mean is two. If encoding the same sentence beside a longer one changes its representation substantially, inspect the mask and denominator first. L2 normalization then scales a nonzero sentence vector to length one.

Further reading

- [Reimers, N., and Gurevych, I. Sentence-BERT. 2019](https://arxiv.org/abs/1908.10084)
- [Sentence Transformers. SentenceTransformer API](https://sbert.net/docs/package_reference/sentence_transformer/model.html)
- [Hugging Face. BERT API, Transformers 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/bert)

<a id="topic-attention"></a>

## Transformer attention and cached decoding

[Exercise 59](exercises/059.txt) · book page 232

Self-attention lets one token combine information from other allowed positions. It produces a query for the current position, compares it with keys, and uses the resulting weights to combine values. A causal language model can attend to earlier positions while generating the next token.

During incremental decoding, saved keys and values avoid recomputing the whole prefix. Their positions must agree with the model’s position encoding. Feed the same short sequence once as a whole and once token by token. Compare the scores at each position to locate the first departure.

Further reading

- [Hugging Face. Transformers cache strategies](https://huggingface.co/docs/transformers/en/kv_cache)
- [Vaswani, A., et al. Attention Is All You Need. 2017](https://arxiv.org/abs/1706.03762)
- [Su, J., et al. RoFormer Enhanced Transformer with Rotary Position Embedding. 2021](https://arxiv.org/abs/2104.09864)

<a id="topic-hfstack"></a>

## Transformers, tokenizers and the Hugging Face libraries

[Exercise 25](exercises/025.txt) · book page 121

Several libraries can participate in one model run. Transformers loads a model implementation and exposes training or generation methods. Its tokenizer turns text into IDs using the saved vocabulary. PEFT attaches parameter-efficient adapters. Accelerate provides device-placement and training utilities. bitsandbytes supplies supported low-precision operations for configurations that use them.

Read the particular loader and training configuration in the companion. An ordinary BF16 LoRA run and a run using a quantized base have different memory requirements. Identify which precision and libraries were actually selected before comparing two results.

Further reading

- [Hugging Face. PEFT LoRA documentation](https://huggingface.co/docs/peft/en/developer_guides/lora)
- [Hugging Face. bitsandbytes installation and supported platforms](https://huggingface.co/docs/bitsandbytes/main/en/installation)
- [Hugging Face. Accelerate: Loading big models into memory](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference)
- [Hugging Face. Transformers 4.57.1 model loading and dtype](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model)
- [Hugging Face. Transformers documentation](https://huggingface.co/docs/transformers/en/index)
- [Hugging Face. Accelerate documentation](https://huggingface.co/docs/accelerate/en/index)

<a id="topic-umap"></a>

## UMAP and the neighbourhood graph

[Exercise 81](exercises/081.txt) · book page 293

UMAP builds a graph of nearby points and looks for a low-dimensional layout that preserves aspects of those relationships. The number of neighbours influences the scale at which it builds the graph. The minimum-distance setting influences how tightly points may be packed in the layout.

Use the same saved vectors and row identities you used for PCA and t-SNE. Change one setting at a time and inspect how the neighbours of a named sentence change. A fitted UMAP model can place new vectors using its transform procedure. Record the fit used for that transformation.

Further reading

- [UMAP. Basic parameters](https://umap-learn.readthedocs.io/en/latest/parameters.html)
- [UMAP. Reproducibility](https://umap-learn.readthedocs.io/en/latest/reproducibility.html)
- [UMAP. Transforming new data](https://umap-learn.readthedocs.io/en/latest/transform.html)

<a id="topic-unsloth"></a>

## Unsloth, QLoRA and a training smoke test

[Exercise 239](exercises/239.txt) · book page 770

Unsloth provides training integrations for supported models and environments. QLoRA stores a frozen base model in low precision and trains small LoRA adapter parameters with higher-precision computation where required. This reduces the storage needed for base weights, while activations and optimizer state still require memory.

A smoke test runs a few bounded updates to establish that this stack works together. Inspect the frozen base identity, the trainable set, supervised tokens and a change in the adapter. For recovery, compare the next optimizer update after resuming. Task quality requires a separate held-out evaluation.

Further reading

- [Dettmers, T., et al. QLoRA: Efficient Finetuning of Quantized LLMs. 2023](https://arxiv.org/abs/2305.14314)
- [Hugging Face. PEFT LoRA documentation](https://huggingface.co/docs/peft/en/developer_guides/lora)
- [Unsloth. Native requirements. Consulted 28 October 2026](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements)
- [Unsloth. Installation with WSL instructions. Consulted 28 October 2026](https://unsloth.ai/docs/get-started/install/windows-installation)
- [Unsloth. Official FastLanguageModel training script. Consulted 28 October 2026](https://github.com/unslothai/unsloth/blob/main/unsloth-cli.py)

<a id="topic-vectorindex"></a>

## Vector indexes and model identity

[Exercise 89](exercises/089.txt) · book page 316

A vector index connects stored document vectors with stable document identifiers so a new query can retrieve the matching text. A small teaching index can score every row directly. Larger systems may trade exact search for speed. Either way, the query and document vectors must occupy compatible coordinate systems.

Store the encoder identity, pooling method, dimension and document order with the index. A replacement encoder can produce vectors of the same length with different meanings. Search for a known passage after reopening the index, and verify the returned document ID as well as the score.

Further reading

- [Reimers, N., and Gurevych, I. Sentence-BERT. 2019](https://arxiv.org/abs/1908.10084)
- [Sentence Transformers. SentenceTransformer API](https://sbert.net/docs/package_reference/sentence_transformer/model.html)

<a id="topic-retained"></a>

## Verifying a CUDA allocation above guest RAM

[Exercise 251](exercises/251.txt) · book page 816

A memory query returns the number reported by one interface at one moment. An allocation test makes a different observation. To show that several blocks were usable together, the program must retain them simultaneously, write distinguishable patterns, read them back and check the results before releasing them.

On the worked laptop, guest MemTotal, the management-interface GPU total and CUDA’s reported budget are separate views of a unified-memory machine. Read the query, guarded 1 GiB pilot and optional 18 GiB target as successive stages. A successful target establishes the verified byte count for that run. Maximum capacity, physical residency and a model’s fit require different experiments.

Further reading

- [NVIDIA. CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [PyTorch. CUDA semantics and memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
- [PyTorch. torch.cuda.memory.mem_get_info](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.mem_get_info.html)
- [procps-ng. free(1)](https://man7.org/linux/man-pages/man1/free.1.html)
- [procps-ng. vmstat(8)](https://man7.org/linux/man-pages/man8/vmstat.8.html)

<a id="topic-vlm"></a>

## Vision-language models and visual grounding

[Exercise 176](exercises/176.txt) · book page 565

A vision-language model combines a visual representation with text so that a language component can generate an answer about an image. The processor resizes or tiles the pixels and prepares the visual input expected by the model. Increasing the visual input size can affect token count, memory use and latency. For the pinned Qwen2.5-VL-3B processor, 14-pixel patches merge in 2 × 2 groups. This gives the 28-pixel unit in the image-budget calculations.

Ask for a property you can verify in the source image, such as the position of a colored shape. Compare the answer with the image itself. Words printed inside a picture are also input data. They can test whether the model follows the caller’s task when the picture contains conflicting text.

Further reading

- [Hugging Face. Qwen2.5 VL Transformers 4.57.1. Consulted 27 October 2026](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_vl)
- [Qwen. Qwen2.5 VL 3B Instruct license. Consulted 27 October 2026](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/66285546d2b821cf421d4f5eb2576359d3770cd3/LICENSE)
- [Qwen. Model and preprocessor configurations at revision 66285546d2b821cf421d4f5eb2576359d3770cd3](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3)

<a id="topic-vllm"></a>

## vLLM and serving language models

[Exercise 148](exercises/148.txt) · book page 468

vLLM is an inference engine and server for language models. It schedules generation for active requests and manages memory used by their attention caches. Continuous batching lets the set of active sequences change as requests arrive or finish. A compatible HTTP API lets the lab use a familiar client.

Qualify the chosen image, model and device together. Check the server’s ready model identity, make one bounded generation request, then raise concurrency. Separate time in the queue, time to first content and completion. After weights and runtime workspace, remaining space and the configured KV-cache limit bound the cached tokens that concurrent sequences can share.

Further reading

- [vLLM. Docker deployment version 0.26.0. Consulted 27 October 2026](https://docs.vllm.ai/en/v0.26.0/deployment/docker/)
- [vLLM. Engine arguments version 0.26.0. Consulted 27 October 2026](https://docs.vllm.ai/en/v0.26.0/configuration/engine_args/)
- [vLLM. Production metrics. Consulted 27 October 2026](https://github.com/vllm-project/vllm/blob/main/docs/usage/metrics.md)

<a id="topic-warps"></a>

## Warps, Tensor Cores, cuBLAS and CUTLASS

[Exercise 61](exercises/061.txt) · book page 236

A warp is a group of 32 CUDA threads. Warp instructions can exchange values among participating lanes, which is useful in a reduction. Tensor Cores carry out supported small matrix multiply-and-accumulate operations. CUDA’s WMMA interface exposes such operations to a cooperating warp.

cuBLAS is NVIDIA’s library of numerical linear algebra routines. CUTLASS provides C++ components for constructing optimized kernels and has a profiler for testing implementations. Match input types, matrix layout, dimensions and the numerical oracle before comparing implementations. A profiler disposition of “Not verified” leaves the correctness comparison open.

Further reading

- [NVIDIA. CUDA Programming Guide. Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
- [NVIDIA. cuBLAS Library documentation](https://docs.nvidia.com/cuda/cublas/index.html)
- [NVIDIA. CUTLASS documentation overview](https://docs.nvidia.com/cutlass/latest/overview.html)
- [NVIDIA. CUTLASS Profiler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html)

<a id="topic-spectrum"></a>

## Waveforms, spectrograms and audio levels

[Exercise 189](exercises/189.txt) · book page 608

A waveform shows amplitude over time. A spectrogram divides the signal into short, often overlapping windows and shows how energy is distributed across frequencies in each window. The short-time Fourier transform is one way to compute it. Window length influences how well nearby times and frequencies can be distinguished.

Start with the known tone and a silent interval. Check the sample rate, time axis, frequency axis and level scale. Then inspect a generated clip and listen at the same locations. A bright band can identify energy at a frequency but cannot score pronunciation, musicality or intelligibility.

Further reading

- [SciPy. Polyphase resampling. Consulted 27 October 2026](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)
- [SciPy. Short time Fourier transform and spectrum scaling](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.signal.stft.html)

<a id="topic-wsl"></a>

## Windows, WSL2 and the mapped GPU

[Exercise 1](exercises/001.txt) · book page 22

WSL2 runs a Linux kernel in a lightweight virtual machine managed by Windows. Windows is the host. Ubuntu is the Linux system in which we open a terminal and run the exercises. ARM64, also called AArch64, describes the processor instruction set that a Linux executable must support on this laptop.

The NVIDIA driver lives on the Windows side. WSL2 makes its GPU interface available to Linux applications. The inventory asks Linux for its kernel and architecture, then runs nvidia-smi, NVIDIA’s management query for the GPU. These observations establish what the guest can see. A later synchronized calculation establishes that a CUDA program can execute there.

Further reading

- [NVIDIA. CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [Microsoft. Advanced settings configuration in WSL](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)
- [Microsoft. Working across Windows and Linux file systems](https://learn.microsoft.com/en-us/windows/wsl/filesystems)
- [Microsoft. What is the Windows Subsystem for Linux?](https://learn.microsoft.com/en-us/windows/wsl/about)

<a id="topic-wer"></a>

## Word error rate and transcript alignment

[Exercise 197](exercises/197.txt) · book page 630

Word error rate compares a recognition hypothesis with reference words. Count substitutions, deletions and insertions in an alignment, then divide their sum by the number of reference words. With four reference words and one substitution, the rate is 0.25. Extra inserted words can even make a rate exceed one.

Write down your rules for case, punctuation and numbers before comparing systems. Keep both the raw and normalized transcripts. An empty reference leaves the usual denominator undefined, so report the actual silence or non-speech behavior directly.

Further reading

- [Hugging Face. Whisper Transformers 4.57.1. Consulted 27 October 2026](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/whisper)
- [Hugging Face Evaluate. Word error rate metric. Consulted 27 October 2026](https://huggingface.co/spaces/evaluate-metric/wer/blob/main/README.md)

<a id="topic-kernels"></a>

## Writing a CUDA kernel with threads and blocks

[Exercise 32](exercises/032.txt) · book page 154

A kernel is a function that many GPU threads execute on different data. Threads are grouped into blocks, and the blocks form a grid. In one dimension, a thread can compute its element index from its block number, block size and position inside the block.

Imagine adding two vectors of length five with blocks of four threads. Two blocks supply eight thread positions. Only indices zero through four correspond to elements. A bounds check protects the tail. A grid-stride loop lets each valid thread handle further elements when the vector is much longer than the grid.

Further reading

- [NVIDIA. CUDA Programming Guide. Intro to CUDA C++](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)
- [NVIDIA. CUDA Programming Guide. Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
- [Harris, M. CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops. NVIDIA, 2013](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)
