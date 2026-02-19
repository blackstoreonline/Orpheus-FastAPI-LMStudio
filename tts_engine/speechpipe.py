from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import time

# Configuration constants
CUDA_CACHE_CLEAR_INTERVAL = 100  # Clear CUDA cache every N iterations to prevent fragmentation
SSE_DATA_PREFIX = "data: "  # Server-Sent Events data prefix
SSE_DATA_PREFIX_LEN = len(SSE_DATA_PREFIX)  # Pre-computed length for efficiency

# Device detection and optimization flags
DEVICE_TYPE = "cpu"
HAS_CUDA = torch.cuda.is_available()
HAS_MPS = torch.backends.mps.is_available()
HAS_GPU = HAS_CUDA or HAS_MPS
GPU_MEMORY_GB = 0  # Default for non-CUDA devices

if HAS_CUDA:
    DEVICE_TYPE = "cuda"
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"SNAC using CUDA: {GPU_NAME} ({GPU_MEMORY_GB:.1f} GB)")
elif HAS_MPS:
    DEVICE_TYPE = "mps"
    print("SNAC using Apple Metal Performance Shaders (MPS)")
else:
    DEVICE_TYPE = "cpu"
    print("SNAC using CPU")

# Try to enable torch.compile if PyTorch 2.0+ is available
TORCH_COMPILE_AVAILABLE = False
try:
    if hasattr(torch, 'compile') and HAS_CUDA:  # Only enable on CUDA for now
        TORCH_COMPILE_AVAILABLE = True
        print("PyTorch 2.0+ with CUDA detected, torch.compile is available")
except:
    pass

# Try to enable CUDA graphs if available
CUDA_GRAPHS_AVAILABLE = False
try:
    if HAS_CUDA and hasattr(torch.cuda, 'make_graphed_callables'):
        CUDA_GRAPHS_AVAILABLE = True
        print("CUDA graphs support is available")
except:
    pass

# Optimize based on device type
if DEVICE_TYPE == "cuda":
    # Set optimal CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("CUDA optimizations enabled: cuDNN benchmark, TF32")
elif DEVICE_TYPE == "cpu":
    # Optimize for CPU - use all available threads for parallel processing
    num_threads = torch.get_num_threads()
    print(f"CPU optimizations enabled: {num_threads} threads")

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

# Set device with proper fallback
snac_device = DEVICE_TYPE
print(f"SNAC model loaded on device: {snac_device}")
model = model.to(snac_device)

# Disable torch.compile as it requires Triton which isn't always installed
# We'll use regular PyTorch optimization techniques instead
print("Using standard PyTorch optimizations (torch.compile disabled)")

# Prepare CUDA streams for parallel processing if available
cuda_stream = None
if DEVICE_TYPE == "cuda":
    cuda_stream = torch.cuda.Stream()
    print("Using CUDA stream for parallel processing")


def convert_to_audio(multiframe, count):
    """
    Optimized version of convert_to_audio for both CPU and GPU with efficient tensor operations.
    Eliminates redundant operations and reduces CPU-GPU transfers.
    """
    if len(multiframe) < 7:
        return None
  
    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames*7]
    
    # Pre-allocate tensors on the target device
    codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=snac_device)
    codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=snac_device)
    codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=snac_device)
    
    # Use vectorized operations - much faster than loop concatenation
    frame_tensor = torch.tensor(frame, dtype=torch.int32, device=snac_device)
    
    # Direct indexing is much faster than concatenation in a loop
    for j in range(num_frames):
        idx = j * 7
        
        # Code 0 - single value per frame
        codes_0[j] = frame_tensor[idx]
        
        # Code 1 - two values per frame
        codes_1[j*2] = frame_tensor[idx+1]
        codes_1[j*2+1] = frame_tensor[idx+4]
        
        # Code 2 - four values per frame
        codes_2[j*4] = frame_tensor[idx+2]
        codes_2[j*4+1] = frame_tensor[idx+3]
        codes_2[j*4+2] = frame_tensor[idx+5]
        codes_2[j*4+3] = frame_tensor[idx+6]
    
    # Reshape codes into expected format
    codes = [
        codes_0.unsqueeze(0), 
        codes_1.unsqueeze(0), 
        codes_2.unsqueeze(0)
    ]
    
    # Check tokens are in valid range
    if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or 
        torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or 
        torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
        return None

    # Use CUDA stream for parallel processing if available, otherwise use no_grad
    stream_ctx = torch.cuda.stream(cuda_stream) if cuda_stream is not None else torch.no_grad()
    
    with stream_ctx, torch.inference_mode():
        # Decode the audio
        audio_hat = model.decode(codes)
        
        # Extract the relevant slice
        audio_slice = audio_hat[:, :, 2048:4096]
        
        # Optimize conversion based on device type
        if DEVICE_TYPE == "cuda":
            # Keep processing on GPU as long as possible
            audio_int16_tensor = (audio_slice * 32767).to(torch.int16)
            # Only transfer the final result to CPU
            audio_bytes = audio_int16_tensor.cpu().numpy().tobytes()
        elif DEVICE_TYPE == "mps":
            # MPS optimization - minimize data movement
            audio_int16_tensor = (audio_slice * 32767).to(torch.int16)
            audio_bytes = audio_int16_tensor.cpu().numpy().tobytes()
        else:
            # CPU - already on CPU, use NumPy directly
            audio_np = audio_slice.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
    
    # Periodic memory cleanup for CUDA to prevent memory fragmentation
    if DEVICE_TYPE == "cuda" and count % CUDA_CACHE_CLEAR_INTERVAL == 0:
        torch.cuda.empty_cache()
            
    return audio_bytes

def turn_token_into_id(token_string, index):
    """Optimized token-to-id conversion with early returns and minimal string operations"""
    token_string = token_string.strip()
    
    # Early return for obvious mismatches
    if "<custom_token_" not in token_string:
        return None
    
    # Find the last token in the string
    last_token_start = token_string.rfind("<custom_token_")
    if last_token_start == -1:
        return None
    
    # Check if the token ends properly
    if not token_string.endswith(">"):
        return None
        
    try:
        # Extract and convert the number directly
        number_str = token_string[last_token_start+14:-1]
        return int(number_str) - 10 - ((index % 7) * 4096)
    except (ValueError, IndexError):
        return None
# Cache for frequently processed tokens to avoid redundant computation
token_cache = {}
MAX_CACHE_SIZE = 1000  # Limit cache size to prevent memory bloat

async def tokens_decoder(token_gen):
    """Optimized token decoder with caching and conservative batch processing to ensure correct output"""
    buffer = []
    count = 0
    # Start with conservative parameters to ensure we get audio output
    min_frames_required = 28  # Default minimum frames (4 chunks of 7)
    process_every_n = 7  # Process every 7 tokens
    
    start_time = time.time()
    token_count = 0
    
    async for token_sim in token_gen:
        token_count += 1
        
        # Check cache first to avoid redundant computation
        cache_key = (token_sim, count % 7)
        if cache_key in token_cache:
            token = token_cache[cache_key]
        else:
            token = turn_token_into_id(token_sim, count)
            # Add to cache if valid token
            if token is not None and len(token_cache) < MAX_CACHE_SIZE:
                token_cache[cache_key] = token
        
        if token is not None and token > 0:
            buffer.append(token)
            count += 1

            # Process in larger batches for better GPU utilization
            if count % process_every_n == 0 and count >= min_frames_required:
                buffer_to_proc = buffer[-min_frames_required:]
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    # Log processing rate occasionally
                    if count % 140 == 0:  # Log every 20 chunks (assuming process_every_n=7)
                        elapsed = time.time() - start_time
                        tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
                        print(f"Processing speed: {tokens_per_sec:.1f} tokens/sec, buffer size: {len(buffer)}")
                    
                    yield audio_samples
    
    # Flush remaining buffer tokens at the end
    if len(buffer) >= min_frames_required:
        buffer_to_proc = buffer[-min_frames_required:]
        print(f"Flushing final buffer with {len(buffer_to_proc)} tokens")
        audio_samples = convert_to_audio(buffer_to_proc, count)
        if audio_samples is not None:
            yield audio_samples
# ------------------ Synchronous Tokens Decoder Wrapper ------------------ #
def tokens_decoder_sync(syn_token_gen):
    """Optimized synchronous decoder with device-appropriate settings"""
    # Use device-appropriate queue size based on GPU memory (computed at module load)
    if DEVICE_TYPE == "cuda":
        if GPU_MEMORY_GB >= 16:  # High-end GPU
            max_queue_size = 32
            batch_size = 16
        elif GPU_MEMORY_GB >= 8:  # Mid-range GPU
            max_queue_size = 24
            batch_size = 12
        else:  # Low-end GPU
            max_queue_size = 16
            batch_size = 8
    else:  # CPU or MPS
        max_queue_size = 8
        batch_size = 4
    
    audio_queue = queue.Queue(maxsize=max_queue_size)
    
    # Convert the synchronous token generator into an async generator with batching
    async def async_token_gen():
        token_batch = []
        for token in syn_token_gen:
            token_batch.append(token)
            # Process in batches for efficiency
            if len(token_batch) >= batch_size:
                for t in token_batch:
                    yield t
                token_batch = []
        # Process any remaining tokens
        for t in token_batch:
            yield t

    async def async_producer():
        # Start timer for performance logging
        start_time = time.time()
        chunk_count = 0
        
        try:
            # Process audio chunks from the token decoder
            async for audio_chunk in tokens_decoder(async_token_gen()):
                audio_queue.put(audio_chunk)
                chunk_count += 1
                
                # Log performance stats periodically
                if chunk_count % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Generated {chunk_count} chunks in {elapsed:.2f}s ({chunk_count/elapsed:.2f} chunks/sec)")
        except Exception as e:
            print(f"Error in async producer: {e}")
        finally:
            # Signal completion
            audio_queue.put(None)  # Sentinel

    def run_async():
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_producer())
        finally:
            loop.close()

    # Start the producer thread
    thread = threading.Thread(target=run_async)
    thread.daemon = True
    thread.start()

    # Use larger buffer for smoother playback
    buffer_size = 5
    audio_buffer = []
    
    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        
        audio_buffer.append(audio)
        # Yield buffered audio chunks for smoother playback
        if len(audio_buffer) >= buffer_size:
            for chunk in audio_buffer:
                yield chunk
            audio_buffer = []
    
    # Yield any remaining audio in the buffer
    for chunk in audio_buffer:
        yield chunk

    # Join thread with timeout to avoid hanging
    thread.join(timeout=30)
    if thread.is_alive():
        print("Warning: Audio producer thread did not terminate cleanly within timeout. "
              "This may indicate resource leaks or incomplete audio generation. "
              "Check for errors in token processing or audio conversion")
