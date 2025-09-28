# import os
# import sys
# import asyncio
# import sounddevice as sd
# import numpy as np
# import queue
# from datetime import datetime
# from dotenv import load_dotenv
# from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
# import time
# import uuid
# import json
# import threading
# import time
# import win32pipe
# import win32file

# # Load environment variables
# load_dotenv()

# # Configuration
# DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
# if not DEEPGRAM_API_KEY:
#     print("Error: DEEPGRAM_API_KEY not found in .env file", file=sys.stderr)
#     sys.exit(1)

# # Audio configuration
# RATE = 16000
# CHANNELS = 1
# BLOCKSIZE = 1024

# # Global variables
# full_transcript = []
# audio_queue = queue.Queue(maxsize=500)
# dropped_frames = 0


# class PipeServer:
#     """
#     Simple named-pipe server for main.py (sender).
#     - Creates the named pipe.
#     - Accepts one client connection in a background thread (ConnectNamedPipe).
#     - write(message) will attempt to WriteFile if connected.
#     """
#     def __init__(self, pipe_name: str):
#         self.pipe_name = pipe_name
#         self.pipe_handle = None
#         self.connected = False
#         self._accept_thread = None
#         self._lock = threading.Lock()
#         self._create_pipe()

#     def _create_pipe(self):
#         # Create a named pipe for outbound-only access from this process's perspective.
#         # Use message mode for clean JSON framing.
#         try:
#             self.pipe_handle = win32pipe.CreateNamedPipe(
#                 self.pipe_name,
#                 win32pipe.PIPE_ACCESS_OUTBOUND,  # server writes
#                 win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_WAIT,
#                 1,              # max instances
#                 65536,          # out buffer size
#                 65536,          # in buffer size (unused on outbound-only)
#                 0,              # default timeout (ms)
#                 None
#             )
#             # start accept thread
#             self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
#             self._accept_thread.start()
#             print(f"[PIPE DEBUG] Named pipe created: {self.pipe_name}", file=sys.stderr)
#         except Exception as e:
#             print(f"[PIPE ERROR] CreateNamedPipe failed: {e}", file=sys.stderr)
#             self.pipe_handle = None

#     def _accept_loop(self):
#         """Block waiting for a single client. If client disconnects, recreate the pipe."""
#         while True:
#             if not self.pipe_handle:
#                 time.sleep(1)
#                 continue
#             try:
#                 # This will block until a client connects
#                 print(f"[PIPE DEBUG] Waiting for client to connect to {self.pipe_name} ...", file=sys.stderr)
#                 win32pipe.ConnectNamedPipe(self.pipe_handle, None)
#                 with self._lock:
#                     self.connected = True
#                 print(f"[PIPE DEBUG] Client connected to pipe!", file=sys.stderr)

#                 # Block here until client disconnects
#                 # We won't actively read on server - but WriteFile will fail if disconnected.
#                 # Sleep-check loop to detect disconnection via WriteFile exceptions when they happen.
#                 while True:
#                     if not self.connected:
#                         break
#                     time.sleep(0.5)

#             except Exception as e:
#                 print(f"[PIPE DEBUG] Accept/connect exception: {e}", file=sys.stderr)
#             finally:
#                 # client disconnected or error -> try recreate pipe
#                 with self._lock:
#                     try:
#                         if self.pipe_handle:
#                             win32file.CloseHandle(self.pipe_handle)
#                     except Exception:
#                         pass
#                     self.connected = False
#                     self.pipe_handle = None
#                 time.sleep(0.5)
#                 # recreate pipe and loop
#                 try:
#                     self._create_pipe()
#                     return  # _create_pipe already starts a new accept thread
#                 except Exception:
#                     time.sleep(1)

#     def write(self, message: dict):
#         """Write JSON message (with newline) to the pipe if connected."""
#         if not self.pipe_handle:
#             print("[PIPE DEBUG] No pipe handle available to write", file=sys.stderr)
#             return False

#         if not self.connected:
#             # If not connected yet, do not block the main audio loop.
#             print("[PIPE DEBUG] No client connected yet; skipping write", file=sys.stderr)
#             return False

#         try:
#             s = json.dumps(message) + "\n"
#             data = s.encode("utf-8")
#             # win32file.WriteFile may raise if client disconnected
#             win32file.WriteFile(self.pipe_handle, data)
#             # Debug print to main terminal (you asked for outgoing printing)
#             print(f"[PIPE OUT] {s.strip()}", file=sys.stdout, flush=True)
#             print(f"[PIPE DEBUG] Wrote {len(data)} bytes to pipe", file=sys.stderr)
#             return True
#         except Exception as e:
#             print(f"[PIPE ERROR] WriteFile failed: {e}", file=sys.stderr)
#             with self._lock:
#                 self.connected = False
#             return False

#     def close(self):
#         try:
#             if self.pipe_handle:
#                 win32file.CloseHandle(self.pipe_handle)
#         except Exception:
#             pass
#         self.pipe_handle = None
#         self.connected = False
#         print("[PIPE DEBUG] PipeServer closed", file=sys.stderr)
# # ---------------------------------------------------------------------------

# class TranscriptionClient:
#     def __init__(self):
#         self.client = DeepgramClient(DEEPGRAM_API_KEY)
#         self.connection = None
#         self.is_running = False
#         self.stream = None
#         self.loop = None
#         self.audio_bytes_sent = 0
#         self.audio_chunks_processed = 0
#         self.start_time = None
        
#         self.pipe_name = r'\\.\pipe\deepgram_transcript'
#         self.pipe_handle = None
#         self.pipe = PipeServer(self.pipe_name)
        
#     def audio_callback(self, indata, frames, time, status):
#         """Callback for the audio stream to put data into our queue."""
#     #     volume_level = np.abs(indata).mean()
        
#     #     if volume_level < 0.001:
#     #         print(f"""
#     # [AUDIO WARNING] Very low audio levels detected:
#     # - Volume level: {volume_level:.6f}
#     # - Device: {sd.query_devices(sd.default.device[0], 'input')['name']}
#     # - Check if:
#     # 1. Microphone is muted
#     # 2. Wrong input device selected
#     # 3. Windows microphone permissions granted
#     # 4. Microphone volume in Windows settings""", file=sys.stderr)
#     #         return
        
#         device_info= sd.query_devices(sd.default.device[0], 'input')
#         print(f"[DEBUG] Using device: {device_info['name']}", file=sys.stderr)
#         print(f"[DEBUG] Audio levels: min={indata.min():.3f}, max={indata.max():.3f}", file=sys.stderr)
#         global dropped_frames
        
#         if status:
#             if status.input_overflow:
#                 dropped_frames += 1
#                 print(f"⚠️  Input overflow! Dropped frames: {dropped_frames}", file=sys.stderr)
#             else:
#                 print(f"Audio callback status: {status}", file=sys.stderr)
        
#         audio_data = (indata * 32767).astype(np.int16).tobytes()
        
#         try:
#             audio_queue.put_nowait(audio_data)
#         except queue.Full:
#             dropped_frames += 1
#             try:
#                 audio_queue.get_nowait()
#                 audio_queue.put_nowait(audio_data)
#                 print(f"⚠️  Queue overflow! Total dropped: {dropped_frames}", file=sys.stderr)
#             except:
#                 pass

#     async def start_transcription(self):
#         """Initiates the connection to Deepgram and starts audio streaming."""
#         self.loop = asyncio.get_running_loop()
#         self.start_time = time.time()
#         TIMEOUT_MINUTES = 45
        
#         try:
#             async def timeout_handler():
#                 await asyncio.sleep(TIMEOUT_MINUTES * 60)
#                 print(f"\n⏰ Timeout reached ({TIMEOUT_MINUTES} minutes). Stopping transcription...", file=sys.stderr)
#                 self.is_running = False
            
#             timeout_task = self.loop.create_task(timeout_handler())    
#             self.connection = self.client.listen.websocket.v("1") # what does this do

#             options = LiveOptions(
#                 model="nova-2",
#                 language="en-US",
#                 interim_results=True,
#                 encoding="linear16",
#                 sample_rate=RATE,
#                 channels=CHANNELS,
#             )
            
#             # options = LiveOptions(
#             #     model="nova-2",
#             #     language="en-US",
#             #     smart_format=True,
#             #     interim_results=True,
#             #     encoding="linear16",
#             #     sample_rate=RATE,
#             #     channels=CHANNELS,
#             #     punctuate=True,
#             #     profanity_filter=False,
#             #     redact=False,
#             #     diarize=False,
#             #     numerals=True,
#             #     utterance_end_ms=1000
#             # )
            
#             self.connection.on(LiveTranscriptionEvents.Transcript, self.on_transcript)
#             self.connection.on(LiveTranscriptionEvents.Error, self.on_error)
#             self.connection.on(LiveTranscriptionEvents.Close, self.on_close)
#             self.connection.on(LiveTranscriptionEvents.Open, self.on_open)
            
#             print("Connecting to Deepgram...", file=sys.stderr)
#             if self.connection.start(options):
#                 print("Connection initiated, waiting for confirmation...", file=sys.stderr)
#                 for _ in range(50):
#                     if self.is_running:
#                         break
#                     await asyncio.sleep(0.1)
                
#                 if self.is_running:
#                     await self.stream_audio()
#                     timeout_task.cancel()
#                 else:
#                     print("Connection timeout - Deepgram did not confirm connection", file=sys.stderr)
#             else:
#                 print("Failed to initiate connection to Deepgram", file=sys.stderr)
                
#         except Exception as e:
#             print(f"Error starting transcription: {e}", file=sys.stderr)
#         finally:
#             if not timeout_task.done():
#                 timeout_task.cancel()

#     def on_open(self, _, open_response):
#         """Handles the connection open event from Deepgram."""
#         print("Connected to Deepgram! Start speaking... (Press Ctrl+C to stop)\n", file=sys.stderr)
#         print("-" * 50, file=sys.stderr)
#         self.is_running = True

#     async def stream_audio(self):
#         """Starts the microphone stream and sends audio data to Deepgram."""
#         self.stream = sd.InputStream(
#             device= sd.default.device[0],
#             callback=self.audio_callback,
#             channels=CHANNELS,
#             samplerate=RATE,
#             blocksize=BLOCKSIZE,
#             dtype=np.float32,
#             latency='low',
#             extra_settings=None,
#         )
        
#         with self.stream:
#             print("Audio stream started. Listening...", file=sys.stderr)
#             print(f"Audio device: {sd.query_devices(sd.default.device[0], 'input')['name']}", file=sys.stderr)
#             print(f"Sample rate: {RATE} Hz, Blocksize: {BLOCKSIZE} samples", file=sys.stderr)
#             print("-" * 50, file=sys.stderr)
            
#             last_keepalive = asyncio.get_event_loop().time()
            
#             while self.is_running:
#                 # main.py only sends, no need to check for incoming pipe messages
                
#                 chunks_to_send = []
#                 deadline = time.time() + 0.2 # can configure
                
#                 while time.time() < deadline:
#                     try:
#                         audio_data = audio_queue.get_nowait()
#                         chunks_to_send.append(audio_data)
#                         self.audio_chunks_processed += 1
#                     except queue.Empty:
#                         break
                
#                 if chunks_to_send and self.connection:
#                     try:
#                         combined_audio = b''.join(chunks_to_send)
#                         print(f"[DEBUG] Sending {len(combined_audio)} bytes of audio", file=sys.stderr)
#                         self.connection.send(combined_audio)
#                         self.audio_bytes_sent += len(combined_audio)
#                     except Exception as e:
#                         print(f"Error sending audio: {e}", file=sys.stderr)
                
#                 current_time = asyncio.get_event_loop().time()
#                 if current_time - last_keepalive > 5:
#                     if self.connection:
#                         try:
#                             self.connection.keep_alive()
#                             last_keepalive = current_time
#                         except:
#                             pass
                
#                 await asyncio.sleep(0.001)
            
#             print("\nAudio streaming stopped", file=sys.stderr)

#     async def stop_transcription(self):
#         """Cleans up all resources."""
#         self.is_running = False
        
#         remaining_chunks = []
#         while not audio_queue.empty():
#             try:
#                 remaining_chunks.append(audio_queue.get_nowait())
#             except queue.Empty:
#                 break
        
#         if remaining_chunks and self.connection:
#             try:
#                 print(f"Sending {len(remaining_chunks)} remaining audio chunks...", file=sys.stderr)
#                 combined_audio = b''.join(remaining_chunks)
#                 self.connection.send(combined_audio)
#                 await asyncio.sleep(2)
#             except:
#                 pass
        
#         if self.stream:
#             self.stream.stop()
#             self.stream.close()
            
#         if self.connection:
#             self.connection.finish()
        
#         if self.start_time:
#             duration = time.time() - self.start_time
#             print("\n" + "=" * 50, file=sys.stderr)
#             print("Session Statistics:", file=sys.stderr)
#             print(f"Duration: {duration:.1f} seconds", file=sys.stderr)
#             print(f"Audio chunks processed: {self.audio_chunks_processed}", file=sys.stderr)
#             print(f"Audio bytes sent: {self.audio_bytes_sent:,}", file=sys.stderr)
#             print(f"Dropped frames: {dropped_frames}", file=sys.stderr)
#             print(f"Average throughput: {self.audio_bytes_sent/duration:.0f} bytes/sec", file=sys.stderr)
#             print("=" * 50, file=sys.stderr)
        
#         self.save_transcript()
        
#         if self.pipe_handle:
#             try:
#                 win32file.CloseHandle(self.pipe_handle)
#                 print("[PIPE DEBUG] Closed pipe handle", file=sys.stderr)
#             except:
#                 pass

#     def on_transcript(self, _, result):
#         """Handles transcription results from Deepgram."""
#         try:
#             sentence = result.channel.alternatives[0].transcript
#             if not sentence:
#                 print("[TRANSCRIPT DEBUG] Empty transcript received; skipping", file=sys.stderr)
#                 return
#             else:
#                 print(f"[TRANSCRIPT DEBUG] Transcript received: '{sentence}'", file=sys.stderr)
#             # Send through named pipe
#             message = {
#                 'type': 'audio',
#                 'text': sentence,
#                 'is_final': result.is_final,
#                 'timestamp': time.time()
#             }
#             self.pipe.write(message)
            
#             # Keep for local logging only on stderr
#             timestamp = datetime.now().strftime("%H:%M:%S")
#             if result.is_final:
#                 full_transcript.append(sentence)
#                 print(f"[{timestamp}] FINAL: {sentence}", flush=True)
#                 print(f"[{timestamp}] ✓ FINAL: {sentence}", file=sys.stderr)
#             else:
#                 print(f"[{timestamp}] INTERIM: {sentence}", flush=True)
#                 print(f"[{timestamp}] INTERIM: {sentence}", file=sys.stderr)
                
#         except Exception as e:
#             print(f"\nError processing transcript: {e}", file=sys.stderr)
    
#     def on_error(self, _, error):
#         print(f"\n⚠️  Deepgram Error: {error}", file=sys.stderr)
        
#     def on_close(self, _, close_event):
#         self.pipe.close()
#         print("\n📡 Deepgram connection closed", file=sys.stderr)
#         self.is_running = False
        
#     def save_transcript(self):
#         """Saves the final transcript to a file."""
#         if full_transcript:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = f"transcript_{timestamp}.txt"
            
#             with open(filename, "w", encoding="utf-8") as f:
#                 f.write(f"Transcript generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#                 f.write(f"Duration: {time.time() - self.start_time:.1f} seconds\n")
#                 f.write(f"Total segments: {len(full_transcript)}\n")
#                 f.write("=" * 50 + "\n\n")
                
#                 for i, segment in enumerate(full_transcript, 1):
#                     f.write(f"{i}. {segment}\n")
            
#             print(f"\n✅ Transcript saved to '{filename}'", file=sys.stderr)
#             print(f"   Total segments: {len(full_transcript)}", file=sys.stderr)
#         else:
#             print("\n⚠️  No transcript to save", file=sys.stderr)

# async def main():
#     print("\n" + "=" * 50, file=sys.stderr)
#     print("🎙️  Deepgram Real-time Transcription (SENDER ONLY)", file=sys.stderr)
#     print("=" * 50, file=sys.stderr)
    
#     print("\nAvailable audio devices:", file=sys.stderr)
#     print("-" * 30, file=sys.stderr)
#     devices = sd.query_devices()
#     for i, device in enumerate(devices):
#         if device['max_input_channels'] > 0:
#             marker = "📍" if i == sd.default.device[0] else "  "
#             print(f"{marker} [{i}] {device['name']} ({device['max_input_channels']} channels)", file=sys.stderr)
#     print("-" * 30, file=sys.stderr)
    
#     client = TranscriptionClient()
    
#     try:
#         # Send start message through stdout (for agent.py to read)
#         session_id = str(uuid.uuid4())
#         start_msg = {
#             "action": "start",
#             "session_id": session_id
#         }
#         print(json.dumps(start_msg), flush=True)
#         print(f"[STDOUT DEBUG] Sent session start: {json.dumps(start_msg)}", file=sys.stderr)
        
#         print(f"Session started with ID: {session_id}", file=sys.stderr)
#         print("=" * 50 + "\n", file=sys.stderr)
        
#         await client.start_transcription()
        
#     except KeyboardInterrupt:
#         print("\n\n⏹️  Transcription stopped by user", file=sys.stderr)
#     except Exception as e:
#         print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
#     finally:
#         await client.stop_transcription()
#         print("🔌 Session ended.", file=sys.stderr)

# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         print("\n👋 Exiting...", file=sys.stderr)
#         sys.exit(0)

# # import os
# # import sys
# # import asyncio
# # import sounddevice as sd
# # import numpy as np
# # import queue
# # from datetime import datetime
# # from dotenv import load_dotenv
# # from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
# # import time
# # import uuid
# # import json

# # import win32pipe
# # import win32file

# # # Load environment variables
# # load_dotenv()

# # # Configuration
# # DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
# # if not DEEPGRAM_API_KEY:
# #     print("Error: DEEPGRAM_API_KEY not found in .env file", file=sys.stderr)
# #     sys.exit(1)

# # # Audio configuration
# # RATE = 16000
# # CHANNELS = 1
# # BLOCKSIZE = 2048

# # # Global variables
# # full_transcript = []
# # audio_queue = queue.Queue(maxsize=500)
# # dropped_frames = 0

# # class TranscriptionClient:
# #     def __init__(self):
# #         self.client = DeepgramClient(DEEPGRAM_API_KEY)
# #         self.connection = None
# #         self.is_running = False
# #         self.stream = None
# #         self.loop = None
# #         self.audio_bytes_sent = 0
# #         self.audio_chunks_processed = 0
# #         self.start_time = None
        
# #         self.pipe_name = r'\\.\pipe\deepgram_transcript'
# #         self.pipe_handle = None
# #         self.setup_pipe()
    
# #     def setup_pipe(self):
# #         """Creates a named pipe for inter-process communication - OUTBOUND ONLY"""
# #         try:
# #             # Create OUTBOUND-only pipe - main.py only sends, agent.py only receives
# #             self.pipe_handle = win32pipe.CreateNamedPipe(
# #                 self.pipe_name,
# #                 win32pipe.PIPE_ACCESS_OUTBOUND,  # Only for sending
# #                 win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
# #                 1, 65536, 65536,
# #                 0,
# #                 None
# #             )
# #             print(f"[PIPE DEBUG] Created OUTBOUND-only named pipe: {self.pipe_name}", file=sys.stderr)
# #         except Exception as e:
# #             print(f"[PIPE ERROR] Error creating pipe: {e}", file=sys.stderr)
# #             sys.exit(1)
    
# #     def send_to_pipe(self, message):
# #         """Sends a message through the named pipe"""
# #         try:
# #             if self.pipe_handle:
# #                 message_str = json.dumps(message) + '\n'
# #                 message_bytes = message_str.encode('utf-8')
                
# #                 # Debug: Print what we're sending
# #                 print(f"[PIPE DEBUG] Sending to pipe: {message_str.strip()}", file=sys.stderr)
                
# #                 # Try to connect if no client is connected
# #                 try:
# #                     win32pipe.ConnectNamedPipe(self.pipe_handle, None)
# #                     print(f"[PIPE DEBUG] Connected to client", file=sys.stderr)
# #                 except Exception as connect_error:
# #                     # Pipe might already be connected, that's okay
# #                     print(f"[PIPE DEBUG] Connect result: {connect_error}", file=sys.stderr)
                
# #                 # Send the message
# #                 win32file.WriteFile(self.pipe_handle, message_bytes)
# #                 print(f"[PIPE DEBUG] Successfully sent {len(message_bytes)} bytes", file=sys.stderr)
                
# #         except Exception as e:
# #             print(f"[PIPE ERROR] Error writing to pipe: {e}", file=sys.stderr)
            
# #     def audio_callback(self, indata, frames, time, status):
# #         """Callback for the audio stream to put data into our queue."""
# #         global dropped_frames
        
# #         if status:
# #             if status.input_overflow:
# #                 dropped_frames += 1
# #                 print(f"⚠️  Input overflow! Dropped frames: {dropped_frames}", file=sys.stderr)
# #             else:
# #                 print(f"Audio callback status: {status}", file=sys.stderr)
        
# #         audio_data = (indata * 32767).astype(np.int16).tobytes()
        
# #         try:
# #             audio_queue.put_nowait(audio_data)
# #         except queue.Full:
# #             dropped_frames += 1
# #             try:
# #                 audio_queue.get_nowait()
# #                 audio_queue.put_nowait(audio_data)
# #                 print(f"⚠️  Queue overflow! Total dropped: {dropped_frames}", file=sys.stderr)
# #             except:
# #                 pass

# #     async def start_transcription(self):
# #         """Initiates the connection to Deepgram and starts audio streaming."""
# #         self.loop = asyncio.get_running_loop()
# #         self.start_time = time.time()
        
# #         try:
# #             self.connection = self.client.listen.websocket.v("1")
            
# #             options = LiveOptions(
# #                 model="nova-2",
# #                 language="en-US",
# #                 smart_format=True,
# #                 interim_results=True,
# #                 encoding="linear16",
# #                 sample_rate=RATE,
# #                 channels=CHANNELS,
# #                 punctuate=True,
# #                 profanity_filter=False,
# #                 redact=False,
# #                 diarize=False,
# #                 numerals=True,
# #                 utterance_end_ms=1000
# #             )
            
# #             self.connection.on(LiveTranscriptionEvents.Transcript, self.on_transcript)
# #             self.connection.on(LiveTranscriptionEvents.Error, self.on_error)
# #             self.connection.on(LiveTranscriptionEvents.Close, self.on_close)
# #             self.connection.on(LiveTranscriptionEvents.Open, self.on_open)
            
# #             print("Connecting to Deepgram...", file=sys.stderr)
# #             if self.connection.start(options):
# #                 print("Connection initiated, waiting for confirmation...", file=sys.stderr)
# #                 for _ in range(50):
# #                     if self.is_running:
# #                         break
# #                     await asyncio.sleep(0.1)
                
# #                 if self.is_running:
# #                     await self.stream_audio()
# #                 else:
# #                     print("Connection timeout - Deepgram did not confirm connection", file=sys.stderr)
# #             else:
# #                 print("Failed to initiate connection to Deepgram", file=sys.stderr)
                
# #         except Exception as e:
# #             print(f"Error starting transcription: {e}", file=sys.stderr)

# #     def on_open(self, _, open_response):
# #         """Handles the connection open event from Deepgram."""
# #         print("Connected to Deepgram! Start speaking... (Press Ctrl+C to stop)\n", file=sys.stderr)
# #         print("-" * 50, file=sys.stderr)
# #         self.is_running = True

# #     async def stream_audio(self):
# #         """Starts the microphone stream and sends audio data to Deepgram."""
# #         self.stream = sd.InputStream(
# #             callback=self.audio_callback,
# #             channels=CHANNELS,
# #             samplerate=RATE,
# #             blocksize=BLOCKSIZE,
# #             dtype=np.float32,
# #             latency='high'
# #         )
        
# #         with self.stream:
# #             print("Audio stream started. Listening...", file=sys.stderr)
# #             print(f"Audio device: {sd.query_devices(sd.default.device[0], 'input')['name']}", file=sys.stderr)
# #             print(f"Sample rate: {RATE} Hz, Blocksize: {BLOCKSIZE} samples", file=sys.stderr)
# #             print("-" * 50, file=sys.stderr)
            
# #             last_keepalive = asyncio.get_event_loop().time()
            
# #             while self.is_running:
# #                 # main.py only sends, no need to check for incoming pipe messages
                
# #                 chunks_to_send = []
# #                 deadline = time.time() + 0.05
                
# #                 while time.time() < deadline:
# #                     try:
# #                         audio_data = audio_queue.get_nowait()
# #                         chunks_to_send.append(audio_data)
# #                         self.audio_chunks_processed += 1
# #                     except queue.Empty:
# #                         break
                
# #                 if chunks_to_send and self.connection:
# #                     try:
# #                         combined_audio = b''.join(chunks_to_send)
# #                         self.connection.send(combined_audio)
# #                         self.audio_bytes_sent += len(combined_audio)
# #                     except Exception as e:
# #                         print(f"Error sending audio: {e}", file=sys.stderr)
                
# #                 current_time = asyncio.get_event_loop().time()
# #                 if current_time - last_keepalive > 5:
# #                     if self.connection:
# #                         try:
# #                             self.connection.keep_alive()
# #                             last_keepalive = current_time
# #                         except:
# #                             pass
                
# #                 await asyncio.sleep(0.001)
            
# #             print("\nAudio streaming stopped", file=sys.stderr)

# #     async def stop_transcription(self):
# #         """Cleans up all resources."""
# #         self.is_running = False
        
# #         remaining_chunks = []
# #         while not audio_queue.empty():
# #             try:
# #                 remaining_chunks.append(audio_queue.get_nowait())
# #             except queue.Empty:
# #                 break
        
# #         if remaining_chunks and self.connection:
# #             try:
# #                 print(f"Sending {len(remaining_chunks)} remaining audio chunks...", file=sys.stderr)
# #                 combined_audio = b''.join(remaining_chunks)
# #                 self.connection.send(combined_audio)
# #                 await asyncio.sleep(2)
# #             except:
# #                 pass
        
# #         if self.stream:
# #             self.stream.stop()
# #             self.stream.close()
            
# #         if self.connection:
# #             self.connection.finish()
        
# #         if self.start_time:
# #             duration = time.time() - self.start_time
# #             print("\n" + "=" * 50, file=sys.stderr)
# #             print("Session Statistics:", file=sys.stderr)
# #             print(f"Duration: {duration:.1f} seconds", file=sys.stderr)
# #             print(f"Audio chunks processed: {self.audio_chunks_processed}", file=sys.stderr)
# #             print(f"Audio bytes sent: {self.audio_bytes_sent:,}", file=sys.stderr)
# #             print(f"Dropped frames: {dropped_frames}", file=sys.stderr)
# #             print(f"Average throughput: {self.audio_bytes_sent/duration:.0f} bytes/sec", file=sys.stderr)
# #             print("=" * 50, file=sys.stderr)
        
# #         self.save_transcript()
        
# #         if self.pipe_handle:
# #             try:
# #                 win32file.CloseHandle(self.pipe_handle)
# #                 print("[PIPE DEBUG] Closed pipe handle", file=sys.stderr)
# #             except:
# #                 pass

# #     def on_transcript(self, _, result):
# #         """Handles transcription results from Deepgram."""
# #         try:
# #             sentence = result.channel.alternatives[0].transcript
# #             if not sentence:
# #                 return

# #             # Send through named pipe
# #             message = {
# #                 'type': 'audio',
# #                 'text': sentence,
# #                 'is_final': result.is_final,
# #                 'timestamp': time.time()
# #             }
# #             self.send_to_pipe(message)

# #             # Keep for local logging only on stderr
# #             timestamp = datetime.now().strftime("%H:%M:%S")
# #             if result.is_final:
# #                 full_transcript.append(sentence)
# #                 print(f"[{timestamp}] ✓ FINAL: {sentence}", file=sys.stderr)
# #             else:
# #                 print(f"[{timestamp}] INTERIM: {sentence}", file=sys.stderr)
                
# #         except Exception as e:
# #             print(f"\nError processing transcript: {e}", file=sys.stderr)
    
# #     def on_error(self, _, error):
# #         print(f"\n⚠️  Deepgram Error: {error}", file=sys.stderr)
        
# #     def on_close(self, _, close_event):
# #         print("\n📡 Deepgram connection closed", file=sys.stderr)
# #         self.is_running = False
        
# #     def save_transcript(self):
# #         """Saves the final transcript to a file."""
# #         if full_transcript:
# #             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# #             filename = f"transcript_{timestamp}.txt"
            
# #             with open(filename, "w", encoding="utf-8") as f:
# #                 f.write(f"Transcript generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
# #                 f.write(f"Duration: {time.time() - self.start_time:.1f} seconds\n")
# #                 f.write(f"Total segments: {len(full_transcript)}\n")
# #                 f.write("=" * 50 + "\n\n")
                
# #                 for i, segment in enumerate(full_transcript, 1):
# #                     f.write(f"{i}. {segment}\n")
            
# #             print(f"\n✅ Transcript saved to '{filename}'", file=sys.stderr)
# #             print(f"   Total segments: {len(full_transcript)}", file=sys.stderr)
# #         else:
# #             print("\n⚠️  No transcript to save", file=sys.stderr)

# # async def main():
# #     print("\n" + "=" * 50, file=sys.stderr)
# #     print("🎙️  Deepgram Real-time Transcription (SENDER ONLY)", file=sys.stderr)
# #     print("=" * 50, file=sys.stderr)
    
# #     print("\nAvailable audio devices:", file=sys.stderr)
# #     print("-" * 30, file=sys.stderr)
# #     devices = sd.query_devices()
# #     for i, device in enumerate(devices):
# #         if device['max_input_channels'] > 0:
# #             marker = "📍" if i == sd.default.device[0] else "  "
# #             print(f"{marker} [{i}] {device['name']} ({device['max_input_channels']} channels)", file=sys.stderr)
# #     print("-" * 30, file=sys.stderr)
    
# #     client = TranscriptionClient()
    
# #     try:
# #         # Send start message through stdout (for agent.py to read)
# #         session_id = str(uuid.uuid4())
# #         start_msg = {
# #             "action": "start",
# #             "session_id": session_id
# #         }
# #         print(json.dumps(start_msg), flush=True)
# #         print(f"[STDOUT DEBUG] Sent session start: {json.dumps(start_msg)}", file=sys.stderr)
        
# #         print(f"Session started with ID: {session_id}", file=sys.stderr)
# #         print("=" * 50 + "\n", file=sys.stderr)
        
# #         await client.start_transcription()
        
# #     except KeyboardInterrupt:
# #         print("\n\n⏹️  Transcription stopped by user", file=sys.stderr)
# #     except Exception as e:
# #         print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
# #     finally:
# #         await client.stop_transcription()
# #         print("🔌 Session ended.", file=sys.stderr)

# # if __name__ == "__main__":
# #     try:
# #         asyncio.run(main())
# #     except KeyboardInterrupt:
# #         print("\n👋 Exiting...", file=sys.stderr)
# #         sys.exit(0)


import os
import sys
import asyncio
import sounddevice as sd
import numpy as np
import queue
import websockets
from datetime import datetime
from dotenv import load_dotenv
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
import threading
import time


load_dotenv()


DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    print("Error: DEEPGRAM_API_KEY not found in .env file")
    sys.exit(1)


RATE = 16000
CHANNELS = 1

BLOCKSIZE = 2048  


full_transcript = []

audio_queue = queue.Queue(maxsize=500)
dropped_frames = 0

class TranscriptionClient:
    def __init__(self):
        self.client = DeepgramClient(DEEPGRAM_API_KEY)
        self.connection = None
        self.is_running = False
        self.stream = None
        
        self.connected_clients = set()
        
        self.loop = None
        
        self.audio_bytes_sent = 0
        self.audio_chunks_processed = 0
        self.start_time = None

    def audio_callback(self, indata, frames, time, status):
        """Callback for the audio stream to put data into our queue."""
        global dropped_frames
        
        if status:
            if status.input_overflow:
                dropped_frames += 1
                print(f"⚠️  Input overflow! Dropped frames: {dropped_frames}")
            else:
                print(f"Audio callback status: {status}")
        
        
        audio_data = (indata * 32767).astype(np.int16).tobytes()
        
        
        try:
            audio_queue.put_nowait(audio_data)
        except queue.Full:
            dropped_frames += 1
            
            try:
                audio_queue.get_nowait()
                audio_queue.put_nowait(audio_data)
                print(f"⚠️  Queue overflow! Total dropped: {dropped_frames}")
            except:
                pass

    async def _websocket_handler(self, websocket):
        """Handles incoming WebSocket connections and keeps them alive."""
        print(f"Client connected from {websocket.remote_address}")
        self.connected_clients.add(websocket)
        try:
            
            await websocket.wait_closed()
        finally:
            print(f"Client from {websocket.remote_address} disconnected")
            self.connected_clients.remove(websocket)

    async def _broadcast(self, message):
        """Sends a message to all connected WebSocket clients."""
        if self.connected_clients:
            
            disconnected = set()
            for client in self.connected_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            
            self.connected_clients -= disconnected

    async def start_transcription(self):
        """Initiates the connection to Deepgram and starts audio streaming."""
        
        self.loop = asyncio.get_running_loop()
        self.start_time = time.time()
        
        try:
            self.connection = self.client.listen.websocket.v("1")
            
            options = LiveOptions(
                model="nova-2",
                language="en-US",
                smart_format=True,
                interim_results=True,
                encoding="linear16",
                sample_rate=RATE,
                channels=CHANNELS,
                
                punctuate=True,
                profanity_filter=False,
                redact=False,
                diarize=False,
                numerals=True,
                utterance_end_ms=1000
            )
            
            self.connection.on(LiveTranscriptionEvents.Transcript, self.on_transcript)
            self.connection.on(LiveTranscriptionEvents.Error, self.on_error)
            self.connection.on(LiveTranscriptionEvents.Close, self.on_close)
            self.connection.on(LiveTranscriptionEvents.Open, self.on_open)
            
            print("Connecting to Deepgram...")
            if self.connection.start(options):
                print("Connection initiated, waiting for confirmation...")
                
                for _ in range(50):  
                    if self.is_running:
                        break
                    await asyncio.sleep(0.1)
                
                if self.is_running:
                    
                    await self.stream_audio()
                else:
                    print("Connection timeout - Deepgram did not confirm connection")
            else:
                print("Failed to initiate connection to Deepgram")
                
        except Exception as e:
            print(f"Error starting transcription: {e}")

    def on_open(self, _, open_response):
        """Handles the connection open event from Deepgram."""
        print("Connected to Deepgram! Start speaking... (Press Ctrl+C to stop)\n")
        print("-" * 50)
        self.is_running = True

    async def stream_audio(self):
        """Starts the microphone stream and sends audio data to Deepgram."""
        
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=CHANNELS,
            samplerate=RATE,
            blocksize=BLOCKSIZE,
            dtype=np.float32,
            
            latency='high'
        )
        
        with self.stream:
            print("Audio stream started. Listening...")
            print(f"Audio device: {sd.query_devices(sd.default.device[0], 'input')['name']}")
            print(f"Sample rate: {RATE} Hz, Blocksize: {BLOCKSIZE} samples")
            print("-" * 50)
            
            last_keepalive = asyncio.get_event_loop().time()
            audio_buffer = bytearray()
            
            
            while self.is_running:
                
                chunks_to_send = []
                
                
                deadline = time.time() + 0.05  
                
                while time.time() < deadline:
                    try:
                        audio_data = audio_queue.get_nowait()
                        chunks_to_send.append(audio_data)
                        self.audio_chunks_processed += 1
                    except queue.Empty:
                        break
                
                
                if chunks_to_send and self.connection:
                    try:
                        
                        combined_audio = b''.join(chunks_to_send)
                        self.connection.send(combined_audio)
                        self.audio_bytes_sent += len(combined_audio)
                    except Exception as e:
                        print(f"Error sending audio: {e}")
                
                
                current_time = asyncio.get_event_loop().time()
                if current_time - last_keepalive > 5:
                    if self.connection:
                        try:
                            self.connection.keep_alive()
                            last_keepalive = current_time
                        except:
                            pass
                
                
                await asyncio.sleep(0.001)
            
            print("\nAudio streaming stopped")

    async def stop_transcription(self):
        """Cleans up all resources."""
        self.is_running = False
        
        
        remaining_chunks = []
        while not audio_queue.empty():
            try:
                remaining_chunks.append(audio_queue.get_nowait())
            except queue.Empty:
                break
        
        if remaining_chunks and self.connection:
            try:
                print(f"Sending {len(remaining_chunks)} remaining audio chunks...")
                combined_audio = b''.join(remaining_chunks)
                self.connection.send(combined_audio)
                
                await asyncio.sleep(2)
            except:
                pass
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
        if self.connection:
            self.connection.finish()
        
        
        if self.start_time:
            duration = time.time() - self.start_time
            print("\n" + "=" * 50)
            print("Session Statistics:")
            print(f"Duration: {duration:.1f} seconds")
            print(f"Audio chunks processed: {self.audio_chunks_processed}")
            print(f"Audio bytes sent: {self.audio_bytes_sent:,}")
            print(f"Dropped frames: {dropped_frames}")
            print(f"Average throughput: {self.audio_bytes_sent/duration:.0f} bytes/sec")
            print("=" * 50)
        
        self.save_transcript()

    def on_transcript(self, _, result):
        """Handles transcription results from Deepgram."""
        try:
            sentence = result.channel.alternatives[0].transcript
            if not sentence:
                return

            
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    self._broadcast(sentence), 
                    self.loop
                )
            
            if result.is_final:
                full_transcript.append(sentence)
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\r[{timestamp}] ✓ {sentence}")
                print("> ", end="", flush=True)
            else:
                
                print(f"\r> {sentence}...", end="", flush=True)
        except Exception as e:
            print(f"\nError processing transcript: {e}")
            
    def on_error(self, _, error):
        print(f"\n⚠️  Deepgram Error: {error}")
        
    def on_close(self, _, close_event):
        print("\n📡 Deepgram connection closed")
        self.is_running = False
        
    def save_transcript(self):
        """Saves the final transcript to a file."""
        if full_transcript:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcript_{timestamp}.txt"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Transcript generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {time.time() - self.start_time:.1f} seconds\n")
                f.write(f"Total segments: {len(full_transcript)}\n")
                f.write("=" * 50 + "\n\n")
                
                for i, segment in enumerate(full_transcript, 1):
                    f.write(f"{i}. {segment}\n")
            
            print(f"\n✅ Transcript saved to '{filename}'")
            print(f"   Total segments: {len(full_transcript)}")
        else:
            print("\n⚠️  No transcript to save")

async def main():
    print("\n" + "=" * 50)
    print("🎙️  Deepgram Real-time Transcription")
    print("=" * 50)
    
    
    print("\nAvailable audio devices:")
    print("-" * 30)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            marker = "📍" if i == sd.default.device[0] else "  "
            print(f"{marker} [{i}] {device['name']} ({device['max_input_channels']} channels)")
    print("-" * 30)
    
    client = TranscriptionClient()
    
    
    websocket_server = await websockets.serve(
        client._websocket_handler, 
        "localhost", 
        8766
    )
    print("\n✅ WebSocket server started on ws://localhost:8766")
    print("=" * 50 + "\n")

    try:
        await client.start_transcription()
    except KeyboardInterrupt:
        print("\n\n⏹️  Transcription stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        await client.stop_transcription()
        websocket_server.close()
        await websocket_server.wait_closed()
        print("WebSocket server stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        sys.exit(0)