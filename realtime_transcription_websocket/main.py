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

# Load environment variables
load_dotenv()

# Configuration
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    print("Error: DEEPGRAM_API_KEY not found in .env file")
    sys.exit(1)

# Audio configuration
RATE = 16000
CHANNELS = 1
# Smaller blocksize for more frequent callbacks and less latency
BLOCKSIZE = 2048  # Reduced from 8000 for better real-time capture

# Global variables
full_transcript = []
# Larger queue size to prevent overflow
audio_queue = queue.Queue(maxsize=100)
dropped_frames = 0

class TranscriptionClient:
    def __init__(self):
        self.client = DeepgramClient(DEEPGRAM_API_KEY)
        self.connection = None
        self.is_running = False
        self.stream = None
        # A set to hold all connected WebSocket clients
        self.connected_clients = set()
        # A placeholder for the main asyncio event loop
        self.loop = None
        # Statistics for monitoring
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
        
        # Convert to int16 format that Deepgram expects
        audio_data = (indata * 32767).astype(np.int16).tobytes()
        
        # Try to put data in queue, but don't block if queue is full
        try:
            audio_queue.put_nowait(audio_data)
        except queue.Full:
            dropped_frames += 1
            # If queue is full, remove oldest item and add new one
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
            # Keep the connection open until the client disconnects
            await websocket.wait_closed()
        finally:
            print(f"Client from {websocket.remote_address} disconnected")
            self.connected_clients.remove(websocket)

    async def _broadcast(self, message):
        """Sends a message to all connected WebSocket clients."""
        if self.connected_clients:
            # Use gather to send to all clients concurrently
            disconnected = set()
            for client in self.connected_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            # Remove disconnected clients
            self.connected_clients -= disconnected

    async def start_transcription(self):
        """Initiates the connection to Deepgram and starts audio streaming."""
        # Capture the current event loop to be used by the callback thread
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
                # Add more options for better transcription
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
                # Wait for the connection to be fully established
                for _ in range(50):  # Wait up to 5 seconds
                    if self.is_running:
                        break
                    await asyncio.sleep(0.1)
                
                if self.is_running:
                    # Start audio streaming in a separate task
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
        # Configure the audio stream with better buffer settings
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=CHANNELS,
            samplerate=RATE,
            blocksize=BLOCKSIZE,
            dtype=np.float32,
            # Increase latency for more stable capture
            latency='high'
        )
        
        with self.stream:
            print("Audio stream started. Listening...")
            print(f"Audio device: {sd.query_devices(sd.default.device[0], 'input')['name']}")
            print(f"Sample rate: {RATE} Hz, Blocksize: {BLOCKSIZE} samples")
            print("-" * 50)
            
            last_keepalive = asyncio.get_event_loop().time()
            audio_buffer = bytearray()
            
            # Process audio in a more efficient loop
            while self.is_running:
                # Collect multiple audio chunks before sending
                chunks_to_send = []
                
                # Try to get all available audio data without blocking too long
                deadline = time.time() + 0.05  # 50ms window
                
                while time.time() < deadline:
                    try:
                        audio_data = audio_queue.get_nowait()
                        chunks_to_send.append(audio_data)
                        self.audio_chunks_processed += 1
                    except queue.Empty:
                        break
                
                # Send all collected chunks
                if chunks_to_send and self.connection:
                    try:
                        # Combine chunks for more efficient sending
                        combined_audio = b''.join(chunks_to_send)
                        self.connection.send(combined_audio)
                        self.audio_bytes_sent += len(combined_audio)
                    except Exception as e:
                        print(f"Error sending audio: {e}")
                
                # Send keep-alive periodically
                current_time = asyncio.get_event_loop().time()
                if current_time - last_keepalive > 5:
                    if self.connection:
                        try:
                            self.connection.keep_alive()
                            last_keepalive = current_time
                        except:
                            pass
                
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.001)
            
            print("\nAudio streaming stopped")

    async def stop_transcription(self):
        """Cleans up all resources."""
        self.is_running = False
        
        # Process any remaining audio in the queue
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
                # Give Deepgram time to process final audio
                await asyncio.sleep(2)
            except:
                pass
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
        if self.connection:
            self.connection.finish()
        
        # Print statistics
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

            # Safely schedule the broadcast coroutine on the main event loop
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    self._broadcast(sentence), 
                    self.loop
                )
            
            if result.is_final:
                full_transcript.append(sentence)
                # Add timestamp for better tracking
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\r[{timestamp}] ✓ {sentence}")
                print("> ", end="", flush=True)
            else:
                # Show interim results with different formatting
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
    
    # Check audio devices
    print("\nAvailable audio devices:")
    print("-" * 30)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            marker = "📍" if i == sd.default.device[0] else "  "
            print(f"{marker} [{i}] {device['name']} ({device['max_input_channels']} channels)")
    print("-" * 30)
    
    client = TranscriptionClient()
    
    # Start the WebSocket server
    websocket_server = await websockets.serve(
        client._websocket_handler, 
        "localhost", 
        8765
    )
    print("\n✅ WebSocket server started on ws://localhost:8765")
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