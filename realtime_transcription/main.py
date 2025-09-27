import os
import sys
import asyncio
import sounddevice as sd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
import queue
import threading

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
BLOCKSIZE = 8000

# Global variable to store transcript
full_transcript = []
audio_queue = queue.Queue()

class TranscriptionClient:
    def __init__(self):
        self.client = DeepgramClient(DEEPGRAM_API_KEY)
        self.connection = None
        self.is_running = False
        self.stream = None
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        # Convert float32 to int16
        audio_data = (indata * 32767).astype(np.int16).tobytes()
        audio_queue.put(audio_data)
        
    async def start_transcription(self):
        try:
            # Create a websocket connection using the new API
            self.connection = self.client.listen.websocket.v("1")
            
            # Configure transcription options
            options = LiveOptions(
                model="nova-2",
                language="en-US",
                smart_format=True,
                punctuate=True,
                profanity_filter=False,
                interim_results=True,
                utterance_end_ms=1000,
                vad_events=True,
                encoding="linear16",
                sample_rate=RATE,
                channels=CHANNELS
            )
            
            # Set up event handlers
            self.connection.on(LiveTranscriptionEvents.Transcript, self.on_transcript)
            self.connection.on(LiveTranscriptionEvents.Error, self.on_error)
            self.connection.on(LiveTranscriptionEvents.Close, self.on_close)
            self.connection.on(LiveTranscriptionEvents.Open, self.on_open)
            
            # Start the connection - no await needed for the new API
            print("Connecting to Deepgram...")
            if self.connection.start(options):
                print("Connection initiated, waiting for confirmation...")
                # Give it a moment to connect
                await asyncio.sleep(1)
                
                if self.is_running:
                    await self.stream_audio()
            else:
                print("Failed to initiate connection to Deepgram")
                
        except Exception as e:
            print(f"Error starting transcription: {e}")
            
    def on_open(self, _, open_response):
        """Handle connection open event"""
        print("Connected to Deepgram! Start speaking... (Press Ctrl+C to stop)\n")
        print("-" * 50)
        self.is_running = True
        
    async def stream_audio(self):
        """Stream audio from microphone to Deepgram"""
        try:
            # Start sounddevice stream
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=CHANNELS,
                samplerate=RATE,
                blocksize=BLOCKSIZE,
                dtype=np.float32
            )
            
            with self.stream:
                print("Audio stream started. Listening...")
                
                # Send a keep-alive message periodically
                last_keepalive = asyncio.get_event_loop().time()
                
                while self.is_running:
                    try:
                        # Get audio data from queue (non-blocking with timeout)
                        try:
                            audio_data = audio_queue.get(timeout=0.05)
                            if self.connection and audio_data:
                                self.connection.send(audio_data)
                        except queue.Empty:
                            pass
                        
                        # Send keep-alive every 5 seconds
                        current_time = asyncio.get_event_loop().time()
                        if current_time - last_keepalive > 5:
                            if self.connection:
                                self.connection.keep_alive()
                                last_keepalive = current_time
                        
                        await asyncio.sleep(0.01)
                        
                    except Exception as e:
                        if self.is_running:  # Only print if we haven't stopped intentionally
                            print(f"Error in audio loop: {e}")
                        
        except KeyboardInterrupt:
            print("\n\nStopping transcription...")
            await self.stop_transcription()
        except Exception as e:
            print(f"Error streaming audio: {e}")
            
    async def stop_transcription(self):
        """Clean up and stop transcription"""
        self.is_running = False
        
        # Clear the audio queue
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
        if self.connection:
            try:
                self.connection.finish()
            except Exception as e:
                print(f"Error closing connection: {e}")
        
        # Save transcript to file
        self.save_transcript()
        
    def on_transcript(self, _, result):
        """Handle transcription results"""
        try:
            sentence = result.channel.alternatives[0].transcript
            
            if not sentence:
                return
                
            if result.is_final:
                # Final transcript - add to full transcript and display
                full_transcript.append(sentence)
                print(f"\r✓ {sentence}")
                print("> ", end="", flush=True)
            else:
                # Interim transcript - show as we type
                # Clear the line and show interim result
                print(f"\r> {sentence}...", end="", flush=True)
        except Exception as e:
            print(f"\nError processing transcript: {e}")
            
    def on_error(self, _, error):
        """Handle errors"""
        print(f"\nError: {error}")
        
    def on_close(self, _, close_event):
        """Handle connection close"""
        print("\nConnection closed")
        self.is_running = False
        
    def save_transcript(self):
        """Save the complete transcript to a file"""
        if full_transcript:
            with open("transcript.txt", "w", encoding="utf-8") as f:
                f.write(f"Transcript generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                for line in full_transcript:
                    f.write(f"{line}\n")
            print(f"\nTranscript saved to 'transcript.txt'")
            print(f"Total sentences transcribed: {len(full_transcript)}")
        else:
            print("\nNo transcript to save")

async def main():
    print("Deepgram Real-time Transcription")
    print("=" * 50)
    
    # List available audio devices
    print("\nAvailable audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            default_marker = " (DEFAULT)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {device['name']}{default_marker}")
    
    print(f"\nUsing default input device: {devices[sd.default.device[0]]['name']}")
    print("=" * 50 + "\n")
    
    client = TranscriptionClient()
    
    try:
        await client.start_transcription()
    except KeyboardInterrupt:
        print("\n\nTranscription stopped by user")
        await client.stop_transcription()
    except Exception as e:
        print(f"Error: {e}")
        if client.is_running:
            await client.stop_transcription()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)