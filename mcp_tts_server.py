"""MCP Server for text-to-speech notifications with timer support.

This server implements the Model Context Protocol (MCP) to provide
text-to-speech capabilities for AI assistants to speak notifications
audibly to humans.

This implementation supports:
- Multiple transport methods (stdio, streamable HTTP)
- Roots for resource boundaries
- Sampling for enhanced notification text generation
- Tools for text-to-speech and notifications
- Timer functionality with automatic notifications
- Comprehensive debugging and logging
- Inspector compatibility
"""

import json
import logging
import os
import tempfile
import time
import traceback
import uuid
import math
import numpy as np
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

# Suppress pygame startup messages that can interfere with JSON communication
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

from gtts import gTTS
from scipy.io import wavfile
from fastmcp import FastMCP

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not available, using system environment variables only

# Try to import Google Cloud TTS - optional feature
try:
    from google.cloud import texttospeech
    GOOGLE_CLOUD_TTS_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_TTS_AVAILABLE = False

# Try to import Google Cloud Speech-to-Text - optional feature
try:
    from google.cloud import speech
    import pyaudio
    GOOGLE_CLOUD_STT_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_STT_AVAILABLE = False

# Configure logging
log_level = os.getenv("MCP_TTS_LOG_LEVEL", "INFO")
DEBUG_MODE = os.getenv("MCP_TTS_DEBUG", "false").lower() in ("true", "1", "yes")
TRACE_REQUESTS = os.getenv("MCP_TTS_TRACE_REQUESTS", "false").lower() in ("true", "1", "yes")

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add request_id to the logging context
class RequestContextFilter(logging.Filter):
    """Add request_id to log records."""
    
    def __init__(self, name="", default_request_id="no-request"):
        super().__init__(name)
        self.default_request_id = default_request_id
        self.request_id = default_request_id
    
    def set_request_id(self, request_id):
        """Set the current request ID."""
        self.request_id = request_id
    
    def reset_request_id(self):
        """Reset to default request ID."""
        self.request_id = self.default_request_id
    
    def filter(self, record):
        if not hasattr(record, 'request_id'):
            record.request_id = self.request_id
        return True

# Create and add the filter
request_context = RequestContextFilter()
logger.addFilter(request_context)

# Log startup information
try:
    from dotenv import load_dotenv
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.info("python-dotenv not available, using system environment variables only")

if GOOGLE_CLOUD_TTS_AVAILABLE:
    logger.info("Google Cloud Text-to-Speech library is available")
else:
    logger.info("Google Cloud Text-to-Speech library not installed - using gTTS only")

# Initialize pygame for audio playback
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
    logger.info("Pygame mixer initialized successfully", extra={"request_id": "startup"})
except Exception as e:
    logger.error(f"Failed to initialize pygame mixer: {e}", extra={"request_id": "startup"})
    logger.error(f"Audio playback will not work", extra={"request_id": "startup"})
    
# Bell sound settings
BELL_SETTINGS = {
    "standard": {
        "frequency": 880,  # A5
        "duration": 0.5,   # seconds
        "amplitude": 0.5,  # volume (0.0 to 1.0)
        "fade": 0.1        # fade out duration (seconds)
    },
    "success": {
        "frequency": 1046.5,  # C6
        "duration": 0.4,
        "amplitude": 0.5,
        "fade": 0.1
    },
    "warning": {
        "frequency": 698.46,  # F5
        "duration": 0.6,
        "amplitude": 0.6,
        "fade": 0.1
    },
    "error": {
        "frequency": 587.33,  # D5
        "duration": 0.7,
        "amplitude": 0.7,
        "fade": 0.15
    }
}

# Configuration
SERVER_VERSION = "1.3.0"
DEFAULT_VOICE = os.getenv("MCP_TTS_DEFAULT_VOICE", "en")
USE_HTTP_TRANSPORT = os.getenv("MCP_TTS_HTTP_TRANSPORT", "false").lower() in ("true", "1", "yes")
HTTP_PORT = int(os.getenv("MCP_TTS_HTTP_PORT", "8080"))
NOTIFICATIONS_ROOT = os.getenv("MCP_TTS_ROOT", "file:///notifications/tts")
MOCK_MODE = os.getenv("MCP_TTS_MOCK_MODE", "false").lower() in ("true", "1", "yes")

# Google Cloud TTS Configuration
GOOGLE_CLOUD_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
USE_GOOGLE_CLOUD_TTS = os.getenv("MCP_TTS_USE_GOOGLE_CLOUD", "false").lower() in ("true", "1", "yes")
GOOGLE_CLOUD_TTS_VOICE_TYPE = os.getenv("MCP_TTS_GOOGLE_VOICE_TYPE", "standard")  # standard or wavenet

# Google Cloud Speech-to-Text Configuration
USE_GOOGLE_CLOUD_STT = os.getenv("MCP_TTS_USE_GOOGLE_CLOUD_STT", "true").lower() in ("true", "1", "yes")
STT_LANGUAGE_CODE = os.getenv("MCP_TTS_STT_LANGUAGE", "en-US")
STT_SAMPLE_RATE = int(os.getenv("MCP_TTS_STT_SAMPLE_RATE", "16000"))

# Initialize MCP server
mcp = FastMCP(
    "tts-notify"
)

# Initialize Google Cloud TTS client if available and configured
google_tts_client = None
if GOOGLE_CLOUD_TTS_AVAILABLE and USE_GOOGLE_CLOUD_TTS:
    try:
        if GOOGLE_CLOUD_CREDENTIALS_PATH and os.path.exists(GOOGLE_CLOUD_CREDENTIALS_PATH):
            google_tts_client = texttospeech.TextToSpeechClient()
            logger.info(f"Google Cloud TTS client initialized with credentials from {GOOGLE_CLOUD_CREDENTIALS_PATH}")
        else:
            # Try to use default credentials (if running on GCP or gcloud auth is set up)
            google_tts_client = texttospeech.TextToSpeechClient()
            logger.info("Google Cloud TTS client initialized with default credentials")
    except Exception as e:
        logger.warning(f"Failed to initialize Google Cloud TTS client: {e}")
        logger.info("Falling back to gTTS for text-to-speech")
        google_tts_client = None
elif USE_GOOGLE_CLOUD_TTS and not GOOGLE_CLOUD_TTS_AVAILABLE:
    logger.warning("Google Cloud TTS requested but library not installed. Install with: pip install google-cloud-texttospeech")
else:
    logger.info("Using gTTS for text-to-speech")

# Initialize Google Cloud Speech-to-Text client if available and configured
google_stt_client = None
if GOOGLE_CLOUD_STT_AVAILABLE and USE_GOOGLE_CLOUD_STT:
    try:
        if GOOGLE_CLOUD_CREDENTIALS_PATH and os.path.exists(GOOGLE_CLOUD_CREDENTIALS_PATH):
            google_stt_client = speech.SpeechClient()
            logger.info(f"Google Cloud STT client initialized with credentials from {GOOGLE_CLOUD_CREDENTIALS_PATH}")
        else:
            # Try to use default credentials
            google_stt_client = speech.SpeechClient()
            logger.info("Google Cloud STT client initialized with default credentials")
    except Exception as e:
        logger.warning(f"Failed to initialize Google Cloud STT client: {e}")
        logger.info("Speech-to-text will not be available")
        google_stt_client = None
elif USE_GOOGLE_CLOUD_STT and not GOOGLE_CLOUD_STT_AVAILABLE:
    logger.warning("Google Cloud STT requested but library not installed. Install with: pip install google-cloud-speech pyaudio")
else:
    logger.info("Speech-to-text not enabled")

# Define roots
ROOTS = [
    {
        "uri": NOTIFICATIONS_ROOT,
        "name": "Text-to-Speech Notifications"
    }
]

# Define example notification resources
RESOURCES = {
    f"{NOTIFICATIONS_ROOT}/templates/success.txt": "Task completed successfully.",
    f"{NOTIFICATIONS_ROOT}/templates/error.txt": "An error occurred during processing.",
    f"{NOTIFICATIONS_ROOT}/templates/warning.txt": "Warning: unusual behavior detected.",
    f"{NOTIFICATIONS_ROOT}/templates/info.txt": "New information is available."
}

# Define the tool schema
TOOL_SCHEMA = {
    "speak": {
        "name": "speak",
        "description": "Speaks the provided text aloud using text-to-speech. Supports both free gTTS and premium Google Cloud TTS (if configured).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to speak aloud"
                },
                "voice": {
                    "type": "string",
                    "description": "Language/voice code (e.g., 'en', 'fr', 'en-US', 'fr-FR'). For Google Cloud TTS use full locale codes.",
                    "default": DEFAULT_VOICE
                }
            },
            "required": ["text"]
        },
        "annotations": {
            "title": "Speak Text",
            "readOnlyHint": True,  # Tool doesn't modify data
            "openWorldHint": False  # Fixed parameter set
        }
    },
    "notify": {
        "name": "notify",
        "description": "Speaks a notification message with a standardized format",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The notification message to speak"
                },
                "type": {
                    "type": "string",
                    "description": "Type of notification",
                    "enum": ["info", "success", "warning", "error"],
                    "default": "info"
                },
                "voice": {
                    "type": "string",
                    "description": "Language/voice code (e.g., 'en', 'fr', 'en-uk')",
                    "default": DEFAULT_VOICE
                },
                "enhance": {
                    "type": "boolean",
                    "description": "Use LLM to enhance the notification text",
                    "default": False
                }
            },
            "required": ["message"]
        },
        "annotations": {
            "title": "Speak Notification",
            "readOnlyHint": True,
            "openWorldHint": False
        }
    },
    "bell": {
        "name": "bell",
        "description": "Play a bell sound to alert the user without speaking text",
        "inputSchema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "Type of bell sound",
                    "enum": ["standard", "success", "warning", "error"],
                    "default": "standard"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of times to ring the bell",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 1
                }
            }
        },
        "annotations": {
            "title": "Ring Bell",
            "readOnlyHint": True,
            "openWorldHint": False
        }
    },
    "smart_notify": {
        "name": "smart_notify",
        "description": "Generate and speak a notification based on event context",
        "inputSchema": {
            "type": "object",
            "properties": {
                "event": {
                    "type": "string",
                    "description": "The event to notify about (e.g., 'task_complete', 'error')"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context about the event"
                },
                "type": {
                    "type": "string",
                    "description": "Type of notification",
                    "enum": ["info", "success", "warning", "error"],
                    "default": "info"
                },
                "voice": {
                    "type": "string",
                    "description": "Language/voice code (e.g., 'en', 'fr', 'en-uk')",
                    "default": DEFAULT_VOICE
                }
            },
            "required": ["event"]
        },
        "annotations": {
            "title": "Smart Notification",
            "readOnlyHint": True,
            "openWorldHint": False
        }
    },
    "test_notification": {
        "name": "test_notification",
        "description": "Send a test notification to verify the server is working",
        "inputSchema": {
            "type": "object",
            "properties": {
                "voice": {
                    "type": "string",
                    "description": "Language/voice code (e.g., 'en', 'fr', 'en-uk')",
                    "default": DEFAULT_VOICE
                }
            }
        },
        "annotations": {
            "title": "Test Notification",
            "readOnlyHint": True,
            "openWorldHint": False,
            "isDebugTool": True
        }
    },
    "listen": {
        "name": "listen",
        "description": "Listen for speech input and convert it to text using speech-to-text",
        "inputSchema": {
            "type": "object",
            "properties": {
                "duration": {
                    "type": "number",
                    "description": "Duration to listen in seconds (default: 5)",
                    "default": 5
                },
                "language": {
                    "type": "string", 
                    "description": "Language code for speech recognition (e.g., 'en-US', 'fr-FR')",
                    "default": STT_LANGUAGE_CODE
                }
            }
        },
        "annotations": {
            "title": "Listen for Speech",
            "readOnlyHint": True,
            "openWorldHint": False
        }
    },
    "tts_status": {
        "name": "tts_status",
        "description": "Get information about available TTS services and current configuration",
        "inputSchema": {
            "type": "object",
            "properties": {}
        },
        "annotations": {
            "title": "TTS Status",
            "readOnlyHint": True,
            "openWorldHint": False,
            "isDebugTool": True
        }
    },
    "set_timer": {
        "name": "set_timer",
        "description": "Set a timer that will notify you with a spoken message when it expires",
        "inputSchema": {
            "type": "object",
            "properties": {
                "duration": {
                    "type": "number",
                    "description": "Timer duration in seconds",
                    "minimum": 1,
                    "maximum": 86400
                },
                "message": {
                    "type": "string",
                    "description": "Message to speak when timer expires"
                },
                "voice": {
                    "type": "string",
                    "description": "Language/voice code (e.g., 'en', 'fr', 'en-US')",
                    "default": DEFAULT_VOICE
                },
                "type": {
                    "type": "string",
                    "description": "Type of notification",
                    "enum": ["info", "success", "warning", "error"],
                    "default": "info"
                },
                "repeat": {
                    "type": "boolean",
                    "description": "Whether to repeat the timer notification",
                    "default": False
                },
                "repeat_interval": {
                    "type": "number",
                    "description": "Interval between repeat notifications in seconds (if repeat is true)",
                    "minimum": 1,
                    "maximum": 3600
                }
            },
            "required": ["duration", "message"]
        },
        "annotations": {
            "title": "Set Timer",
            "readOnlyHint": False,
            "openWorldHint": False
        }
    },
    "list_timers": {
        "name": "list_timers",
        "description": "List all active timers",
        "inputSchema": {
            "type": "object",
            "properties": {}
        },
        "annotations": {
            "title": "List Timers",
            "readOnlyHint": True,
            "openWorldHint": False
        }
    },
    "cancel_timer": {
        "name": "cancel_timer",
        "description": "Cancel an active timer by its ID",
        "inputSchema": {
            "type": "object",
            "properties": {
                "timer_id": {
                    "type": "string",
                    "description": "The ID of the timer to cancel"
                }
            },
            "required": ["timer_id"]
        },
        "annotations": {
            "title": "Cancel Timer",
            "readOnlyHint": False,
            "openWorldHint": False
        }
    },
    "cancel_all_timers": {
        "name": "cancel_all_timers",
        "description": "Cancel all active timers",
        "inputSchema": {
            "type": "object",
            "properties": {}
        },
        "annotations": {
            "title": "Cancel All Timers",
            "readOnlyHint": False,
            "openWorldHint": False
        }
    }
}

# Performance tracking
performance_stats = {
    "tool_calls": {},
    "speech_synthesis": {
        "count": 0,
        "total_time": 0,
        "avg_time": 0,
        "max_time": 0
    }
}


# Timer management
timers = {}  # Dictionary to store active timers
timer_counter = 0  # Counter for generating unique timer IDs
timer_lock = threading.Lock()  # Thread lock for timer operations


class Timer:
    """Timer class for managing timed notifications."""
    
    def __init__(self, timer_id: str, duration: float, message: str, voice: str = None, 
                 timer_type: str = "info", repeat: bool = False, repeat_interval: float = None):
        self.timer_id = timer_id
        self.duration = duration
        self.message = message
        self.voice = voice or DEFAULT_VOICE
        self.timer_type = timer_type
        self.repeat = repeat
        self.repeat_interval = repeat_interval or duration
        self.created_at = datetime.now()
        self.expires_at = self.created_at + timedelta(seconds=duration)
        self.is_active = True
        self.thread = None
        self.notification_count = 0
    
    def __repr__(self):
        return f"Timer({self.timer_id}, expires_at={self.expires_at}, message='{self.message[:20]}...')"


def timer_thread_worker(timer: Timer):
    """Worker function that runs in a separate thread for each timer."""
    try:
        # Wait for the timer duration
        time.sleep(timer.duration)
        
        # Check if timer is still active (not cancelled)
        with timer_lock:
            if not timer.is_active or timer.timer_id not in timers:
                logger.debug(f"Timer {timer.timer_id} was cancelled before expiry")
                return
        
        # Execute the timer notification
        # Use asyncio.run() to create a new event loop for this thread
        asyncio.run(execute_timer_notification(timer))
        
        # Handle repeat timers
        if timer.repeat and timer.is_active:
            # Schedule next notification
            timer.expires_at = datetime.now() + timedelta(seconds=timer.repeat_interval)
            timer.notification_count += 1
            new_thread = threading.Thread(target=timer_thread_worker, args=(timer,))
            new_thread.daemon = True
            timer.thread = new_thread
            new_thread.start()
            logger.info(f"Repeat timer {timer.timer_id} scheduled for next notification")
    
    except Exception as e:
        logger.error(f"Error in timer thread {timer.timer_id}: {e}")
        logger.debug(traceback.format_exc())


async def execute_timer_notification(timer: Timer):
    """Execute the notification for an expired timer."""
    try:
        # Generate request ID for this timer notification
        request_id = f"timer-{timer.timer_id}-{uuid.uuid4().hex[:8]}"
        request_context.set_request_id(request_id)
        
        logger.info(f"Timer {timer.timer_id} expired, executing notification")
        
        # Format notification message
        if timer.repeat and timer.notification_count > 0:
            message = f"Timer reminder #{timer.notification_count + 1}: {timer.message}"
        else:
            message = f"Timer notification: {timer.message}"
        
        # Add type prefix
        prefix = ""
        if timer.timer_type == "success":
            prefix = "Success! "
        elif timer.timer_type == "warning":
            prefix = "Warning! "
        elif timer.timer_type == "error":
            prefix = "Error! "
        elif timer.timer_type == "info":
            prefix = "Timer: "
        
        full_message = f"{prefix}{message}"
        
        # Speak the notification
        success, error = await synthesize_and_play(full_message, timer.voice, request_id)
        
        if success:
            logger.info(f"Timer notification spoken successfully for {timer.timer_id}")
        else:
            logger.error(f"Failed to speak timer notification for {timer.timer_id}: {error}")
        
        # Remove timer if it's not repeating or if it failed
        if not timer.repeat or not success:
            with timer_lock:
                if timer.timer_id in timers:
                    del timers[timer.timer_id]
                    logger.info(f"Timer {timer.timer_id} removed")
    
    except Exception as e:
        logger.error(f"Error executing timer notification {timer.timer_id}: {e}")
        logger.debug(traceback.format_exc())


def track_timing(category: str, name: str, duration: float):
    """Track performance timing for a category and operation."""
    if category not in performance_stats:
        performance_stats[category] = {}
    
    if name not in performance_stats[category]:
        performance_stats[category][name] = {
            "count": 0,
            "total_time": 0,
            "avg_time": 0,
            "max_time": 0
        }
    
    stats = performance_stats[category][name]
    stats["count"] += 1
    stats["total_time"] += duration
    stats["avg_time"] = stats["total_time"] / stats["count"]
    stats["max_time"] = max(stats["max_time"], duration)


async def listen_for_speech(duration: float = 5.0, language_code: str = STT_LANGUAGE_CODE) -> Tuple[bool, Optional[str], Optional[str]]:
    """Listen for speech input and convert to text using Google Cloud STT.
    
    Args:
        duration: Duration to listen in seconds
        language_code: Language code for speech recognition
        
    Returns:
        Tuple of (success, transcribed_text, error_message)
    """
    if not google_stt_client:
        return False, None, "Google Cloud Speech-to-Text client not available"
    
    try:
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        
        # Audio recording parameters
        chunk = 1024
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 1
        fs = STT_SAMPLE_RATE  # Record at sample rate
        
        logger.info(f"Listening for speech for {duration} seconds...")
        
        # Start recording
        stream = audio.open(format=sample_format,
                           channels=channels,
                           rate=fs,
                           frames_per_buffer=chunk,
                           input=True)
        
        frames = []  # Initialize array to store frames
        
        # Store data in chunks for duration seconds
        for i in range(0, int(fs / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        logger.info("Finished recording, processing speech...")
        
        # Convert recorded audio to bytes
        audio_data = b''.join(frames)
        
        # Configure the audio for Google Cloud STT
        audio_config = speech.RecognitionAudio(content=audio_data)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=fs,
            language_code=language_code,
        )
        
        # Perform the transcription
        response = google_stt_client.recognize(config=config, audio=audio_config)
        
        # Extract the transcribed text
        if response.results:
            transcript = response.results[0].alternatives[0].transcript
            confidence = response.results[0].alternatives[0].confidence
            logger.info(f"Transcribed: '{transcript}' (confidence: {confidence:.2f})")
            return True, transcript, None
        else:
            logger.info("No speech detected or could not transcribe")
            return False, None, "No speech detected or could not transcribe"
            
    except Exception as e:
        error_msg = f"Error during speech recognition: {e}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        return False, None, error_msg


def generate_bell_sound(bell_type: str, count: int = 1) -> Tuple[bool, Optional[str]]:
    """Generate and play a bell sound.
    
    Args:
        bell_type: Type of bell sound to play
        count: Number of times to play the sound
        
    Returns:
        Tuple of (success, error_message)
    """
    if count < 1 or count > 5:
        return False, "Count must be between 1 and 5"
    
    if bell_type not in BELL_SETTINGS:
        return False, f"Unknown bell type: {bell_type}"
    
    # Get bell settings
    settings = BELL_SETTINGS[bell_type]
    frequency = settings["frequency"]
    duration = settings["duration"]
    amplitude = settings["amplitude"]
    fade = settings["fade"]
    
    try:
        # Generate bell sound
        sample_rate = 44100  # Hz
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        
        # Generate a sine wave
        tone = amplitude * np.sin(frequency * 2 * np.pi * t)
        
        # Apply fade out
        fade_samples = int(fade * sample_rate)
        if fade_samples > 0:
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            tone[-fade_samples:] *= fade_out
        
        # Convert to 16-bit PCM
        audio = (tone * 32767).astype(np.int16)
        
        # Create a stereo sound (duplicate the mono channel)
        stereo = np.column_stack((audio, audio))
        
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        # Save the audio to the temporary file
        wavfile.write(temp_filename, sample_rate, stereo)
        
        # Play the bell sound(s)
        sound = pygame.mixer.Sound(temp_filename)
        
        for i in range(count):
            sound.play()
            # Wait for sound to finish
            time.sleep(duration + 0.1)
        
        # Clean up
        os.unlink(temp_filename)
        
        return True, None
        
    except Exception as e:
        error_msg = f"Error generating or playing bell sound: {e}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        return False, error_msg


async def synthesize_with_google_cloud(text: str, voice: str = DEFAULT_VOICE, request_id: str = None) -> Tuple[bool, Optional[str]]:
    """Convert text to speech using Google Cloud TTS and play it.
    
    Args:
        text: The text to convert to speech
        voice: Language/voice code (e.g., 'en-US', 'fr-FR')
        request_id: Current request ID for logging
        
    Returns:
        Tuple of (success, error_message)
    """
    if not google_tts_client:
        return False, "Google Cloud TTS client not available"
    
    if not text:
        logger.warning("Empty text provided for Google Cloud TTS synthesis")
        return False, "Empty text provided"
    
    # Set the request context for logging
    if request_id:
        request_context.set_request_id(request_id)
    
    try:
        start_time = time.time()
        
        # Parse voice string to extract language and voice name
        language_code = voice if '-' in voice else f"{voice}-US"
        
        # Determine voice type based on configuration
        if GOOGLE_CLOUD_TTS_VOICE_TYPE.lower() == "wavenet":
            voice_name = f"{language_code}-Wavenet-A"
        else:
            voice_name = f"{language_code}-Standard-A"
        
        # Create the synthesis input
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Build the voice request
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )
        
        # Select the type of audio file to return
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        logger.debug(f"Generating speech with Google Cloud TTS: text='{text}', voice={voice_name}")
        
        # Perform the text-to-speech request
        response = google_tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config
        )
        
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(response.audio_content)
            logger.debug(f"Created temporary file: {temp_filename}")
        
        # Play the speech
        logger.info(f"Playing Google Cloud TTS synthesized speech: '{text}'")
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        # Clean up
        pygame.mixer.music.unload()
        os.unlink(temp_filename)
        logger.debug(f"Cleaned up temporary file: {temp_filename}")
        
        # Track performance
        duration = time.time() - start_time
        track_timing("speech_synthesis", f"google-{GOOGLE_CLOUD_TTS_VOICE_TYPE}", duration)
        logger.debug(f"Google Cloud TTS synthesis completed in {duration:.2f} seconds")
        
        return True, None
        
    except Exception as e:
        error_msg = f"Error with Google Cloud TTS synthesis: {e}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        return False, error_msg


async def synthesize_and_play(text: str, voice: str = DEFAULT_VOICE, request_id: str = None) -> Tuple[bool, Optional[str]]:
    """Convert text to speech and play it.
    
    Args:
        text: The text to convert to speech
        voice: Language/voice code
        request_id: Current request ID for logging
        
    Returns:
        Tuple of (success, error_message)
    """
    if not text:
        logger.warning("Empty text provided for speech synthesis")
        return False, "Empty text provided"
    
    # Set the request context for logging
    if request_id:
        request_context.set_request_id(request_id)
    
    # Mock mode for testing
    if MOCK_MODE:
        logger.info(f"MOCK MODE: Would speak: '{text}' with voice '{voice}'")
        time.sleep(0.5)  # Simulate speech synthesis time
        return True, None
    
    # Try Google Cloud TTS first if available and enabled
    if google_tts_client and USE_GOOGLE_CLOUD_TTS:
        logger.debug("Using Google Cloud Text-to-Speech")
        success, error = await synthesize_with_google_cloud(text, voice, request_id)
        if success:
            return success, error
        else:
            logger.warning(f"Google Cloud TTS failed: {error}. Falling back to gTTS.")
    
    # Fall back to gTTS
    logger.debug("Using gTTS for speech synthesis")
    try:
        start_time = time.time()
        
        # Create a temporary file for the speech audio
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_filename = temp_file.name
            logger.debug(f"Created temporary file: {temp_filename}")
        
        # Generate speech using gTTS
        logger.debug(f"Generating speech with gTTS: text='{text}', lang={voice}")
        tts = gTTS(text=text, lang=voice, slow=False)
        tts.save(temp_filename)
        
        # Play the speech
        logger.info(f"Playing synthesized speech: '{text}'")
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        # Clean up
        pygame.mixer.music.unload()
        os.unlink(temp_filename)
        logger.debug(f"Cleaned up temporary file: {temp_filename}")
        
        # Track performance
        duration = time.time() - start_time
        track_timing("speech_synthesis", voice, duration)
        logger.debug(f"Speech synthesis completed in {duration:.2f} seconds")
        
        # Send log message to client
        try:
            if request_id and hasattr(mcp, 'send_log_message') and callable(getattr(mcp, 'send_log_message')):
                await mcp.send_log_message({
                    "level": "info",
                    "message": f"Spoke message: '{text}' in {duration:.2f} seconds"
                })
        except Exception as e:
            logger.debug(f"Could not send log message to client: {e}")
        
        return True, None
        
    except Exception as e:
        error_msg = f"Error synthesizing or playing speech: {e}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        return False, error_msg


@mcp.tool
async def listen(params: Dict[str, Any]):
    """Listen for speech input and convert it to text using speech-to-text.
    
    Args:
        params: Dictionary containing:
            - duration: Duration to listen in seconds (optional, default: 5)
            - language: Language code for speech recognition (optional)
        
    Returns:
        Tool response with transcribed text or error
    """
    # Generate unique request ID
    request_id = f"listen-{uuid.uuid4().hex[:8]}"
    request_context.set_request_id(request_id)
    
    # Log the request if tracing is enabled
    if TRACE_REQUESTS:
        logger.debug(f"Tool call received: listen with params: {params}")
    
    start_time = time.time()
    
    duration = params.get("duration", 5.0)
    language = params.get("language", STT_LANGUAGE_CODE)
    
    # Validate duration
    if duration < 1 or duration > 30:
        logger.warning(f"Invalid duration: {duration}, clamping to 1-30 seconds")
        duration = max(1, min(30, duration))
    
    logger.info(f"Starting speech recognition for {duration} seconds in {language}")
    
    success, transcript, error = await listen_for_speech(duration, language)
    
    # Track performance
    duration_actual = time.time() - start_time
    track_timing("tool_calls", "listen", duration_actual)
    
    if success:
        logger.info(f"Successfully transcribed speech: '{transcript}' (completed in {duration_actual:.2f}s)")
        return {
            "success": True,
            "content": {
                "result": "Successfully transcribed speech",
                "transcript": transcript,
                "duration_seconds": round(duration_actual, 2),
                "language": language
            }
        }
    else:
        logger.error(f"Failed to transcribe speech: {error}")
        return {
            "success": False,
            "content": {"error": error or "Failed to transcribe speech"}
        }


@mcp.tool
async def speak(params: Dict[str, Any]):
    """Speaks the given text aloud using text-to-speech.
    
    Args:
        params: Dictionary containing:
            - text: The text to speak
            - voice: Language/voice code (optional)
        
    Returns:
        Tool response indicating success or failure
    """
    # Generate unique request ID
    request_id = f"speak-{uuid.uuid4().hex[:8]}"
    request_context.set_request_id(request_id)
    
    # Log the request if tracing is enabled
    if TRACE_REQUESTS:
        logger.debug(f"Tool call received: speak with params: {params}")
    
    start_time = time.time()
    
    text = params.get("text", "")
    voice = params.get("voice", DEFAULT_VOICE)
    
    if not text:
        logger.warning("No text provided to speak")
        return {
            "success": False,
            "content": {"error": "No text provided to speak"}
        }
    
    logger.info(f"Speaking text: '{text}' with voice: {voice}")
    
    success, error = await synthesize_and_play(text, voice, request_id)
    
    # Track performance
    duration = time.time() - start_time
    track_timing("tool_calls", "speak", duration)
    
    if success:
        logger.info(f"Successfully spoke text (completed in {duration:.2f}s)")
        return {
            "success": True,
            "content": {
                "result": f"Successfully spoke: '{text}'",
                "duration_seconds": round(duration, 2)
            }
        }
    else:
        logger.error(f"Failed to speak text: {error}")
        return {
            "success": False,
            "content": {"error": error or "Failed to synthesize or play speech"}
        }


@mcp.tool
async def notify(params: Dict[str, Any]):
    """Speaks a notification with a specific format based on the type.
    
    Args:
        params: Dictionary containing:
            - message: The notification message
            - type: Type of notification (info, success, warning, error)
            - voice: Language/voice code (optional)
            - enhance: Whether to enhance the notification with LLM (optional)
        
    Returns:
        Tool response indicating success or failure
    """
    # Generate unique request ID
    request_id = f"notify-{uuid.uuid4().hex[:8]}"
    request_context.set_request_id(request_id)
    
    # Log the request if tracing is enabled
    if TRACE_REQUESTS:
        logger.debug(f"Tool call received: notify with params: {params}")
    
    start_time = time.time()
    
    message = params.get("message", "")
    msg_type = params.get("type", "info")
    voice = params.get("voice", DEFAULT_VOICE)
    enhance = params.get("enhance", False)
    
    if not message:
        logger.warning("No message provided for notification")
        return {
            "success": False,
            "content": {"error": "No message provided for notification"}
        }
    
    original_message = message
    
    # Use LLM to enhance the notification if requested
    if enhance:
        try:
            logger.info(f"Enhancing notification message with LLM")
            
            # Send log message to client
            try:
                if hasattr(mcp, 'send_log_message') and callable(getattr(mcp, 'send_log_message')):
                    await mcp.send_log_message({
                        "level": "info",
                        "message": f"Enhancing notification with LLM: '{message}'"
                    })
            except Exception as e:
                logger.debug(f"Could not send log message to client: {e}")
            
            # Create a sampling request to enhance the notification
            enhanced_message = await enhance_with_llm(message, msg_type, request_id)
            if enhanced_message:
                logger.info(f"Enhanced message: '{enhanced_message}'")
                message = enhanced_message
        except Exception as e:
            logger.warning(f"Failed to enhance notification with LLM: {e}")
            logger.debug(traceback.format_exc())
            # Continue with original message if enhancement fails
    
    prefix = ""
    
    if msg_type == "success":
        prefix = "Success! "
    elif msg_type == "warning":
        prefix = "Warning! "
    elif msg_type == "error":
        prefix = "Error! "
    elif msg_type == "info":
        prefix = "Notification: "
    
    full_message = f"{prefix}{message}"
    logger.info(f"Speaking notification: '{full_message}' with voice: {voice}")
    
    success, error = await synthesize_and_play(full_message, voice, request_id)
    
    # Track performance
    duration = time.time() - start_time
    track_timing("tool_calls", "notify", duration)
    
    if success:
        logger.info(f"Successfully spoke notification (completed in {duration:.2f}s)")
        result = {
            "result": f"Successfully spoke notification: '{message}'",
            "duration_seconds": round(duration, 2)
        }
        
        # Add enhancement info if applicable
        if enhance and message != original_message:
            result["original_message"] = original_message
            result["enhanced_message"] = message
        
        return {
            "success": True,
            "content": result
        }
    else:
        logger.error(f"Failed to speak notification: {error}")
        return {
            "success": False,
            "content": {"error": error or "Failed to synthesize or play notification"}
        }


@mcp.tool
async def smart_notify(params: Dict[str, Any]):
    """Generate and speak a notification based on event context using LLM.
    
    Args:
        params: Dictionary containing:
            - event: The event to notify about
            - context: Additional context about the event (optional)
            - type: Type of notification (optional)
            - voice: Language/voice code (optional)
        
    Returns:
        Tool response indicating success or failure
    """
    # Generate unique request ID
    request_id = f"smart-notify-{uuid.uuid4().hex[:8]}"
    request_context.set_request_id(request_id)
    
    # Log the request if tracing is enabled
    if TRACE_REQUESTS:
        logger.debug(f"Tool call received: smart_notify with params: {params}")
    
    start_time = time.time()
    
    event = params.get("event", "")
    context = params.get("context", "")
    msg_type = params.get("type", "info")
    voice = params.get("voice", DEFAULT_VOICE)
    
    if not event:
        logger.warning("No event provided for smart notification")
        return {
            "success": False,
            "content": {"error": "No event provided for smart notification"}
        }
    
    try:
        # Generate notification message using LLM
        prompt = f"Event: {event}\nContext: {context}\nNotification Type: {msg_type}\n\nGenerate a clear, concise notification message for the user:"
        
        logger.info(f"Generating smart notification for event: {event}")
        
        # Send log message to client
        try:
            if hasattr(mcp, 'send_log_message') and callable(getattr(mcp, 'send_log_message')):
                await mcp.send_log_message({
                    "level": "info",
                    "message": f"Generating smart notification for event: '{event}'"
                })
        except Exception as e:
            logger.debug(f"Could not send log message to client: {e}")
        
        # Create sampling request
        sampling_response = await mcp.sample({
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "modelPreferences": {
                "temperature": 0.3,
                "maxTokens": 100
            },
            "systemPrompt": "You are a helpful assistant that creates clear, concise notification messages. Keep notifications brief and to the point."
        })
        
        if not sampling_response.success:
            error_msg = f"Failed to generate notification: {sampling_response.error}"
            logger.error(error_msg)
            return {
                "success": False,
                "content": {"error": error_msg}
            }
        
        # Extract message from sampling response
        notification_message = sampling_response.completion.strip()
        logger.info(f"Generated notification message: '{notification_message}'")
        
        # Use the generated message for notification
        prefix = ""
        if msg_type == "success":
            prefix = "Success! "
        elif msg_type == "warning":
            prefix = "Warning! "
        elif msg_type == "error":
            prefix = "Error! "
        elif msg_type == "info":
            prefix = "Notification: "
        
        # Speak the notification
        full_message = f"{prefix}{notification_message}"
        logger.info(f"Speaking smart notification: '{full_message}' with voice: {voice}")
        
        success, error = await synthesize_and_play(full_message, voice, request_id)
        
        # Track performance
        duration = time.time() - start_time
        track_timing("tool_calls", "smart_notify", duration)
        
        if success:
            logger.info(f"Successfully spoke smart notification (completed in {duration:.2f}s)")
            return {
                "success": True,
                "content": {
                    "result": "Successfully spoke smart notification",
                    "message": notification_message,
                    "event": event,
                    "duration_seconds": round(duration, 2)
                }
            }
        else:
            logger.error(f"Failed to speak notification: {error}")
            return {
                "success": False,
                "content": {"error": error or "Failed to synthesize or play notification"}
            }
    
    except Exception as e:
        error_msg = f"Error in smart_notify: {e}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        return {
            "success": False,
            "content": {"error": error_msg}
        }


@mcp.tool
async def bell(params: Dict[str, Any]):
    """Play a bell sound to alert the user.
    
    Args:
        params: Dictionary containing:
            - type: Type of bell sound (standard, success, warning, error)
            - count: Number of times to ring the bell (1-5)
        
    Returns:
        Tool response indicating success or failure
    """
    # Generate unique request ID
    request_id = f"bell-{uuid.uuid4().hex[:8]}"
    request_context.set_request_id(request_id)
    
    # Log the request if tracing is enabled
    if TRACE_REQUESTS:
        logger.debug(f"Tool call received: bell with params: {params}")
    
    start_time = time.time()
    
    bell_type = params.get("type", "standard")
    count = params.get("count", 1)
    
    # Validate count
    try:
        count = int(count)
    except (ValueError, TypeError):
        logger.warning(f"Invalid count value: {count}, using default (1)")
        count = 1
    
    # Clamp count to allowed range
    count = max(1, min(5, count))
    
    logger.info(f"Playing {bell_type} bell sound {count} time(s)")
    
    # Mock mode for testing
    if MOCK_MODE:
        logger.info(f"MOCK MODE: Would play {bell_type} bell sound {count} time(s)")
        time.sleep(0.5)  # Simulate bell sound time
        success, error = True, None
    else:
        # Generate and play bell sound
        success, error = generate_bell_sound(bell_type, count)
    
    # Track performance
    duration = time.time() - start_time
    track_timing("tool_calls", "bell", duration)
    
    if success:
        logger.info(f"Successfully played bell sound (completed in {duration:.2f}s)")
        
        # Send log message to client
        try:
            if hasattr(mcp, 'send_log_message') and callable(getattr(mcp, 'send_log_message')):
                await mcp.send_log_message({
                    "level": "info",
                    "message": f"Played {bell_type} bell sound {count} time(s)"
                })
        except Exception as e:
            logger.debug(f"Could not send log message to client: {e}")
        
        return {
            "success": True,
            "content": {
                "result": f"Successfully played {bell_type} bell sound {count} time(s)",
                "duration_seconds": round(duration, 2)
            }
        }
    else:
        logger.error(f"Failed to play bell sound: {error}")
        return {
            "success": False,
            "content": {"error": error or "Failed to generate or play bell sound"}
        }


@mcp.tool
async def test_notification(params: Dict[str, Any]):
    """Send a test notification to verify the server is working.
    
    Args:
        params: Dictionary containing:
            - voice: Language/voice code (optional)
        
    Returns:
        Tool response indicating success or failure
    """
    # Generate unique request ID
    request_id = f"test-{uuid.uuid4().hex[:8]}"
    request_context.set_request_id(request_id)
    
    voice = params.get("voice", DEFAULT_VOICE)
    
    logger.info(f"Sending test notification with voice: {voice}")
    
    # Test message
    message = "This is a test notification from the MCP text-to-speech server. If you can hear this message, the server is working correctly."
    
    success, error = await synthesize_and_play(message, voice, request_id)
    
    if success:
        # Log server statistics
        stats_summary = {
            "tool_calls": {k: v["count"] for k, v in performance_stats["tool_calls"].items()},
            "speech_synthesis": {
                "count": performance_stats["speech_synthesis"]["count"],
                "avg_time": round(performance_stats["speech_synthesis"]["avg_time"], 2) if performance_stats["speech_synthesis"]["count"] > 0 else 0
            }
        }
        
        logger.info(f"Test notification successful. Server stats: {json.dumps(stats_summary)}")
        
        return {
            "success": True,
            "content": {
                "result": "Test notification successful",
                "server_info": {
                    "version": SERVER_VERSION,
                    "transport": "HTTP" if USE_HTTP_TRANSPORT else "stdio",
                    "debug_mode": DEBUG_MODE,
                    "mock_mode": MOCK_MODE
                },
                "statistics": stats_summary
            }
        }
    else:
        logger.error(f"Test notification failed: {error}")
        return {
            "success": False,
            "content": {"error": error or "Failed to synthesize or play test notification"}
        }


@mcp.tool
async def tts_status(params: Dict[str, Any]):
    """Get information about available TTS services and current configuration.
    
    Returns:
        Tool response with TTS service status information
    """
    # Generate unique request ID
    request_id = f"status-{uuid.uuid4().hex[:8]}"
    request_context.set_request_id(request_id)
    
    logger.info("Getting TTS service status")
    
    # Check which services are available
    services = {
        "gTTS": {
            "available": True,
            "description": "Free Google Text-to-Speech via public API",
            "voice_quality": "Basic",
            "cost": "Free",
            "requires_credentials": False
        }
    }
    
    # Check Google Cloud TTS availability
    if GOOGLE_CLOUD_TTS_AVAILABLE:
        google_cloud_status = {
            "available": True,
            "description": "Premium Google Cloud Text-to-Speech API",
            "voice_quality": "High (Standard) / Premium (WaveNet)",
            "cost": "Free tier: 4M chars/month (Standard), 1M chars/month (WaveNet)",
            "requires_credentials": True,
            "configured": google_tts_client is not None,
            "voice_type": GOOGLE_CLOUD_TTS_VOICE_TYPE,
            "enabled": USE_GOOGLE_CLOUD_TTS
        }
        
        if google_tts_client:
            google_cloud_status["status"] = "Ready"
        elif USE_GOOGLE_CLOUD_TTS:
            google_cloud_status["status"] = "Enabled but not initialized (check credentials)"
        else:
            google_cloud_status["status"] = "Available but disabled"
    else:
        google_cloud_status = {
            "available": False,
            "description": "Premium Google Cloud Text-to-Speech API",
            "status": "Library not installed (pip install google-cloud-texttospeech)",
            "requires_credentials": True
        }
    
    services["Google Cloud TTS"] = google_cloud_status
    
    # Check Google Cloud STT availability
    if GOOGLE_CLOUD_STT_AVAILABLE:
        google_cloud_stt_status = {
            "available": True,
            "description": "Google Cloud Speech-to-Text API",
            "features": "Real-time speech recognition with high accuracy",
            "cost": "Free tier: 60 minutes/month",
            "requires_credentials": True,
            "configured": google_stt_client is not None,
            "language": STT_LANGUAGE_CODE,
            "enabled": USE_GOOGLE_CLOUD_STT
        }
        
        if google_stt_client:
            google_cloud_stt_status["status"] = "Ready"
        elif USE_GOOGLE_CLOUD_STT:
            google_cloud_stt_status["status"] = "Enabled but not initialized (check credentials)"
        else:
            google_cloud_stt_status["status"] = "Available but disabled"
    else:
        google_cloud_stt_status = {
            "available": False,
            "description": "Google Cloud Speech-to-Text API",
            "status": "Library not installed (pip install google-cloud-speech pyaudio)",
            "requires_credentials": True
        }
    
    services["Google Cloud STT"] = google_cloud_stt_status
    
    # Current configuration
    current_config = {
        "primary_tts_service": "Google Cloud TTS" if (google_tts_client and USE_GOOGLE_CLOUD_TTS) else "gTTS",
        "fallback_tts_service": "gTTS",
        "stt_service": "Google Cloud STT" if (google_stt_client and USE_GOOGLE_CLOUD_STT) else "Not available",
        "default_voice": DEFAULT_VOICE,
        "stt_language": STT_LANGUAGE_CODE,
        "mock_mode": MOCK_MODE,
        "debug_mode": DEBUG_MODE
    }
    
    # Environment variables guide
    env_guide = {
        "GOOGLE_APPLICATION_CREDENTIALS": "Path to Google Cloud service account JSON file",
        "MCP_TTS_USE_GOOGLE_CLOUD_STT": "Set to 'true' to enable Google Cloud Speech-to-Text",
        "MCP_TTS_STT_LANGUAGE": "Language code for speech recognition (e.g., 'en-US')",
        "MCP_TTS_USE_GOOGLE_CLOUD": "Set to 'true' to enable Google Cloud TTS",
        "MCP_TTS_GOOGLE_VOICE_TYPE": "Set to 'standard' or 'wavenet' for voice quality",
        "MCP_TTS_DEFAULT_VOICE": "Default language/voice code",
        "MCP_TTS_MOCK_MODE": "Set to 'true' for testing without actual audio"
    }
    
    return {
        "success": True,
        "content": {
            "services": services,
            "current_configuration": current_config,
            "environment_variables": env_guide,
            "server_version": SERVER_VERSION
        }
    }


@mcp.tool
async def set_timer(params: Dict[str, Any]):
    """Set a timer that will notify you with a spoken message when it expires.
    
    Args:
        params: Dictionary containing:
            - duration: Timer duration in seconds
            - message: Message to speak when timer expires
            - voice: Language/voice code (optional)
            - type: Type of notification (optional)
            - repeat: Whether to repeat the timer (optional)
            - repeat_interval: Interval between repeats (optional)
        
    Returns:
        Tool response with timer ID or error
    """
    global timer_counter
    
    # Generate unique request ID
    request_id = f"set-timer-{uuid.uuid4().hex[:8]}"
    request_context.set_request_id(request_id)
    
    # Log the request if tracing is enabled
    if TRACE_REQUESTS:
        logger.debug(f"Tool call received: set_timer with params: {params}")
    
    start_time = time.time()
    
    duration = params.get("duration")
    message = params.get("message", "")
    voice = params.get("voice", DEFAULT_VOICE)
    timer_type = params.get("type", "info")
    repeat = params.get("repeat", False)
    repeat_interval = params.get("repeat_interval")
    
    # Validate inputs
    if not duration or duration < 1 or duration > 86400:  # Max 24 hours
        logger.warning(f"Invalid duration: {duration}")
        return {
            "success": False,
            "content": {"error": "Duration must be between 1 and 86400 seconds (24 hours)"}
        }
    
    if not message:
        logger.warning("No message provided for timer")
        return {
            "success": False,
            "content": {"error": "Message is required for timer"}
        }
    
    if repeat and repeat_interval and (repeat_interval < 1 or repeat_interval > 3600):
        logger.warning(f"Invalid repeat interval: {repeat_interval}")
        return {
            "success": False,
            "content": {"error": "Repeat interval must be between 1 and 3600 seconds (1 hour)"}
        }
    
    try:
        with timer_lock:
            timer_counter += 1
            timer_id = f"timer_{timer_counter}"
        
        # Create timer object
        timer = Timer(
            timer_id=timer_id,
            duration=duration,
            message=message,
            voice=voice,
            timer_type=timer_type,
            repeat=repeat,
            repeat_interval=repeat_interval
        )
        
        # Start timer thread
        timer_thread = threading.Thread(target=timer_thread_worker, args=(timer,))
        timer_thread.daemon = True
        timer.thread = timer_thread
        
        # Store timer
        with timer_lock:
            timers[timer_id] = timer
        
        # Start the timer
        timer_thread.start()
        
        # Track performance
        duration_elapsed = time.time() - start_time
        track_timing("tool_calls", "set_timer", duration_elapsed)
        
        logger.info(f"Timer {timer_id} set for {duration} seconds: '{message}'")
        
        result = {
            "result": f"Timer set successfully",
            "timer_id": timer_id,
            "duration_seconds": duration,
            "message": message,
            "expires_at": timer.expires_at.isoformat(),
            "repeat": repeat
        }
        
        if repeat:
            result["repeat_interval"] = repeat_interval or duration
        
        return {
            "success": True,
            "content": result
        }
    
    except Exception as e:
        error_msg = f"Error setting timer: {e}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        return {
            "success": False,
            "content": {"error": error_msg}
        }


@mcp.tool
async def list_timers(params: Dict[str, Any]):
    """List all active timers.
    
    Returns:
        Tool response with list of active timers
    """
    # Generate unique request ID
    request_id = f"list-timers-{uuid.uuid4().hex[:8]}"
    request_context.set_request_id(request_id)
    
    # Log the request if tracing is enabled
    if TRACE_REQUESTS:
        logger.debug(f"Tool call received: list_timers with params: {params}")
    
    start_time = time.time()
    
    try:
        with timer_lock:
            active_timers = []
            for timer_id, timer in timers.items():
                if timer.is_active:
                    time_remaining = (timer.expires_at - datetime.now()).total_seconds()
                    active_timers.append({
                        "timer_id": timer_id,
                        "message": timer.message,
                        "expires_at": timer.expires_at.isoformat(),
                        "time_remaining_seconds": max(0, round(time_remaining, 1)),
                        "type": timer.timer_type,
                        "voice": timer.voice,
                        "repeat": timer.repeat,
                        "notification_count": timer.notification_count
                    })
        
        # Track performance
        duration = time.time() - start_time
        track_timing("tool_calls", "list_timers", duration)
        
        logger.info(f"Listed {len(active_timers)} active timers")
        
        return {
            "success": True,
            "content": {
                "result": f"Found {len(active_timers)} active timer(s)",
                "timers": active_timers,
                "count": len(active_timers)
            }
        }
    
    except Exception as e:
        error_msg = f"Error listing timers: {e}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        return {
            "success": False,
            "content": {"error": error_msg}
        }


@mcp.tool
async def cancel_timer(params: Dict[str, Any]):
    """Cancel an active timer by its ID.
    
    Args:
        params: Dictionary containing:
            - timer_id: The ID of the timer to cancel
        
    Returns:
        Tool response indicating success or failure
    """
    # Generate unique request ID
    request_id = f"cancel-timer-{uuid.uuid4().hex[:8]}"
    request_context.set_request_id(request_id)
    
    # Log the request if tracing is enabled
    if TRACE_REQUESTS:
        logger.debug(f"Tool call received: cancel_timer with params: {params}")
    
    start_time = time.time()
    
    timer_id = params.get("timer_id", "")
    
    if not timer_id:
        logger.warning("No timer_id provided for cancellation")
        return {
            "success": False,
            "content": {"error": "timer_id is required"}
        }
    
    try:
        with timer_lock:
            if timer_id not in timers:
                logger.warning(f"Timer {timer_id} not found")
                return {
                    "success": False,
                    "content": {"error": f"Timer {timer_id} not found"}
                }
            
            timer = timers[timer_id]
            timer.is_active = False
            del timers[timer_id]
        
        # Track performance
        duration = time.time() - start_time
        track_timing("tool_calls", "cancel_timer", duration)
        
        logger.info(f"Timer {timer_id} cancelled successfully")
        
        return {
            "success": True,
            "content": {
                "result": f"Timer {timer_id} cancelled successfully",
                "timer_id": timer_id
            }
        }
    
    except Exception as e:
        error_msg = f"Error cancelling timer: {e}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        return {
            "success": False,
            "content": {"error": error_msg}
        }


@mcp.tool
async def cancel_all_timers(params: Dict[str, Any]):
    """Cancel all active timers.
    
    Returns:
        Tool response indicating success or failure
    """
    # Generate unique request ID
    request_id = f"cancel-all-timers-{uuid.uuid4().hex[:8]}"
    request_context.set_request_id(request_id)
    
    # Log the request if tracing is enabled
    if TRACE_REQUESTS:
        logger.debug(f"Tool call received: cancel_all_timers with params: {params}")
    
    start_time = time.time()
    
    try:
        with timer_lock:
            cancelled_count = len(timers)
            cancelled_ids = list(timers.keys())
            
            # Mark all timers as inactive and clear the dictionary
            for timer in timers.values():
                timer.is_active = False
            timers.clear()
        
        # Track performance
        duration = time.time() - start_time
        track_timing("tool_calls", "cancel_all_timers", duration)
        
        logger.info(f"Cancelled {cancelled_count} timers: {cancelled_ids}")
        
        return {
            "success": True,
            "content": {
                "result": f"Cancelled {cancelled_count} timer(s) successfully",
                "cancelled_count": cancelled_count,
                "cancelled_timer_ids": cancelled_ids
            }
        }
    
    except Exception as e:
        error_msg = f"Error cancelling all timers: {e}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        return {
            "success": False,
            "content": {"error": error_msg}
        }


async def enhance_with_llm(message: str, msg_type: str, request_id: str = None) -> Optional[str]:
    """Enhance a notification message using LLM.
    
    Args:
        message: The original notification message
        msg_type: The type of notification
        request_id: Current request ID for logging
        
    Returns:
        Enhanced message if successful, None otherwise
    """
    # Set the request context for logging
    if request_id:
        request_context.set_request_id(request_id)
    
    try:
        prompt = f"Original notification ({msg_type}): {message}\n\nPlease improve this notification to be more natural and helpful while keeping it concise:"
        
        logger.debug(f"Sending sampling request to enhance message")
        
        # Create sampling request
        sampling_response = await mcp.sample({
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "modelPreferences": {
                "temperature": 0.3,
                "maxTokens": 100
            },
            "systemPrompt": "You are a helpful assistant that improves notification messages. Make them clear, natural, and concise."
        })
        
        if not sampling_response.success:
            logger.warning(f"Sampling failed: {sampling_response.error}")
            return None
        
        # Extract enhanced message
        enhanced_message = sampling_response.completion.strip()
        return enhanced_message
    
    except Exception as e:
        logger.error(f"Error enhancing message with LLM: {e}")
        logger.debug(traceback.format_exc())
        return None


# Add resources to the MCP server
from fastmcp.resources import TextResource

for uri, content in RESOURCES.items():
    name = uri.split("/")[-1]
    resource = TextResource(
        uri=uri,
        name=name,
        description=f"Notification template: {name}",
        text=content,
        mime_type="text/plain"
    )
    mcp.add_resource(resource)


# Configure server on connection
# @mcp.on_initialize
async def on_initialize(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle initialization request from client.
    
    This is called when the client first connects to the server.
    """
    client_protocol_version = params.get("protocolVersion", "unknown")
    logger.info(f"Client initialized connection with protocol version: {client_protocol_version}", 
                extra={"request_id": "initialize"})
    
    # Log connection details if in debug mode
    if DEBUG_MODE:
        logger.debug(f"Client initialization parameters: {json.dumps(params)}", 
                     extra={"request_id": "initialize"})
    
    # Send log message to client
    try:
        if hasattr(mcp, 'send_log_message') and callable(getattr(mcp, 'send_log_message')):
            await mcp.send_log_message({
                "level": "info",
                "message": f"TTS Notification Server v{SERVER_VERSION} initialized"
            })
    except Exception as e:
        logger.debug(f"Could not send log message to client: {e}")
    
    # Return server capabilities
    return {
        "capabilities": {
            "tools": True,
            "sampling": True,
            "roots": True,
            "resources": True
        },
        "serverInfo": {
            "name": "TTS Notification MCP Server",
            "version": SERVER_VERSION,
            "debug": DEBUG_MODE,
            "mock": MOCK_MODE
        },
        "roots": ROOTS
    }


# @mcp.on_initialized
async def on_initialized():
    """Handle initialized notification from client."""
    logger.info("Client sent initialized notification", extra={"request_id": "initialize"})
    
    # Reset request ID
    request_context.reset_request_id()


# @mcp.on_shutdown
async def on_shutdown():
    """Handle shutdown request from client."""
    logger.info("Client requested shutdown", extra={"request_id": "shutdown"})
    
    # Log performance statistics
    if DEBUG_MODE:
        logger.debug(f"Performance statistics: {json.dumps(performance_stats)}", 
                     extra={"request_id": "shutdown"})


if __name__ == "__main__":
    # Run the MCP server
    logger.info("Starting TTS Notification MCP Server", extra={"request_id": "startup"})
    logger.info(f"Version: {SERVER_VERSION}", extra={"request_id": "startup"})
    logger.info(f"Available tools: {list(TOOL_SCHEMA.keys())}", extra={"request_id": "startup"})
    logger.info(f"Transport: {'HTTP' if USE_HTTP_TRANSPORT else 'stdio'}", extra={"request_id": "startup"})
    logger.info(f"HTTP Port: {HTTP_PORT if USE_HTTP_TRANSPORT else 'N/A'}", extra={"request_id": "startup"})
    logger.info(f"Roots: {ROOTS}", extra={"request_id": "startup"})
    logger.info(f"Debug mode: {DEBUG_MODE}", extra={"request_id": "startup"})
    logger.info(f"Mock mode: {MOCK_MODE}", extra={"request_id": "startup"})
    
    # Print inspector launch command for debugging
    inspector_cmd = f"npx @modelcontextprotocol/inspector --server=\"node {os.path.abspath(__file__)}\""
    logger.info(f"To debug with Inspector, run: {inspector_cmd}", extra={"request_id": "startup"})
    
    mcp.run()