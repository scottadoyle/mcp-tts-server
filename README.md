# MCP Text-to-Speech Notification Server

An MCP (Model Context Protocol) server that provides text-to-speech capabilities for AI assistants to speak notifications audibly to humans.

## Features

- Implements the Model Context Protocol (MCP)
- **Dual TTS support**: 
  - **gTTS** (default): Free, no credentials required
  - **Google Cloud TTS** (optional): Premium voices, requires setup
- **Timer functionality**: Set timers with spoken notifications when they expire
- **Repeating timers**: Support for recurring notifications at specified intervals
- Automatic fallback from Google Cloud TTS to gTTS
- Supports multiple languages and voices
- Designed for AI assistants to notify users when not actively watching the screen
- Supports multiple transport methods (stdio, streamable HTTP)
- Implements LLM sampling for enhanced notification messages
- Defines roots for resource boundaries
- Includes smart notification generation
- Comprehensive logging and debugging capabilities
- Compatible with MCP Inspector
- Performance tracking and metrics

## Quick Start

```bash
# Clone the repository
git clone https://github.com/scottadoyle/mcp-tts-server.git
cd mcp-tts-server

# Install dependencies
pip install -r requirements.txt

# Ready to use with default gTTS!
# No configuration needed for basic functionality
```

## TTS Services

### gTTS (Default)
- ✅ **Free and anonymous**
- ✅ **No credentials required**
- ✅ **Works immediately**
- ⚠️ Basic voice quality
- ⚠️ Limited voice options

### Google Cloud TTS (Optional)
- ✅ **Premium voice quality**
- ✅ **380+ voices in 50+ languages**
- ✅ **WaveNet AI voices**
- ✅ **Generous free tier** (1-4M chars/month)
- ⚠️ **Requires Google Cloud setup**
- ⚠️ **Requires billing account**
- ⚠️ **Uses your credentials**

➡️ **See [GOOGLE_CLOUD_SETUP.md](GOOGLE_CLOUD_SETUP.md) for optional premium setup**

## Usage

### Starting the Server

```bash
# Run the MCP server with stdio transport (default)
python mcp_tts_server.py

# Or run with HTTP transport
MCP_TTS_HTTP_TRANSPORT=true python mcp_tts_server.py

# Run in debug mode
MCP_TTS_DEBUG=true python mcp_tts_server.py

# Run in mock mode (for testing without audio)
MCP_TTS_MOCK_MODE=true python mcp_tts_server.py
```

### Tool Capabilities

This MCP server provides the following tools:

1. `speak` - Converts text to speech and plays it audibly
   - Parameters:
     - `text` (required): The text to speak
     - `voice` (optional): Language/voice code (default: "en")

2. `notify` - Speaks a notification with a standardized format
   - Parameters:
     - `message` (required): The notification message
     - `type` (optional): Type of notification ("info", "success", "warning", "error")
     - `voice` (optional): Language/voice code (default: "en")
     - `enhance` (optional): Use LLM to enhance the notification text (default: false)

3. `bell` - Plays a bell sound to alert the user without speaking text
   - Parameters:
     - `type` (optional): Type of bell sound ("standard", "success", "warning", "error")
     - `count` (optional): Number of times to ring the bell (1-5, default: 1)

4. `smart_notify` - Generate and speak a notification based on event context
   - Parameters:
     - `event` (required): The event to notify about (e.g., 'task_complete', 'error')
     - `context` (optional): Additional context about the event
     - `type` (optional): Type of notification (default: "info")
     - `voice` (optional): Language/voice code (default: "en")

5. `test_notification` - Send a test notification to verify the server is working
   - Parameters:
     - `voice` (optional): Language/voice code (default: "en")

6. `set_timer` - Set a timer that will notify you with a spoken message when it expires
   - Parameters:
     - `duration` (required): Timer duration in seconds (1-86400)
     - `message` (required): Message to speak when timer expires
     - `voice` (optional): Language/voice code (default: "en")
     - `type` (optional): Type of notification ("info", "success", "warning", "error")
     - `repeat` (optional): Whether to repeat the timer notification (default: false)
     - `repeat_interval` (optional): Interval between repeat notifications in seconds (1-3600)

7. `list_timers` - List all active timers
   - Returns information about all currently running timers including time remaining

8. `cancel_timer` - Cancel an active timer by its ID
   - Parameters:
     - `timer_id` (required): The ID of the timer to cancel

9. `cancel_all_timers` - Cancel all active timers
   - Removes all currently active timers

10. `listen` - Listen for speech input and convert it to text using speech-to-text
    - Parameters:
      - `duration` (optional): Duration to listen in seconds (default: 5)
      - `language` (optional): Language code for speech recognition (default: "en-US")

11. `tts_status` - Get information about available TTS services and current configuration
    - Returns detailed status of TTS services and server configuration

### Timer Usage Examples

The timer functionality allows you to set reminders that will speak notifications when they expire:

```python
# Set a simple 5-minute timer
await set_timer({
    "duration": 300,  # 5 minutes
    "message": "Your 5-minute break is over!",
    "type": "info"
})

# Set a repeating reminder every 30 seconds
await set_timer({
    "duration": 30,
    "message": "Remember to drink water",
    "type": "info",
    "repeat": True,
    "repeat_interval": 30
})

# Set a timer with a specific voice
await set_timer({
    "duration": 60,
    "message": "Une minute s'est écoulée",
    "voice": "fr-FR",
    "type": "success"
})
```

## Advanced Features

### LLM Sampling

The server uses LLM sampling to:
1. Enhance notification messages when requested
2. Generate smart notifications based on event context

This allows the notifications to be more natural and context-aware.

### Multiple Transport Methods

The server supports two transport methods:
- `stdio`: Standard input/output transport (default)
- `HTTP`: Streamable HTTP transport with server-sent events

### Resources

The server provides example notification templates as resources that can be accessed by clients. These resources are located under the notification root URI.

### Debugging & Monitoring

The server includes comprehensive debugging features:
- Request ID tracking in logs
- Performance metrics for tool calls and speech synthesis
- Mock mode for testing without audio
- Detailed logging with configurable levels
- Client-side log messages via MCP
- Support for MCP Inspector for interactive testing

## Integration with Claude

To use this MCP server with Claude:

1. Start the MCP server:
   ```bash
   python mcp_tts_server.py
   ```

2. Configure Claude to use the MCP server using the appropriate flags or settings in your Claude integration.

3. Claude can then use the tools to audibly notify the user.

## Configuration

You can configure the server by setting the following environment variables:

- `MCP_TTS_DEFAULT_VOICE`: Default voice/language to use (default: "en")
- `MCP_TTS_LOG_LEVEL`: Logging level (default: "INFO")
- `MCP_TTS_HTTP_TRANSPORT`: Use HTTP transport (default: "false")
- `MCP_TTS_HTTP_PORT`: Port for HTTP transport (default: "8080")
- `MCP_TTS_ROOT`: Root URI for notifications (default: "file:///notifications/tts")
- `MCP_TTS_DEBUG`: Enable debug mode (default: "false")
- `MCP_TTS_TRACE_REQUESTS`: Log all request parameters (default: "false")
- `MCP_TTS_MOCK_MODE`: Run in mock mode without audio (default: "false")

## Testing

### Included Test Script

To test the server, you can use the included test script:

```bash
python test_speak.py
```

This will simulate a client connection and request the server to speak a test message.

### Using MCP Inspector

For interactive testing and debugging, you can use the MCP Inspector:

```bash
# Install the inspector (if you haven't already)
npm install -g @modelcontextprotocol/inspector

# Run the inspector with the server
npx @modelcontextprotocol/inspector --server="python mcp_tts_server.py"
```

The Inspector provides a graphical interface for exploring the server's capabilities, testing tools, and viewing logs.

### Test Notification Tool

You can also use the built-in `test_notification` tool to verify the server is working:

```bash
# This will appear in the Inspector's tool list
```

## License

MIT