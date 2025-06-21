# Integrating with Claude

This guide explains how to integrate the MCP Text-to-Speech Server with Claude to enable audible notifications.

## Prerequisites

- Claude with MCP support
- The MCP TTS Server installed and working
- Basic understanding of the Claude interface

## Setup

1. **Install the MCP TTS Server**

   Follow the installation instructions in the README.md file.

2. **Launch Claude with MCP Support**

   Launch Claude with the appropriate flags to enable MCP support:

   ```bash
   claude-code --mcp-server=./mcp_tts_server.py
   ```

   This tells Claude to use our MCP TTS server for tool operations.

## Using the Speak Tool in Claude

Claude can now use the `speak` tool to send audible notifications. Here's how it might look in a conversation:

**User:** "I need to step away for a bit to make coffee. Please analyze this data and let me know when you're done."

**Claude:** "I'll analyze the data and notify you audibly when I'm finished."

Claude will then analyze the data, and when finished, it can use the `speak` tool to notify you:

```
Claude: I've completed the analysis of your data. Let me notify you audibly.

[Claude uses the speak tool]
// The system speaks: "Notification: Data analysis is complete."
```

## Tool Options

There are two tools available in this MCP server:

1. **speak** - Basic text-to-speech
   - Parameters:
     - `text`: The text to speak (required)
     - `voice`: Language/voice code (optional, default: "en")

2. **notify** - Formatted notifications
   - Parameters:
     - `message`: The notification message (required)
     - `type`: Notification type (optional, one of: "info", "success", "warning", "error")
     - `voice`: Language/voice code (optional, default: "en")

## Troubleshooting

If Claude is unable to use the speech tools:

1. Verify the MCP server is running correctly:
   ```bash
   python test_speak.py
   ```

2. Check that Claude has the correct permissions to access the MCP server

3. Review Claude's console output for any error messages related to MCP tool access

## Security Considerations

- The MCP TTS Server has access to your computer's audio system
- It can only speak text provided to it; it cannot access other system resources
- Review tool calls before approving them in Claude's interface