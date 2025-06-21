"""Test script for the MCP TTS server.

This script simulates what would happen when an LLM like Claude uses the speak tool.
It allows testing the MCP server without needing to configure Claude.
"""

import os
import subprocess
import sys
import time
import json
import argparse
import uuid

def main():
    """Run a test of the MCP TTS server."""
    parser = argparse.ArgumentParser(description='Test the MCP TTS server')
    parser.add_argument('--tool', type=str, default='test_notification', 
                        choices=['speak', 'notify', 'smart_notify', 'test_notification', 'bell'],
                        help='Tool to test (default: test_notification)')
    parser.add_argument('--text', type=str, 
                        default='Hello! This is a test of the MCP text to speech server.',
                        help='Text to speak (for speak tool)')
    parser.add_argument('--message', type=str, 
                        default='Task has been completed successfully.',
                        help='Message to notify (for notify tool)')
    parser.add_argument('--event', type=str, 
                        default='task_complete',
                        help='Event to notify about (for smart_notify tool)')
    parser.add_argument('--type', type=str, default='info',
                        choices=['info', 'success', 'warning', 'error'],
                        help='Notification type (for notify and smart_notify tools)')
    parser.add_argument('--voice', type=str, default='en',
                        help='Voice/language code')
    parser.add_argument('--enhance', action='store_true',
                        help='Enhance notification with LLM (for notify tool)')
    parser.add_argument('--context', type=str, default='',
                        help='Event context (for smart_notify tool)')
    parser.add_argument('--bell-type', type=str, default='standard',
                        choices=['standard', 'success', 'warning', 'error'],
                        help='Type of bell sound (for bell tool)')
    parser.add_argument('--bell-count', type=int, default=1,
                        help='Number of times to ring the bell (1-5, for bell tool)')
    parser.add_argument('--mock', action='store_true',
                        help='Run server in mock mode (no audio)')
    parser.add_argument('--debug', action='store_true',
                        help='Run server in debug mode')
    
    args = parser.parse_args()
    
    # Prepare tool parameters based on selected tool
    tool_params = {}
    if args.tool == 'speak':
        tool_params = {
            "text": args.text,
            "voice": args.voice
        }
    elif args.tool == 'notify':
        tool_params = {
            "message": args.message,
            "type": args.type,
            "voice": args.voice,
            "enhance": args.enhance
        }
    elif args.tool == 'smart_notify':
        tool_params = {
            "event": args.event,
            "context": args.context,
            "type": args.type,
            "voice": args.voice
        }
    elif args.tool == 'bell':
        tool_params = {
            "type": args.bell_type,
            "count": args.bell_count
        }
    elif args.tool == 'test_notification':
        tool_params = {
            "voice": args.voice
        }
    
    # Simulate an MCP tool call
    tool_call = {
        "method": "tools/execute",
        "params": {
            "name": args.tool,
            "params": tool_params
        }
    }
    
    # Generate a unique request ID
    request_id = f"test-{uuid.uuid4().hex[:8]}"
    
    # Convert to JSON-RPC format
    jsonrpc_request = json.dumps({
        "jsonrpc": "2.0",
        "id": request_id,
        **tool_call
    })
    
    print(f"Testing MCP TTS server with tool: {args.tool}")
    print(f"Parameters: {json.dumps(tool_params, indent=2)}")
    
    # Set environment variables for the server
    env = os.environ.copy()
    if args.mock:
        env["MCP_TTS_MOCK_MODE"] = "true"
        print("Running in MOCK MODE (no audio)")
    if args.debug:
        env["MCP_TTS_DEBUG"] = "true"
        print("Running in DEBUG MODE")
    
    # Since we're using stdio transport, we need to start the server in a separate process
    # and communicate with it through stdin/stdout
    server_process = subprocess.Popen(
        ["python", "mcp_tts_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    
    # Wait for server to initialize
    print("\nStarting MCP server...")
    time.sleep(2)
    
    # Send the initialization message
    init_message = json.dumps({
        "jsonrpc": "2.0",
        "id": "init-1",
        "method": "initialize",
        "params": {
            "protocolVersion": "0.1.0"
        }
    })
    
    print("\nSending initialization request...")
    server_process.stdin.write(init_message + "\n")
    server_process.stdin.flush()
    
    # Read and parse initialization response
    response_line = server_process.stdout.readline()
    init_response = json.loads(response_line)
    print(f"Server capabilities: {json.dumps(init_response.get('result', {}).get('capabilities', {}), indent=2)}")
    
    # Send initialized notification
    init_notification = json.dumps({
        "jsonrpc": "2.0",
        "method": "initialized",
        "params": {}
    })
    
    print("\nSending initialized notification...")
    server_process.stdin.write(init_notification + "\n")
    server_process.stdin.flush()
    
    # Send the tool call
    print(f"\nSending {args.tool} tool request...")
    server_process.stdin.write(jsonrpc_request + "\n")
    server_process.stdin.flush()
    
    # Read and parse tool response
    response_line = server_process.stdout.readline()
    tool_response = json.loads(response_line)
    
    # Print formatted response
    print("\nTool response:")
    if "result" in tool_response and "content" in tool_response["result"]:
        content = tool_response["result"]["content"]
        print(json.dumps(content, indent=2))
    else:
        print(json.dumps(tool_response, indent=2))
    
    # Allow time for speech to complete (if not in mock mode)
    if not args.mock:
        print("\nWaiting for speech to complete...")
        time.sleep(5)
    
    # Send shutdown request
    shutdown_request = json.dumps({
        "jsonrpc": "2.0",
        "id": "shutdown-1",
        "method": "shutdown",
        "params": {}
    })
    
    print("\nSending shutdown request...")
    server_process.stdin.write(shutdown_request + "\n")
    server_process.stdin.flush()
    
    # Read shutdown response
    response_line = server_process.stdout.readline()
    
    # Clean up
    print("\nTerminating MCP server...")
    server_process.terminate()
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()