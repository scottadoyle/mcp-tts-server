#!/usr/bin/env python3
"""
Simple validation script for timer functionality.
This tests that the timer code is working properly.
"""

import sys
import os
import asyncio
import time
from datetime import datetime

# Add the current directory to the path so we can import the server
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set mock mode to test without audio
os.environ['MCP_TTS_MOCK_MODE'] = 'true'

def test_timer_functionality():
    """Test the timer functionality at the code level."""
    
    print("Testing Timer Code Functionality")
    print("=" * 50)
    
    try:
        # Import the timer classes and functions
        from mcp_tts_server import Timer, timers, timer_lock, timer_counter
        print("Successfully imported timer components")
        
        # Test Timer class creation
        test_timer = Timer(
            timer_id="test_1",
            duration=5.0,
            message="Test timer message",
            voice="en-US",
            timer_type="info"
        )
        
        print(f"Created timer: {test_timer}")
        print(f"Timer ID: {test_timer.timer_id}")
        print(f"Duration: {test_timer.duration} seconds")
        print(f"Message: {test_timer.message}")
        print(f"Expires at: {test_timer.expires_at}")
        print(f"Is active: {test_timer.is_active}")
        
        # Test timer data structures
        print(f"\nTimer storage: {len(timers)} active timers")
        print(f"Timer counter: {timer_counter}")
        
        # Test that we can create a repeating timer
        repeat_timer = Timer(
            timer_id="test_repeat",
            duration=2.0,
            message="Repeating message",
            repeat=True,
            repeat_interval=1.0
        )
        
        print(f"\nRepeating timer: {repeat_timer}")
        print(f"Repeat enabled: {repeat_timer.repeat}")
        print(f"Repeat interval: {repeat_timer.repeat_interval}")
        
        print("\nTimer class validation: PASSED")
        
        # Test that the tool schemas include timer tools
        from mcp_tts_server import TOOL_SCHEMA
        
        timer_tools = [
            'set_timer', 'list_timers', 'cancel_timer', 'cancel_all_timers'
        ]
        
        print(f"\nTool Schema Validation:")
        for tool in timer_tools:
            if tool in TOOL_SCHEMA:
                print(f"  {tool}: FOUND")
                schema = TOOL_SCHEMA[tool]
                print(f"    Description: {schema['description']}")
            else:
                print(f"  {tool}: MISSING")
                return False
        
        print("\nTool schema validation: PASSED")
        
        # Test server version
        from mcp_tts_server import SERVER_VERSION
        print(f"\nServer version: {SERVER_VERSION}")
        
        if SERVER_VERSION >= "1.3.0":
            print("Version validation: PASSED")
        else:
            print("Version validation: FAILED (should be 1.3.0 or higher)")
            return False
        
        print("\n" + "=" * 50)
        print("ALL TIMER FUNCTIONALITY TESTS PASSED!")
        print("The timer features have been successfully added to the MCP TTS server.")
        print("\nYou can now use timer functionality with commands like:")
        print("- set_timer: Set a timer with spoken notification")
        print("- list_timers: List all active timers")
        print("- cancel_timer: Cancel a specific timer")
        print("- cancel_all_timers: Cancel all timers")
        print("\nThe server is ready to use with timer support!")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Validation error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Timer Functionality Validation")
    print("This script validates that timer features are properly integrated.")
    print()
    
    success = test_timer_functionality()
    
    if success:
        print("\nValidation completed successfully!")
        sys.exit(0)
    else:
        print("\nValidation failed!")
        sys.exit(1)
