#!/usr/bin/env python3
"""
Test script for the MCP TTS Server timer functionality.

This script demonstrates how to use the new timer features.
"""

import asyncio
import json
import sys
import time
from datetime import datetime

async def test_timer_functionality():
    """Test the timer functionality by simulating MCP tool calls."""
    
    print("Testing Timer Functionality")
    print("=" * 50)
    
    # Import the MCP server
    try:
        from mcp_tts_server import set_timer, list_timers, cancel_timer, cancel_all_timers
        print("Successfully imported timer functions")
    except ImportError as e:
        print(f"Failed to import timer functions: {e}")
        return
    
    # Test 1: Set a simple timer
    print("\nTest 1: Setting a 5-second timer")
    timer_params = {
        "duration": 5,
        "message": "Your 5-second timer has expired!",
        "voice": "en-US",
        "type": "info"
    }
    
    result = await set_timer(timer_params)
    if result["success"]:
        timer_id = result["content"]["timer_id"]
        print(f"Timer set successfully: {timer_id}")
        print(f"   Duration: {result['content']['duration_seconds']} seconds")
        print(f"   Message: {result['content']['message']}")
        print(f"   Expires at: {result['content']['expires_at']}")
    else:
        print(f"Failed to set timer: {result['content']['error']}")
        return
    
    # Test 2: List active timers
    print("\nTest 2: Listing active timers")
    result = await list_timers({})
    if result["success"]:
        print(f"Found {result['content']['count']} active timer(s)")
        for timer in result["content"]["timers"]:
            print(f"   Timer ID: {timer['timer_id']}")
            print(f"   Message: {timer['message']}")
            print(f"   Time remaining: {timer['time_remaining_seconds']} seconds")
    else:
        print(f"Failed to list timers: {result['content']['error']}")
    
    # Test 3: Set a repeating timer
    print("\nTest 3: Setting a repeating timer (3 times)")
    repeat_timer_params = {
        "duration": 3,
        "message": "This is a repeating reminder!",
        "voice": "en-US",
        "type": "warning",
        "repeat": True,
        "repeat_interval": 2
    }
    
    result = await set_timer(repeat_timer_params)
    if result["success"]:
        repeat_timer_id = result["content"]["timer_id"]
        print(f"Repeating timer set: {repeat_timer_id}")
        print(f"   Will repeat every {result['content']['repeat_interval']} seconds")
    else:
        print(f"Failed to set repeating timer: {result['content']['error']}")
    
    # Wait a bit and check timers again
    print("\nWaiting 2 seconds...")
    await asyncio.sleep(2)
    
    # Test 4: List timers again
    print("\nTest 4: Listing timers after wait")
    result = await list_timers({})
    if result["success"]:
        print(f"Found {result['content']['count']} active timer(s)")
        for timer in result["content"]["timers"]:
            print(f"   Timer ID: {timer['timer_id']}")
            print(f"   Time remaining: {timer['time_remaining_seconds']} seconds")
    
    # Test 5: Cancel a specific timer
    print(f"\nTest 5: Cancelling timer {timer_id}")
    result = await cancel_timer({"timer_id": timer_id})
    if result["success"]:
        print(f"Timer cancelled successfully")
    else:
        print(f"Failed to cancel timer: {result['content']['error']}")
    
    # Wait for the repeating timer to trigger a few times
    print("\nWaiting 8 seconds to see repeat timer notifications...")
    await asyncio.sleep(8)
    
    # Test 6: Cancel all remaining timers
    print("\nTest 6: Cancelling all remaining timers")
    result = await cancel_all_timers({})
    if result["success"]:
        print(f"Cancelled {result['content']['cancelled_count']} timer(s)")
        if result['content']['cancelled_timer_ids']:
            print(f"   Cancelled IDs: {result['content']['cancelled_timer_ids']}")
    else:
        print(f"Failed to cancel all timers: {result['content']['error']}")
    
    # Final check
    print("\nFinal check: Listing remaining timers")
    result = await list_timers({})
    if result["success"]:
        print(f"{result['content']['count']} timer(s) remaining")
    
    print("\nTimer functionality test completed!")


if __name__ == "__main__":
    print("Timer Test Script")
    print("This script tests the timer functionality of the MCP TTS Server")
    print("\nNote: Audio playback requires proper audio setup.")
    print("Set MCP_TTS_MOCK_MODE=true to test without audio.\n")
    
    # Run the test
    try:
        asyncio.run(test_timer_functionality())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
