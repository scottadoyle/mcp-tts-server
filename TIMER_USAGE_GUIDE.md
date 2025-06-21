# Timer Functionality Usage Guide

The MCP TTS Server now includes comprehensive timer functionality! This allows you to set timers that will notify you with spoken messages when they expire.

## Quick Timer Examples

### Basic Timer
```python
# Set a 5-minute timer
{
    "duration": 300,
    "message": "Your 5-minute break is over!",
    "type": "info"
}
```

### Timer with Custom Voice
```python
# Set a timer with French voice
{
    "duration": 60,
    "message": "Une minute s'est écoulée",
    "voice": "fr-FR",
    "type": "success"
}
```

### Repeating Timer
```python
# Set a timer that repeats every 30 seconds
{
    "duration": 30,
    "message": "Remember to drink water",
    "type": "info",
    "repeat": True,
    "repeat_interval": 30
}
```

### Different Notification Types
```python
# Success notification
{
    "duration": 120,
    "message": "Task completed successfully",
    "type": "success"
}

# Warning notification  
{
    "duration": 60,
    "message": "Meeting starts in 1 minute",
    "type": "warning"
}

# Error notification
{
    "duration": 10,
    "message": "Critical alert",
    "type": "error"
}
```

## Available Timer Tools

### `set_timer`
Sets a new timer with the specified parameters.

**Parameters:**
- `duration` (required): Timer duration in seconds (1-86400, max 24 hours)
- `message` (required): Message to speak when timer expires
- `voice` (optional): Language/voice code (default: "en")
- `type` (optional): "info", "success", "warning", or "error"
- `repeat` (optional): Whether to repeat the timer (default: false)
- `repeat_interval` (optional): Seconds between repeats (1-3600, max 1 hour)

**Returns:**
- `timer_id`: Unique identifier for the timer
- `duration_seconds`: Timer duration
- `expires_at`: When the timer will expire (ISO format)
- `repeat`: Whether timer repeats
- `repeat_interval`: Repeat interval if applicable

### `list_timers`
Lists all currently active timers.

**Parameters:** None

**Returns:**
- `count`: Number of active timers
- `timers`: Array of timer objects with:
  - `timer_id`: Unique timer identifier
  - `message`: Timer message
  - `expires_at`: Expiration time (ISO format)
  - `time_remaining_seconds`: Seconds until expiry
  - `type`: Notification type
  - `voice`: Voice/language code
  - `repeat`: Whether timer repeats
  - `notification_count`: Number of notifications sent (for repeating timers)

### `cancel_timer`
Cancels a specific timer by its ID.

**Parameters:**
- `timer_id` (required): The ID of the timer to cancel

**Returns:**
- `timer_id`: ID of the cancelled timer

### `cancel_all_timers`
Cancels all active timers.

**Parameters:** None

**Returns:**
- `cancelled_count`: Number of timers cancelled
- `cancelled_timer_ids`: Array of cancelled timer IDs

## Common Use Cases

### Research Session Timer
```python
# 25-minute Pomodoro timer
{
    "duration": 1500,
    "message": "Pomodoro session complete! Time for a 5-minute break.",
    "type": "success"
}
```

### Meeting Reminders
```python
# 5-minute warning before meeting
{
    "duration": 300,
    "message": "Meeting starts in 5 minutes",
    "type": "warning"
}
```

### Medication Reminders
```python
# Hourly medication reminder
{
    "duration": 3600,
    "message": "Time to take your medication",
    "type": "info",
    "repeat": True,
    "repeat_interval": 3600
}
```

### Hydration Reminders
```python
# Every 30 minutes
{
    "duration": 1800,
    "message": "Remember to drink water!",
    "type": "info",
    "repeat": True,
    "repeat_interval": 1800
}
```

### Cooking Timers
```python
# Pasta cooking timer
{
    "duration": 480,
    "message": "Your pasta is ready!",
    "type": "success"
}
```

## Timer Behavior

### Single Timers
- Run once and automatically remove themselves when expired
- Speak the notification message when timer expires
- Include timer prefix: "Timer notification: [your message]"

### Repeating Timers
- Continue running until manually cancelled
- Each notification includes a counter: "Timer reminder #2: [your message]"
- Useful for ongoing reminders and recurring tasks

### Notification Types
- **info**: "Timer: [message]" - Standard blue notification sound
- **success**: "Success! [message]" - Higher pitched success sound
- **warning**: "Warning! [message]" - Lower pitched warning sound  
- **error**: "Error! [message]" - Urgent error sound

### Voice Support
Supports all languages available in Google Cloud TTS and gTTS:
- English: "en", "en-US", "en-GB", "en-AU"
- Spanish: "es", "es-ES", "es-MX"
- French: "fr", "fr-FR", "fr-CA"
- German: "de", "de-DE"
- Italian: "it", "it-IT"
- Japanese: "ja", "ja-JP"
- Chinese: "zh", "zh-CN", "zh-TW"
- And many more...

## Technical Implementation

### Threading
- Each timer runs in its own background thread
- Thread-safe timer management with locks
- Timers continue running even during other operations

### Error Handling
- Automatic cleanup of expired timers
- Graceful handling of cancellation
- Fallback error reporting

### Performance
- Minimal memory footprint per timer
- Efficient time calculation
- No polling - event-driven notifications

## Tips and Best Practices

1. **Timer Duration Limits**
   - Minimum: 1 second
   - Maximum: 86400 seconds (24 hours)
   - Repeat interval max: 3600 seconds (1 hour)

2. **Managing Multiple Timers**
   - Use descriptive messages to identify timers
   - Use `list_timers` to check what's active
   - Use `cancel_all_timers` for cleanup

3. **Voice Selection**
   - Use locale-specific voices for better pronunciation
   - Test different voices to find your preference
   - Consider the language of your timer message

4. **Notification Types**
   - Use "warning" for urgent reminders
   - Use "success" for positive completions
   - Use "error" for critical alerts
   - Use "info" for general notifications

## Example Integration

Here's how you might use timers in a research workflow:

```python
# Start a research session
set_timer({
    "duration": 1500,  # 25 minutes
    "message": "Research session complete! Time for a break.",
    "type": "success"
})

# Set hydration reminder
set_timer({
    "duration": 900,   # 15 minutes
    "message": "Hydration check - drink some water!",
    "type": "info",
    "repeat": True,
    "repeat_interval": 900
})

# Set meeting warning
set_timer({
    "duration": 1200,  # 20 minutes (5 min before 25-min session ends)
    "message": "Meeting starts in 5 minutes",
    "type": "warning"
})
```

This would give you a 25-minute focused work session with hydration reminders every 15 minutes and a meeting warning before your session ends.

The timer functionality integrates seamlessly with the existing TTS server and provides a powerful way to manage time-based notifications with spoken alerts.
