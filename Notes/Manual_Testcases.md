#Presenter tests

## Text cursor placement

### 1 Single click event on a letter
  - Move cursor over a letter
  - Click left mouse button
  - *Blinking text cursor should appear after the letter*

### 2 Single click event after the text in line
  - Move cursor over a white space after the text in the line
  - Click left mouse button
  - *Blinking text cursor should appear after the last letter on the same line*

### 3 Single click event in an empty line after the text
  - Move cursor to an empty line after the text
  - Click left mouse button
  - *Blinking text cursor should appear after the last character of the Text*

### 4 Single click event before the first symbol
  - Move cursor right before the text
  - Click left mouse button
  - *Blinking text cursor should appear before the first character*

### 5 Single click event outside the page
  - Move cursor outside of the page
  - Click left mouse button
  - *Cursor shouldn't appear*

## Text selection drawing

### 6 Select one letter in the middle of the text
  - Move cursor over a letter in the middle of the text
  - Click and hold left mouse button
  - Move mouse one character away (either direction)
  - Release mouse button
  - *A character should be highlighted in blue*
  - Press Shift+C on the keyboard
  - Click left mouse button at the end of the text
  - Press Shift+V on the keyboard
  - *Only the character that was selected should appear*

### 7 Select text over multiple lines
  - Move cursor over the last word in one of the lines (except the last one)
  - Click and hold left mouse button
  - Move mouse to the first word of the next line
  - Release mouse button
  - *The highlight should extend from the selection point to the end of the
    line and from the beginning of the next line to the end point*
  - Press Shift+C on the keyboard
  - Click left mouse button at the end of the text
  - Press Enter and then Shift+V on the keyboard
  - *The selected text should appear without the line break*

### 8 Select text until very end
  - Move cursor anywhere on the text
  - Click and hold left mouse button
  - Move mouse to the empty area below the text
  - Release mouse button
  - *The highlight should extend from the selection point to the end point*
  - Press Shift+C on the keyboard
  - Click left mouse button at the end of the text
  - Press Enter and then Shift+V on the keyboard
  - *The selected text should appear with its own line breaks without additional
    white space at the end*

### 9 Select text until very beginning
  - Move cursor anywhere on the text
  - Click and hold left mouse button
  - Move mouse to the area above the first line of the text
  - Release mouse button
  - *The highlight should extend from the selection point to the first letter
    of the text*
  - Press Shift+C on the keyboard
  - Click left mouse button at the end of the text
  - Press Enter and then Shift+V on the keyboard
  - *The selected text should appear with its own line breaks without additional
    white space at the beginning*

### 10 Select empty area without text
  - Copy some text form another application
  - Move cursor below to the empty area below the text
  - Click and hold left mouse button
  - Move mouse to a different point of the empty area below the text
  - Release moues button
  - *The highlight should extend from the selection point to the end point*
  - Press Shift+C on the keyboard
  - Click left mouse button at the end of the text
  - Press Enter and then Shift+V on the keyboard
  - *Text copied from another application should appear*

## Copy and Paste actions

### 11 Copy text on the same line
  - Select any part of the text that is on the same line
  - Press Shift+C on the keyboard
  - Go to the notepad application
  - Press Ctrl+V on the keyboard
  - *The selected text should appear*

### 12 Copy text on the different lines
  - Select any part of the text that spans multiple lines
  - Press Shift+C on the keyboard
  - Paste the text into the notepad application (Ctrl+V)
  - *The selected text should appear without line breaks*

### 13 Paste one word
  - Copy one word from any other application with text (Ctrl+C)
  - Run text editor component
  - Click left mouse button at the end of the text
  - Press Shift+V on the keyboard
  - *The copied text should appear*

### 14 Paste longer paragraph
  - Copy longer paragraph from any other application with text (Ctrl+C)
  - Run text editor component
  - Click left mouse button at the end of the text
  - Press Shift+V on the keyboard
  - *The copied text should appear and have lines breaks to fit in the plane*

### 15 Paste text with line breaks
  - Copy text with line breaks from any other application (Ctrl+C)
  - Run text editor component
  - Click left mouse button at the end of the text
  - Press Shift+V on the keyboard
  - *The copied text should appear with the original line breaks and might
    have more line breaks to fit within the plane*

## Undo and Redo actions

### 16 Undo one change
  - Change something in the text (one action)
  - Press Shift+Z on the keyboard
  - *The change should disappear and the cursor should be at the end of the
    text*

### 17 Undo less than 10 changes
  - Change something in the text 5 times
  - Press Shift+Z on the keyboard 5 times
  - *The text should go to the original state and the cursor should be at the
    end of the text*

### 18 Undo more than 10 changes
  - Change something in the text 3 times
  - Note what the text looks like
  - Change something in the text 9 more times
  - Press Shift+Z on the keyboard 12 times
  - *The text should be as noted after the 3 changes and the cursor should be
    at the end of the text*

### 19 Redo one change
  - Change something in the text and remember the change
  - Change something in the text again
  - Press Shift+Z on the keyboard
  - *The change should disappear and the cursor should be at the end of the
    text*
  - Press Shift+Y on the keyboard
  - *The text should be the same as it was after the first change and the
    cursor is at the end of the text*

### 20 Redo less than 10 changes
  - Change something in the text 3 times
  - Note what the text looks like
  - Press Shift+Z on the keyboard 3 times
  - *The text should go to the original state and the cursor should be at the
    end of the text*
  - Press Shift+Y on the keyboard 3 times
  - *The text should be as noted after the 3 changes and the cursor should be
    at the end of the text*

### 21 Redo 10 changes
  - Change something in the text 3 times
  - Note what the text looks like
  - Change something in the text 9 more times
  - Note what the text looks like again
  - Press Shift+Z on the keyboard 12 times
  - *The text should be as noted after the 3 changes and the cursor should be
    at the end of the text*
  - Press Shift+Y on the keyboard 15 times
  - *The text should be as noted after the 12 changes and the cursor should be
    at the end of the text*

## Selected text deletion

### 22 Delete selected text
  - Copy any piece of text from anywhere
  - Select big part of one line of text
  - Press backspace on the keyboard
  - *The selected text should disappear and the cursor should be at the point
    where the selected text was. Line wrapping should be done on the new
    text nicely*
  - Click left mouse button at the end of the text
  - Press Shift+V on the keyboard
  - *The text that appears should be the one that was copied and not the one
    that was deleted*

## Entering text

### 23 Enter various characters
  - Click left mouse button anywhere on the text
  - Press random characters on the keyboard
  - *Each character should appear in the text where after the location of the
    cursor before the press*
  - Press enter on the keyboard
  - *Cursor and any text after the cursor should move to the new line*

### 24 Deletion of characters
  - Click left moues button anywhere on the text
  - Press backspace on the keyboard
  - *The cursor should move back one character and the character that was
    before the cursor should disappear*

## Gesture controls

### 25-47 Repeat tests 1-22 using gesture controls
  - Moving mouse cursor and clicking left mouse button -> touching the plane
    with index finger
  - Click, hold and drag, release -> touch the plane with the hand in a gun
    formation, move the hand parallel to the plane while holding the same
    formation, tap the thumb or move the hand away from the plane
  - Shift+C -> form the hand into mirrored C shape
  - Shift+V -> form V with index and middle fingers (other fingers bent)
  - Shift+Z -> with only the index finger extended, do a full anticlockwise
    circle by rotating the wrist
  - Shift+Y -> with only the index finger extended, do a full clockwise circle
    by rotating the wrist

### 48 Gesture selection confirmation
  - Start selecting text (touch in formation and drag)
  - Tap the thumb with the index finger still touching the plane
  - *The selection should be confirmed and the frame should be semitransparent*

## Gesture recognition framework

### 49 Gesture while touching the plane
  - Copy some text (from anywhere)
  - Touch the plane with index finger
  - Do the paste gesture while touching the plane
  - *Nothing should happen*

## Changing focus

### 50 Remove focus from a plane
  - The gaze hits a plane
  - Rotate the camera to remove gaze from the plane
  - *The frame of the plane should disappear*
  - Do any editing action
  - *Nothing should happen*
  - Attempt to touch the plane
  - *The hand should simply pass through without any unresponsiveness from the
    plane*

### 51 Bring focus to the plane
  - The gaze is not focused on any plane
  - Rotate the camera to bring focus to the plane
  - *A semi transparent frame should appear*
  - Do any editing action
  - *The action should be executed*
  - Try to touch the plane
  - *The frame should become opaque and the cursor should be placed where the
    finger hits the plane*

### 52 Switching focus
  - The gaze is on one plane
  - Select some text
  - Do a copy action
  - Rotate the camera to bring focus to another plane
  - *A semi transparent frame of first plane should disappear and appear around
    the second plane should*
  - Do a paste action
  - *The text should be inserted at the end of the second plane text*
  - Try to touch the first plane
  - *It should be unresponsive*
  - Touch the second plane
  - *The cursor should move to where the index finger touched the plane*

### 53 Multiple switches
  - The gaze is on one plane
  - Bring it to another plane
  - *A semi transparent frame of first plane should disappear and appear around
    the second plane should*
  - Bring the gaze away
  - *The frame of the second plane should disappear*
  - Bring the gaze back to the first plane
  - *A semi transparent frame should appear on the first plane*
  - Bring the gaze to the second plane
  - *A semi transparent frame of first plane should disappear and appear around
    the second plane should*
  - Try to touch the first plane
  - *It should be unresponsive*
  - Touch the second plane
  - *The cursor should move to where the index finger touched the plane*

## Feedback

### 54 Single gesture action
  - Wait until no feedback shown
  - Do any gesture action
  - *A popup with the name of the action should appear for 0.6s*

### 55 Two successive quick gesture action
  - Wait until no feedback shown
  - Place the cursor at the end of the text
  - Do delete gesture twice very quickly (in less than 0.5s)
  - *Only one character should be deleted*

### 56 Two successive quick keyboard actions
  - Wait until no feedback shown
  - Place the cursor at the end of the text
  - Press backspace twice very quickly (in less than 0.5s)
  - *Two characters should be deleted*

### 57 Two successive slow gesture action
  - Wait until no feedback shown
  - Place the cursor at the end of the text
  - Do a delete gesture
  - Wait 1s
  - Do a delete gesture again
  - *Two characters should be deleted*

### 58 Different gesture actions
  - Wait until no feedback shown
  - Do a copy gesture
  - Very quickly do a delete gesture (in less than 0.5s after copy)
  - *No characters should be deleted*
