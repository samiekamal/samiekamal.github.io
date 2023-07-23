---
classes: wide
header:
  overlay_image: /images/shiny-bot-images/kanto.png

title: Shiny Hunting Bot
toc: true
toc_label: "Overview"

---

<style type="text/css">
body {
  font-size: 13pt;
}

/* pre {
  background-color: white;
}

code {
  background-color: white;
} */
</style>

## Background

I've always been a huge Pokémon fan and still play the games to this day. However, I've always had trouble finding shiny Pokémon. A shiny Pokémon is a rare variant of a Pokémon that has different colors than other Pokémon of their species. 

It is a 1/4096 to get one, which means it could take days or even months if you're unlucky to get one. I do not have time to find the Pokémon I want and battle it, then reset the game and repeat that process until I get the shiny variant. So, I developed a bot that can do it for me.

## Libraries Needed

When choosing a language to program this bot, Python was my first choice due to it having an easy library of bottling tools. We can first import the following libraries. We will create a file called ShinyBot.py to store all our methods.

~~~ python
from pyautogui import *
import pyautogui
import time
import win32gui
~~~

[PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/) is a Python library that allows you to programmatically control the mouse cursor's movement, simulate mouse clicks and scrolling, as well as perform keyboard actions such as typing and key combinations. This is exactly what we need to create the bot. 

The [time](https://docs.python.org/3/library/time.html) library allows us to add any delays we may need in the bot.

And finally, we import [win32gui](https://pypi.org/project/win32gui/) which is a library that allows me to manipulate windows.

## Methods Required

Now, with these libraries, we can create methods we need our bot to perform. We first want to always make sure our Pokémon game is always at a certain point on the screen so the bot will know where to look.

~~~ python
def moveWindow(windowName, posX, posY):
    """
    Pass the name of the window and the window coordinate destination
    EX: moveWindow("Pokémon Insurgence", 0, 0) 

    Args:
        windowName: name of the window you want to move
        posX: x-cord
        posY: y-cord
    """
    window = win32gui.FindWindow(None, windowName)
    windowRect = win32gui.GetWindowRect(window)
    x = windowRect[0]
    y = windowRect[1]
    w = windowRect[2] - x
    h = windowRect[3] - y
    win32gui.MoveWindow(window, posX, posY, w, h, True)
~~~

This method will tell the bot where to move our Pokémon game window. Typically a good location would be (`[0,0]`) which is the top left of your monitor, but you can set it wherever just as long as no other window is obstructing it. 

We then make another method called keyPress which will tell the bot what keyboard keys to press in the game in order to enter battles. 

~~~ python
def keyPress(keyboardKey, interval):
    """
    Pass the keyboardKey for input and then the delay after it
    EX: keyPress('a', 1)

    Args:
        keyboardKey: input key
        interval: delay after the input
    """
    pyautogui.keyDown(keyboardKey)
    time.sleep(0.1)
    pyautogui.keyUp(keyboardKey)
    time.sleep(interval)
~~~

***Note, it is important to use time.sleep so our bot holds down a key for 0.1 seconds then releases the key.**

Next, we need another method in order to perform shortcuts. In a Pokémon game, we can use the shortcut 'ctrl r' to go back to the main menu and load the game to see if we got the shiny.

~~~ python
def keyPressShortCut(keyboardKey1, keyBoardKey2, interval):
    """
    Pass keyboardKey1 and keyBoardKey2 to perform a shortcut input and then the delay after it
    EX: keyPress('ctrl', 'r', 1)

    Args:
        keyboardKey1: first input key
        keyboardKey2: second input key
        interval: delay after both inputs
    """
    pyautogui.keyDown(keyboardKey1)
    time.sleep(0.1)
    pyautogui.keyDown(keyBoardKey2)
    time.sleep(0.1)
    pyautogui.keyUp(keyboardKey1)
    pyautogui.keyUp(keyBoardKey2)
    time.sleep(interval)
~~~

Now we can perform any shortcut on our keyboard.

Finally, we need to tell the bot how to check if a Pokémon is shiny. 

~~~ python
def checkShiny(posX, posY, tPokemonRGB):
    """
    Pass the coordinates of where to check for shiny and the NON-shiny RGB
    EX: boolean variable = checkShiny(512, 454, (115, 73, 105))

    Args: 
        posX: x-coord
        posY: y-coord
        tPokemonRGB: pixel color of normal Pokémon
    """

    if(pyautogui.pixel(posX,posY) != (tPokemonRGB)):
        bShiny = True
        print(bShiny)
    else:
        bShiny = False
        print(bShiny)

    return bShiny
~~~

We need to tell the coordinates of what pixel to look at and the normal color of a Pokémon which will be a tuple with the RGB values. This method will return a boolean saying true or false if the Pokémon is shiny.

## The Set Up

Now that we have all our methods ready, we actually need to set up the bot. We should first get the RGB value of the Pokémon we want to shiny hunt. In order to figure out the RGB value and window coordinates for the bot to check, open the Python *IDLE Shell and do the following steps:

* Open your Pokémon game and battle the Pokémon you want to shiny hunt.
* In the Python IDLE Shell, type import pyautogui
* Next, type pyautogui.displayMousePosition()
* Move your mouse over any pixel on the Pokémon and the Python IDLE Shell will return a tuple with the RGB values of the pixel and the coordinates.
* Then press 'ctrl c' to stop it from printing your RGB values.

![](/images/shiny-bot-images/find_rgb.gif)

Now that we have our pixel coordinates and pixel RGB values, we can finally create a script for the bot. We will call this file RenegadeShines.py since the game I'm displaying is called Pokémon Renegade Platinum (a ROM hack of Pokémon Platinum).

Let’s first import the libraries we need.
~~~ python
from pyautogui import *
import time
import ShinyBot
~~~

Next, let's finally code the bot and give it the pixel coordinates and pixel RGB values as arguments.

~~~ python
# this bot is meant to be executed immediately when you open the game.
def shinyHunt(posX, posY, pixelRGB):
    """
    1 iteration is approximately 30 seconds
    """
    shinyFound = False
    count = 0
    
    time.sleep(11) # 11-second delay, so the start-up screens gets to the main menu before the bot starts pressing any buttons.
    ShinyBot.moveWindow("DeSmuME 0.9.14 git#91efef9 x64-JIT SSE2 | Pokémon Platinum",0,0)

~~~

* We create a boolean at the top called shinyFound which will be false by default.
* Create a count integer that stores how many iterations this bot has done.
* Set a delay of 11 seconds since that's how long it takes to go to the main menu when in the game.
* Tell the bot to move the name of our game window to (`[0,0]`).

Let's then create a main loop to run this bot until a shiny Pokémon is found.

~~~ python

 # number of times the bot will hit 'x' to start a battle with the Pokémon you are shiny hunting
    while(not shinyFound):
        ShinyBot.keyPress('x', 2)
        ShinyBot.keyPress('x',2.5) 
        ShinyBot.keyPress('x',2)
        ShinyBot.keyPress('x',1)
        ShinyBot.keyPress('x',1)

        time.sleep(9) # 9-second delay to transition to battle once you encounter the Pokémon.
        shinyFound = ShinyBot.checkShiny(posX, posY, pixelRGB)
        
        if(shinyFound): 
            break # loop ends once the bot looks at the pixel coordinates if it's != pixelRGB

        # displays the number of times the bot has to go back to the main menu and repeat this whole process
        count+=1
        print(count) 
        ShinyBot.keyPressShortCut('ctrl','r',11) 

~~~

* We create a while loop that will run until the shiny Pokémon is found.
* Tell the bot to keep pressing the key 'x' with certain delays so it can load into the game properly.
* Tell it to wait 9 seconds for the animation to play of entering a battle.
* Check if the pixel and coordinates we passed in are shiny pixels.
* If not, this loop will repeat and increment our counter and perform a shortcut key to go back to the main menu until the shiny Pokémon is found.

Finally, we need to call this method and pass in our pixel coordinates and RGB values.

~~~ python
# Main
shinyHunt(410,243, (174,223,101))

~~~

Now, the bot is complete and is ready to start shiny hunting for you. When you run the bot, make sure you are tabbed into your game window and no other window is obstructing it. 

## Demonstration

***Note, the footage is sped up.**

![](/images/shiny-bot-images/shiny_false.gif)

This bot will continue to print the output and the number of iterations in the terminal until the shiny Pokémon is found.

# Finding a shiny Pokémon

![](/images/shiny-bot-images/shiny_found.gif)

The bot has successfully identified if a Pokémon is shiny or not. This bot has saved me hours building up my shiny Pokémon collection. 

In the game I was playing, Pokémon Renegade Platinum, it took the bot 2 months to shiny hunt every legendary Pokémon in the game. Imagine sitting at your desktop doing this yourself manually. Would be very boring and a big waste of time. 

## The Full Code

This is all the code together for the bot how it appears on my [ShinyBot](https://github.com/samikamal21/Shiny-Bot) repository:

# ShinyBot.py

~~~ python
from pyautogui import *
import pyautogui
import time
import win32gui

def moveWindow(windowName, posX, posY):
    """
    Pass the name of the window and the window coordinate destination
    EX: moveWindow("Pokémon Insurgence", 0, 0) 

    Args:
        windowName: name of the window you want to move
        posX: x-cord
        posY: y-cord
    """
    window = win32gui.FindWindow(None, windowName)
    windowRect = win32gui.GetWindowRect(window)
    x = windowRect[0]
    y = windowRect[1]
    w = windowRect[2] - x
    h = windowRect[3] - y
    win32gui.MoveWindow(window, posX, posY, w, h, True)

def keyPress(keyboardKey, interval):
    """
    Pass the keyboardKey for input and then the delay after it
    EX: keyPress('a', 1)

    Args:
        keyboardKey: input key
        interval: delay after the input
    """
    pyautogui.keyDown(keyboardKey)
    time.sleep(0.1)
    pyautogui.keyUp(keyboardKey)
    time.sleep(interval)

def keyPressShortCut(keyboardKey1, keyBoardKey2, interval):
    """
    Pass keyboardKey1 and keyBoardKey2 to perform a shortcut input and then the delay after it
    EX: keyPress('ctrl', 'r', 1)

    Args:
        keyboardKey1: first input key
        keyboardKey2: second input key
        interval: delay after both inputs
    """
    pyautogui.keyDown(keyboardKey1)
    time.sleep(0.1)
    pyautogui.keyDown(keyBoardKey2)
    time.sleep(0.1)
    pyautogui.keyUp(keyboardKey1)
    pyautogui.keyUp(keyBoardKey2)
    time.sleep(interval)

def checkShiny(posX, posY, tPokemonRGB):
    """
    Pass the coordinates of where to check for shiny and the NON-shiny RGB
    EX: boolean variable = checkShiny(512, 454, (115, 73, 105))

    Args: 
        posX: x-coord
        posY: y-coord
        tPokemonRGB: pixel color of normal Pokémon
    """

    if(pyautogui.pixel(posX,posY) != (tPokemonRGB)):
        bShiny = True
        print(bShiny)
    else:
        bShiny = False
        print(bShiny)

    return bShiny

~~~

# RenegadeShines.py

~~~ python
from pyautogui import *
import time
import ShinyBot

# this bot is meant to be executed immediately when you open the game.
def shinyHunt(posX, posY, pixelRGB):
    """
    1 iteration is approximately 30 seconds
    """
    shinyFound = False
    count = 0
    
    time.sleep(11) # 11-second delay, so the start-up screens gets to the main menu before the bot starts pressing any buttons.
    ShinyBot.moveWindow("DeSmuME 0.9.14 git#91efef9 x64-JIT SSE2 | Pokémon Platinum",0,0)

    # number of times the bot will hit 'x' to start a battle with the Pokemon you are shiny hunting
    while(not shinyFound):
        ShinyBot.keyPress('x', 2)
        ShinyBot.keyPress('x',2.5) 
        ShinyBot.keyPress('x',2)
        ShinyBot.keyPress('x',1)
        ShinyBot.keyPress('x',1)

        time.sleep(9) # 9-second delay to transition to battle once you encounter the Pokemon.
        shinyFound = ShinyBot.checkShiny(posX, posY, pixelRGB)
        
        if(shinyFound): 
            break # loop ends once the bot looks at the pixel coordinates if it's != pixelRGB

        # displays the number of times the bot has to go back to the main menu and repeat this whole process
        count+=1
        print(count) 
        ShinyBot.keyPressShortCut('ctrl','r',11) 

# Main
shinyHunt(348,202, (60,101,182))

~~~

