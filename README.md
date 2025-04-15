# board-game-accessibility

Game-Board-Mapping directory contains the source code for the bame board system embedded backend application. 

## Setup
This repo includes a git submodule for the carddetection system, which is used to detect the cards in the game. When cloning this repo, use the following command:
```
git clone --recurse-submodules https://github.com/srikargudimella/board-game-accessibility.git
```
There are also a few other dependencies that need to be installed to run the code. These are listed in the requirements.txt file. To setup a local venv and install the dependencies, run the following command:

```
cd board-game-accessibility/Game-Board-Mapping
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the application

The application can be run from the command line with the following syntax:
```
python gamesession.py [-s | Flag to enable streaming of game state to the frontend through websockets] [optional path to config.json file]
```

```
sudo python gamesession.py -s config.json
```
If the config file isn't provided, the program will show interactive popup windows to configure the game setup. If the streaming flag is provided, the program will stream game state update messages to the websocket running at localhost:8089. To have it work with the frontend, run the frontend code in a separate process. 


## gamesession.py

GameSession class manages the main state for the game session and also includes the logic for the main game loop, turn management, piece movement, frontend communication, and game logic. 

## gameboard.py

GameBoard class manages the game board state and includes the logic for detecting the board and pieces on the board. 

## carddetection 

This git submodule includes the source code for the card detection system

## referencemap.py

This file contains the source code for the reference map generation system, which is used by gameboard.py to map features of the Candyland game board to the webcam footage.



