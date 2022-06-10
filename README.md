# Dynamic_Maze_Solver_RL_COMP6247

# Running the code

To set up the python libraries needed to run this project, simply run the command:

``pip install -r requirements.txt``

Then to run the code use:

``python main.py``

# What should you expect after running the code

A Qtable.pt file is supplied with the code, so if you will just run the code, it will solve the maze and create a ``moves.txt`` file describing every move made up to solution and showing the around cells of the maze at every move. An already generated ``moves.txt`` file is also provided in the ``results`` folder, same one that was used in the coursework submission.

# Maintainability

If you wanted to retrain the Qtable from scratch you just have to uncomment the lines 244 and 245, then a new Qtable will be created after 100 episodes of training. The process is quite lengthy so keep that in mind. After a new ``Qtable.pt`` file will be created it will replace the provided file and the algorithm will use the new file to solve the maze.

Overall the code is well-commented so it is easy to figure out how to update it. If you wanted to train with more episodes or wanted to train with fire, just edit the parameters of the explore_maze method.


 
