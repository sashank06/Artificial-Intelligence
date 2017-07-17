# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?  
A: Constraint propagation is applied to the naked twins problem as another means of reducing the sudoku puzzle. This is done by identifying all the pairs of twins that share the same set of values. When twins share the same set of values this indicates that other unsolved boxes in the peers cannot use these values and these twin values can be eliminated from the other digits of their peers
    the following piece of code below helps in identifying the twins
/* code starts
possible_twins = {box:values[box] for box in unit if len(values[box]) == 2}
#find all the naked twins from the possible twins list and we use set operation to make sure the values are equal in a string
naked_twin = [[box1,box2] for box1 in possible_twins.keys() for box2 in possible_twins.keys() if (box1!=box2 and set(possible_twins[box1]) == set(possible_twins[box2]))] */code ends
By reducing the possiblitiy of values that are available we apply constraint propagation to the naked twins problem
# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?  
A: Constraint Progpagation can be applied to solve the diagonal sudoku problem by introducing the the 2 main diagonals to the unit list. By adding the two main diagonals to the unit list, it adds another constraint to the solve the problem and also helps in reducing the possible values for a box.

### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 
Please try using the environment we provided in the Anaconda lesson of the Nanodegree.

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Code

* `solution.py` - You'll fill this in as part of your solution.
* `solution_test.py` - Do not modify this. You can test your solution by running `python solution_test.py`.
* `PySudoku.py` - Do not modify this. This is code for visualizing your solution.
* `visualize.py` - Do not modify this. This is code for visualizing your solution.

### Visualizing

To visualize your solution, please only assign values to the values_dict using the `assign_value` function provided in solution.py

### Submission
Before submitting your solution to a reviewer, you are required to submit your project to Udacity's Project Assistant, which will provide some initial feedback.  

The setup is simple.  If you have not installed the client tool already, then you may do so with the command `pip install udacity-pa`.  

To submit your code to the project assistant, run `udacity submit` from within the top-level directory of this project.  You will be prompted for a username and password.  If you login using google or facebook, visit [this link](https://project-assistant.udacity.com/auth_tokens/jwt_login) for alternate login instructions.

This process will create a zipfile in your top-level directory named sudoku-<id>.zip.  This is the file that you should submit to the Udacity reviews system.

