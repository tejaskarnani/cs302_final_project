# CS302: Final Project

Final Project for CS302: Artificial Life at Northwestern. Evolutionary algorithm with open-loop control for rigid bodies (taichi + [difftaichi](https://github.com/yuanming-hu/difftaichi))

## Background

I started with code from the difftaichi repository, specifically `rigid_body.py`, which conducts gradient descent on 'rigid bodies' (called robots) and teaches them to locomote by setting neural network weights. The gradient descent is done with a `compute_loss` function, which measures movement towards the goal.

## Work done 
For Lab 1, I created a new robot called fork robot, and for Lab 2, I added further complexity by making a **Procedural Constructor function**, which would take in parameters and make new robots. This way, I could make a whole family of robots with different values for:
* Horizontal Width: Float, between 0.5 and 5.0
* Branching Factor (number of spring pairs): Integer, between 1 and 3 
* Spring Stiffness: Integer, between 10 and 200

Here are some of the different robots this constructor can make:
![image](https://github.com/user-attachments/assets/9c66c602-33f1-4d9a-b614-167e11551eee)

For Lab 3, I improved this to be an **Evolutionary Optimization Loop** to figure out the optimal configuration of robot parameters. There are 20 generations, with 10 robots in each generation. The first generation has robots with random configurations within the range. I ran simulations for all of them and computed their loss, selecting the best two (with minimum loss) in this generation to be the parents for the next. Then the next generation of robots is generated with paramter values between their parents'. In order to still conduct exploration and have diversity in the population, I also had a `mutation_rate`, set at 0.2, or 20%, which would randomly set one of the parameters.

I then saved all the data from the simulation: the robot parameters and the losses, to a json file called `robot_logs.json`. This is helpful as we have a large amount of data. I wrote some files to analyze this json data and produce plots like these:
![image](https://github.com/user-attachments/assets/ebfa0f78-5d78-41af-9e4f-071fd9953adb)

![image](https://github.com/user-attachments/assets/6093538d-4676-4b5e-a1b8-cc7cb5f04804)

For Lab 4, I added the element of *open-loop control* to control the spring actuation patterns. Specifically, I added the spring amplitudes and phases as parameters that could be set (floats between 0.5 and 2.0). This led to better performance overall from the robots and the evolutionary algorithm, but also made the performance of robots vary greatly. Particularly in earlier generations, some robots would move backwards, or just be unstable- like the GIF below (probably because the springs were out of phase, being randomly generated).
![not_working](https://github.com/user-attachments/assets/23c35d17-3902-40a8-86aa-6fc325985c51)


However, by the later generations, the robots learned to move, and I saw a particular, periodic motion that robots seemed to prefer across generations, seen below.

![walkinggif](https://github.com/user-attachments/assets/4e6727a8-ef80-4b3c-93ff-88c8c262dd83)


## Usage

After installing all the requirements, the code can be run with `python3 compare_robots.py`. This starts the evolutionary loop, which took me ~5 hours to run. The results are saved in the json file, `robot_logs.json`. Plots can be generated with `python3 analyze_evolution.py`, which generates plots that show the progression of robots through generations, and how their parameters change. 
In case you want to run a simulation with a robot with paramters of your choice, you can also run `python3 rigid_body.py 0 train`, then input the parameters for your robot, including the amps and phases for each spring.

## Further Work

Refine Fitness Function: Modify `compute_loss` to reward motion (total displacement) alongside distance to the goal, encouraging consistent patterns like swimming or jumping.
Add Closed-Loop Control: Implement feedback-based control (reinforcement learning) to allow robots to adapt their motion in real-time, improving robustness.
Incorporate Multi-Objective Optimization: Optimize for multiple goals, such as minimizing loss, maximizing motion efficiency, and reducing energy usage.
