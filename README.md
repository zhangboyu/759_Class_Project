# 759_Class_Project
This is our 759 final project repo.

Team member: Boyu Zhang, Ying Li

Note: This code requires compute capability 6.0 devices.

How to build: Just type "make". The program itself is named as "collision_detection".

How to use: three ways:

1). On a machine with GPU card, type "./collision_detection DIM(2/3) OBJ_NUMBER SPACE_SCALING_FACTOR". (SPACE_SCALING_FACTOR should not be larger than 0.7)

2). On a machine with GPU card, type "./run_collision_detection.sh" to run the program with some predefined arguments.

3). On Euler, type "sbatch run_collision_detection.sh" to run the program with some predefined arguments.

After the program finish, a file named "collision_pairs.txt" will be generated, which contains the index of collide object pairs.
