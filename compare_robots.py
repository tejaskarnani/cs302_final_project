import math
import taichi as ti
import numpy as np
from robot_config import build_fork_robot
import subprocess
import time
import sys
import json
import random
import os
import signal

ti.init(arch=ti.gpu)

# Global variable to handle pausing
PAUSE_FLAG = False

def signal_handler(sig, frame):
    """Handle Ctrl+C to pause the GA."""
    global PAUSE_FLAG
    print("\nCtrl+C detected. Saving progress and pausing...")
    PAUSE_FLAG = True

def run_simulation(horizontal, branching, stiffness, amps, phases):
    horizontal = max(0.5, min(5.0, horizontal))
    branching = max(1, min(3, branching))
    stiffness = max(10, min(200, stiffness))
    """Run the simulation for a given robot configuration and return average loss, individual losses, and time."""
    start_time = time.time()
    print(f"Running simulation for Robot (H={horizontal:.2f}, B={branching}, S={stiffness}, Amps={amps}, Phases={phases})")
    process = subprocess.Popen(
        ["python3", "rigid_body_auto.py", str(horizontal), str(branching), str(stiffness), ','.join(map(str, amps)), ','.join(map(str, phases)), "train"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    losses = []
    for line in process.stdout:
        sys.stdout.write(line)
        if "Loss= " in line:
            loss_value = float(line.split("Loss= ")[-1].strip())
            losses.append(loss_value)
    process.wait()
    elapsed_time = time.time() - start_time
    avg_loss = sum(losses) / len(losses) if losses else float('inf')
    print(f"Robot (H={horizontal:.2f}, B={branching}, S={stiffness}) Average Loss: {avg_loss}, Time: {elapsed_time:.2f}s")
    return avg_loss, losses, elapsed_time, amps, phases  # Return amps and phases

def compare_robots(num_generations=20, population_size=10, log_file="robot_logs.json", checkpoint_file="checkpoint.json"):
    best_robot = None
    best_avg_loss = float('inf')
    log_data = []

    # Load existing data if the file exists (though you prefer clearing)
    if os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint: {checkpoint_file}")
        with open(checkpoint_file, "r") as f:
            try:
                checkpoint = json.load(f)
                current_generation = checkpoint["current_generation"]
                population = checkpoint["population"]
                best_robot = tuple(checkpoint["best_robot"]) if checkpoint["best_robot"] else None
                best_avg_loss = checkpoint["best_avg_loss"]
                log_data = checkpoint["log_data"]
                print(f"Resuming from Generation {current_generation}...")
            except json.JSONDecodeError:
                print("Checkpoint corrupted. Starting fresh.")
                current_generation = 1
                population = [
                    {"horizontal": random.uniform(0.5, 5.0), "branching": random.randint(1, 3), "stiffness": random.randint(10, 200),
                    "amps": [random.uniform(0.0, 0.2) for _ in range(5)], "phases": [random.uniform(0.0, 2.0) for _ in range(5)]}
                    for _ in range(population_size)
                ]
    else:
        current_generation = 1
        population = [
            {"horizontal": random.uniform(0.5, 5.0), "branching": random.randint(1, 3), "stiffness": random.randint(10, 200),
            "amps": [random.uniform(0.0, 0.2) for _ in range(5)], "phases": [random.uniform(0.0, 2.0) for _ in range(5)]}
            for _ in range(population_size)
        ]

    signal.signal(signal.SIGINT, signal_handler)  # Catch Ctrl+C
    start_time = time.time()

    for generation in range(current_generation, num_generations + 1):
        print(f"Generation {generation}/{num_generations}")
        generation_data = {"generation": generation, "robots": [], "mutation_count": 0}

        fitness_scores = []
        for individual in population:
            avg_loss, losses, sim_time, amps, phases = run_simulation(individual["horizontal"], individual["branching"], individual["stiffness"], individual["amps"], individual["phases"])
            robot_data = {
                "horizontal": individual["horizontal"],
                "branching": individual["branching"],
                "stiffness": individual["stiffness"],
                "amps": amps,
                "phases": phases,
                "losses": losses[-5:] if losses else [],  # Keep last 5 losses
                "average_loss": avg_loss,
                "simulation_time": sim_time
            }
            generation_data["robots"].append(robot_data)
            fitness_scores.append((individual, avg_loss))
            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss
                best_robot = (individual["horizontal"], individual["branching"], individual["stiffness"], amps, phases)

        log_data.append(generation_data)

        # Selection: Sort by fitness (lower loss is better) and take top performers
        fitness_scores.sort(key=lambda x: x[1])
        population = [ind for ind, _ in fitness_scores]
        parents = population[:2]  # Keep top 2 as parents

        # Next generation
        next_population = parents.copy()  # Elitism: keep the best 2

        # Crossover and Mutation to fill the rest of the population
        while len(next_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            if parent1["branching"] == parent2["branching"]:
                child = {
                    "horizontal": random.uniform(parent1["horizontal"], parent2["horizontal"]),
                    "branching": parent1["branching"],  # Simplified
                    "stiffness": random.randint(min(max(10, int(parent1["stiffness"])), max(10, int(parent2["stiffness"]))), max(max(10, int(parent1["stiffness"])), max(10, int(parent2["stiffness"])))) if parent1["stiffness"] != parent2["stiffness"] else parent1["stiffness"],
                    "amps": [random.uniform(p1, p2) for p1, p2 in zip(parent1["amps"], parent2["amps"])],
                    "phases": [random.uniform(p1, p2) for p1, p2 in zip(parent1["phases"], parent2["phases"])]
                }
            else:
                child = {
                    "horizontal": random.uniform(parent1["horizontal"], parent2["horizontal"]),
                    "branching": parent1["branching"],  # Simplified
                    "stiffness": random.randint(min(max(10, int(parent1["stiffness"])), max(10, int(parent2["stiffness"]))), max(max(10, int(parent1["stiffness"])), max(10, int(parent2["stiffness"])))) if parent1["stiffness"] != parent2["stiffness"] else parent1["stiffness"],
                    "amps": [random.uniform(p1, p2) for p1, p2 in zip(parent1["amps"], parent2["amps"])],
                    "phases": [random.uniform(p1, p2) for p1, p2 in zip(parent1["phases"], parent2["phases"])]
                } 
            # Track crossover details
            child["parents"] = [(parent1["horizontal"], parent1["branching"], parent1["stiffness"], parent1["amps"], parent1["phases"]),
                            (parent2["horizontal"], parent2["branching"], parent2["stiffness"], parent2["amps"], parent2["phases"])]
            # Mutation (20% chance to randomly change a parameter)
            mutation = 0.2
            if random.random() < mutation:
                child["horizontal"] = random.uniform(0.5, 5.0)
                generation_data["mutation_count"] += 1
            if random.random() < mutation:
                child["branching"] = random.randint(1, 3)
                generation_data["mutation_count"] += 1
            if random.random() < mutation:
                child["stiffness"] = random.randint(10, 200)
                generation_data["mutation_count"] += 1
            if random.random() < mutation:
                child["amps"] = [random.uniform(0.0, 0.2) for _ in range(5)]
                generation_data["mutation_count"] += 1
            if random.random() < mutation:
                child["phases"] = [random.uniform(0.0, 2.0) for _ in range(5)]
                generation_data["mutation_count"] += 1
            child["parents"] = [(parent1["horizontal"], parent1["branching"], parent1["stiffness"], parent1["amps"], parent1["phases"]),
                        (parent2["horizontal"], parent2["branching"], parent2["stiffness"], parent2["amps"], parent2["phases"])]
            next_population.append(child)

        population = next_population[:population_size]  # Ensure population size stays constant

        if os.path.exists(log_file) and os.access(log_file, os.W_OK):
            try:
                with open(log_file, "a") as f:  # Test write access
                    f.write("")
            except IOError:
                print(f"{log_file} is locked or inaccessible. Close any editors and retry.")
                raise
        # Save logs
        print("Log data structure:", log_data)
        try:
            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=4)
        except (IOError, PermissionError) as e:
            print(f"Error writing to {log_file}: {e}")
            raise
        except TypeError as e:
            print(f"Serialization error with log_data: {e}")
            print("Log data content:", log_data)
            raise

        # Save checkpoint
        try:
            checkpoint = {
                "current_generation": generation + 1,
                "population": population,
                "best_robot": list(best_robot) if best_robot else None,
                "best_avg_loss": best_avg_loss,
                "log_data": log_data
            }
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f, indent=4)
        except (IOError, PermissionError) as e:
            print(f"Error writing to {checkpoint_file}: {e}")
            raise
        except TypeError as e:
            print(f"Serialization error with checkpoint: {e}")
            print("Checkpoint content:", checkpoint)
            raise
        
        print(f"Completed Generation {generation}, moving to {generation + 1}")

        # Check if we should pause
        if PAUSE_FLAG:
            print(f"Paused at Generation {generation}. Resume by running the script again.")
            sys.exit(0)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Best performing robot: Robot {best_robot} with final loss {best_avg_loss}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Logs saved to {log_file}")

    # Clear checkpoint file upon completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

if __name__ == "__main__":
    compare_robots()