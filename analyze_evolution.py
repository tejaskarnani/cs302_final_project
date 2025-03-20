import json
import matplotlib.pyplot as plt
import numpy as np

def load_robot_logs(log_file="robot_logs.json"):
    with open(log_file, "r") as f:
        return json.load(f)

def plot_fitness_over_generations(log_data):
    generations = [g["generation"] for g in log_data]
    best_losses = [min(r["average_loss"] for r in g["robots"]) for g in log_data]
    avg_losses = [np.mean([r["average_loss"] for r in g["robots"]]) for g in log_data]

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_losses, marker='o', linestyle='-', color='b', label="Best Average Loss")
    plt.plot(generations, avg_losses, marker='o', linestyle='-', color='g', label="Population Average Loss")
    plt.xlabel("Generation")
    plt.ylabel("Average Loss")
    plt.title("Fitness Over Generations")
    plt.grid(True)
    plt.legend()
    plt.xticks(generations)
    plt.tight_layout()
    plt.show()

def plot_parameter_evolution(log_data):
    generations = [g["generation"] for g in log_data]
    best_robots = [min(g["robots"], key=lambda r: r["average_loss"]) for g in log_data]
    horizontals = [r["horizontal"] for r in best_robots]
    branchings = [r["branching"] for r in best_robots]
    stiffnesses = [r["stiffness"] for r in best_robots]

    # Create a figure with 3 subplots (one for each parameter)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot Horizontal (H)
    ax1.plot(generations, horizontals, marker='o', linestyle='-', color='b', label="Horizontal")
    ax1.set_ylabel("Horizontal Value")
    ax1.set_title("Horizontal Evolution (Best Robot per Generation)")
    ax1.grid(True)
    ax1.legend()

    # Plot Branching (B)
    ax2.plot(generations, branchings, marker='o', linestyle='-', color='g', label="Branching")
    ax2.set_ylabel("Branching Value")
    ax2.set_title("Branching Evolution (Best Robot per Generation)")
    ax2.grid(True)
    ax2.legend()

    # Plot Stiffness (S)
    ax3.plot(generations, stiffnesses, marker='o', linestyle='-', color='r', label="Stiffness")
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Stiffness Value")
    ax3.set_title("Stiffness Evolution (Best Robot per Generation)")
    ax3.grid(True)
    ax3.legend()

    plt.xticks(generations)
    plt.tight_layout()
    plt.show()

def plot_actuation_patterns(log_data, generations_to_plot=[1, 10, 20]):
    for gen in generations_to_plot:
        generation_data = next(g for g in log_data if g["generation"] == gen)
        best_robot = min(generation_data["robots"], key=lambda r: r["average_loss"])
        amps = best_robot["amps"]
        phases = best_robot["phases"]

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.bar(range(1, len(amps) + 1), amps, color='b')
        plt.xlabel("Spring Index")
        plt.ylabel("Amplitude")
        plt.title(f"Actuation Amps (Gen {gen})")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.bar(range(1, len(phases) + 1), phases, color='r')
        plt.xlabel("Spring Index")
        plt.ylabel("Phase")
        plt.title(f"Actuation Phases (Gen {gen})")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

def analyze_evolution():
    log_data = load_robot_logs()
    plot_fitness_over_generations(log_data)
    plot_parameter_evolution(log_data)
    plot_actuation_patterns(log_data)

if __name__ == "__main__":
    analyze_evolution()