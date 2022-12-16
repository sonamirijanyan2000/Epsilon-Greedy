from packages import *
from EpsGreedy import EpsilonGreedy
from Bernoulli import BernoulliArm


def test_algorithm(alg, arms, num_sims, horizon):
    chosen_arms = np.zeros((num_sims, horizon))
    rewards = np.zeros((num_sims, horizon))
    for sim in range(num_sims):
        alg.initialize(len(arms))

        for m in range(horizon):
            chosen_arm = alg.select_arm()
            chosen_arms[sim, m] = chosen_arm

            reward = arms[chosen_arm].draw()
            rewards[sim, m] = reward

            alg.update(chosen_arm, reward)

    avg_rewards = np.mean(rewards, axis=1)
    cum_rewards = np.cumsum(avg_rewards)

    return chosen_arms, avg_rewards, cum_rewards


"""
Plot the performance of the Bandit algorithms
"""
ALGORITHMS = {
    "epsilon-Greedy": EpsilonGreedy
}


def plot_algorithm(
        algorithm_name="epsilon-Greedy", arms=None, best_arm_index=None,
        hyper_parameters=None, num_sims=1000, horizon=100, label=None,
        fig_size=(21, 7)):
    # Check if the algorithm doesn't have hyperparameter
    if hyper_parameters is None:
        # Run the algorithm
        algorithm = ALGORITHMS[algorithm_name]()
        chosen_arms, avg_rewards, cum_rewards = test_algorithm(
            algorithm, arms, num_sims, horizon)
        probabilities_average = np.where(chosen_arms == best_arm_index, 1, 0).sum(
            axis=0) / num_sims

        # Plot the 3 metrics of the algorithm
        fig, axes = plt.subplots(1, 3, figsize=fig_size)
        axes[0].plot(probabilities_average)
        axes[0].set_xlabel("Time", fontsize=12)
        axes[0].set_ylabel("The probability of choosing the best arm is", fontsize=12)
        axes[0].set_title(
            f"The accuracy of {algorithm_name} alg.", y=1.07, fontsize=17)
        axes[0].set_ylim([0, 1.07])
        axes[1].plot(avg_rewards)
        axes[1].set_xlabel("Time", fontsize=15)
        axes[1].set_ylabel("Average Reward", fontsize=15)
        axes[1].set_title(
            f"The average rewards of {algorithm_name} alg. is", y=1.07, fontsize=17)
        axes[1].set_ylim([0, 1.0])
        axes[2].plot(cum_rewards)
        axes[2].set_xlabel("Time", fontsize=15)
        axes[2].set_ylabel("The cumulative rewards of chosen arm is", fontsize=15)
        axes[2].set_title(
            f" The cumulative rewards of {algorithm_name} alg. is", y=1.07, fontsize=17)
        plt.tight_layout()

    else:
        fig, axes = plt.subplots(1, 3, figsize=fig_size)
        for hyper_parameter in hyper_parameters:
            # Running algorithm
            algorithm = ALGORITHMS[algorithm_name](hyper_parameter)
            chosen_arms, avg_rewards, cum_rewards = test_algorithm(
                algorithm, arms, num_sims, horizon)
            probabilities_average = np.where(chosen_arms == best_arm_index, 1, 0).sum(
                axis=0) / num_sims

            # Plotting three metrics of algorithm
            axes[0].plot(probabilities_average, label=f"{label} = {hyper_parameter}")
            axes[0].set_xlabel("Time", fontsize=15)
            axes[0].set_ylabel(
                "The probability on selecting the best arm is", fontsize=15)
            axes[0].set_title(
                f"The accuracy of {algorithm_name} alg. is ", y=1.07, fontsize=17)
            axes[0].legend()
            axes[0].set_ylim([0, 1.07])
            axes[1].plot(avg_rewards, label=f"{label} = {hyper_parameter}")
            axes[1].set_xlabel("Time", fontsize=15)
            axes[1].set_ylabel("Average Reward", fontsize=15)
            axes[1].set_title(
                f"The average rewards of {algorithm_name} alg. is", y=1.07, fontsize=15)
            axes[1].legend()
            axes[1].set_ylim([0, 1.0])
            axes[2].plot(cum_rewards, label=f"{label} = {hyper_parameter}")
            axes[2].set_xlabel("Time", fontsize=15)
            axes[2].set_ylabel("The cumulative rewards of the chosen arm ", fontsize=15)
            axes[2].set_title(
                f"The cumulative rewards of the {algorithm_name} alg. is", y=1.07, fontsize=17)
            axes[2].legend(loc="lower right")
            plt.tight_layout()

        axes[0].plot(probabilities_average, label=algorithm.__ne__())
        axes[0].set_xlabel("Time", fontsize=12)
        axes[0].set_ylabel("The probability of choosing the best arm ", fontsize=15)
        axes[0].set_title(
            f"The accuracy of different algorithms ", y=1.07, fontsize=15)
        axes[0].set_ylim([0, 1.07])
        axes[0].legend(loc="lower right")
        axes[1].plot(avg_rewards, label=algorithm.__ne__())
        axes[1].set_xlabel("Time", fontsize=12)
        axes[1].set_ylabel("Average Reward", fontsize=12)
        axes[1].set_title(
            f"The average rewards of different algorithms ", y=1.07, fontsize=15)
        axes[1].set_ylim([0, 1.0])
        axes[1].legend(loc="lower right")
        axes[2].plot(cum_rewards, label=algorithm.__ne__())
        axes[2].set_xlabel("Time", fontsize=12)
        axes[2].set_ylabel("The cumulative rewards of the chosen arm", fontsize=15)
        axes[2].set_title(
            f"The cumulative rewards of the different algorithms", y=1.07, fontsize=15)
        axes[2].legend(loc="lower right")
        plt.tight_layout()
        plt.show()


np.random.seed(1)
means = [0.1, 0.1, 0.1, 0.1, 0.9]
n_arms = len(means)
np.random.shuffle(means)
arms = list(map(lambda mu: BernoulliArm(mu), means))
print("Best arm is " + str(means.index(max(means))))

best_arm_index = np.argmax(means) # Getting best arm index
epsilon = [0.01, 0.1, 0.2, 0.5] # Checking for different epsilon values

# Plot the epsilon-Greedy algorithm
plot_algorithm(algorithm_name="epsilon-Greedy", arms=arms, best_arm_index=best_arm_index,
               hyper_parameters=epsilon, num_sims=500, horizon=500, label="eps")