#include <iostream>
#include <fstream>
#include <vector>

#define NUM_WALKERS 5000

struct State {
    std::vector<State*> neighbours;
    std::vector<double> transition_probabilities;

    int num_walkers;
    int num_incoming;
};

double g(double x, double dx) {
    if (std::abs(x - 0.5) < dx) {
        return 1;
    }

    return 0;
}

double u(double t, double x, std::vector<State> &states, double dx, double dt, double prev_t=0) {
    // Clear the number of walkers in each state
    if (prev_t == 0) {
        for (State &state : states) {
            state.num_walkers = 0;
            state.num_incoming = 0;
        }
        // Start the walkers at the initial position
        int init_x = (int)(x / dx);
        states[init_x].num_walkers = NUM_WALKERS;
    }

    // Simulate the walkers
    int num_timesteps = (int)((t - prev_t) / dt);

    for (int timestep = 0; timestep < num_timesteps; timestep++) {
        for (int i = 0; i < states.size(); i++) {
            for (int w = 0; w < states[i].num_walkers; w++) {
                double r = (double) rand() / RAND_MAX;
                double p = 0;
                for (int k = 0; k < states[i].neighbours.size(); k++) {
                    p += states[i].transition_probabilities[k];
                    if (r < p) {
                        states[i].num_walkers--;
                        states[i].neighbours[k]->num_incoming++;
                        break;
                    }
                }
            }
        }

        // Update the number of walkers in each state
        for (State &state : states) {
            state.num_walkers += state.num_incoming;
            state.num_incoming = 0;
        }
    }

    double sum = 0;
    for (int i = 0; i < states.size(); i++) {
        sum += states[i].num_walkers * g(i * dx, dx);
    }

    return sum / NUM_WALKERS;
}

int main(int argc, char *argv[]) {
    // The user passes the filename as a command-line argument
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    // Load the model parameters from file
    std::string filename = argv[1];
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << argv[1] << std::endl;
        return 1;
    }

    // Read the model parameters
    // The first line contains the length (in metres) of the domain
    double length;
    file >> length;

    // The second line contains the number of states
    int num_states;
    file >> num_states;

    // The third line contains dx
    double dx;
    file >> dx;

    // The fourth line contains dt
    double dt;
    file >> dt;

    // The remaining num_states line contains the transition probabilities
    // Create the states
    std::vector<State> states(num_states);
    for (int i = 0; i < num_states; i++) {
        // left
        double p_left;
        file >> p_left;
        if (i > 0) {
            states[i].neighbours.push_back(&states[i - 1]);
            states[i].transition_probabilities.push_back(p_left);
        }

        // right
        double p_right;
        file >> p_right;
        if (i < num_states - 1) {
            states[i].neighbours.push_back(&states[i + 1]);
            states[i].transition_probabilities.push_back(p_right);
        }

        // stay
        double p_stay;
        file >> p_stay;
        states[i].neighbours.push_back(&states[i]);
        states[i].transition_probabilities.push_back(p_stay);
    }

    // Close the file
    file.close();

    // Solve the PDE
    double t = 60;
    int num_steps = (int)(t / dt);

    std::vector<std::vector<double>> u_values(num_steps, std::vector<double>(num_states));
    for (int i = 0; i < num_states; i++) {
        std::cerr << i << " / " << num_states << "\r";
        for (int k = 0; k < num_steps; k++) {
            u_values[k][i] = u(k * dt, i * dx, states, dx, dt, k == 0 ? 0 : (k - 1) * dt);
        }
    }

    // Output the solution
    for (int k = 0; k < num_steps; k++) {
        for (int i = 0; i < num_states; i++) {
            std::cout << u_values[k][i] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
